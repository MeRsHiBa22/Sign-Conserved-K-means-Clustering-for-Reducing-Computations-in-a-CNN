import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Uint8ActPerTensorFloat

# ----------------------------- config -----------------------------
W_BITS = 8
A_BITS = 8
EPOCHS = 10
BATCH = 128
LR = 1e-3
K = 3                              # clusters per 3x3 kernel
CLUSTER_LAYERS = ["conv2"]         # alternate layers: conv1 original, conv2 clustered
SEED = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s=SEED):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ----------------------------- models -----------------------------
class FloatLeNet(nn.Module):
    """conv3x3(1->6) - pool - conv3x3(6->16) - pool - fc 400-120-84-10"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QuantLeNet(nn.Module):
    """Same architecture, INT8 weights + INT8 activations (Brevitas)."""

    def __init__(self):
        super().__init__()
        self.quant_in = QuantIdentity(act_quant=Int8ActPerTensorFloat, bit_width=A_BITS)
        self.conv1 = QuantConv2d(1, 6, 3, weight_quant=Int8WeightPerTensorFloat, weight_bit_width=W_BITS)
        self.relu1 = QuantReLU(act_quant=Uint8ActPerTensorFloat, bit_width=A_BITS)
        self.conv2 = QuantConv2d(6, 16, 3, weight_quant=Int8WeightPerTensorFloat, weight_bit_width=W_BITS)
        self.relu2 = QuantReLU(act_quant=Uint8ActPerTensorFloat, bit_width=A_BITS)
        self.fc1 = QuantLinear(400, 120, bias=True, weight_quant=Int8WeightPerTensorFloat, weight_bit_width=W_BITS)
        self.relu3 = QuantReLU(act_quant=Uint8ActPerTensorFloat, bit_width=A_BITS)
        self.fc2 = QuantLinear(120, 84, bias=True, weight_quant=Int8WeightPerTensorFloat, weight_bit_width=W_BITS)
        self.relu4 = QuantReLU(act_quant=Uint8ActPerTensorFloat, bit_width=A_BITS)
        self.fc3 = QuantLinear(84, 10, bias=True, weight_quant=Int8WeightPerTensorFloat, weight_bit_width=W_BITS)

    def forward(self, x):
        x = self.quant_in(x)
        x = F.max_pool2d(self.relu1(self.conv1(x)), 2)
        x = F.max_pool2d(self.relu2(self.conv2(x)), 2)
        x = x.flatten(1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        return self.fc3(x)


def load_float_weights(qmodel, fmodel):
    """Copy FP32 weights/biases into the Brevitas model (activation scales start fresh)."""
    with torch.no_grad():
        for n in ["conv1", "conv2", "fc1", "fc2", "fc3"]:
            getattr(qmodel, n).weight.copy_(getattr(fmodel, n).weight)
            getattr(qmodel, n).bias.copy_(getattr(fmodel, n).bias)


# ----------------------------- data / train / eval -----------------------------
def get_loaders():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    te = datasets.MNIST("./data", train=False, download=True, transform=tf)
    return (DataLoader(tr, batch_size=BATCH, shuffle=True),
            DataLoader(te, batch_size=1000, shuffle=False))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
    return 100.0 * correct / len(loader.dataset)


def train(model, train_loader, test_loader, tag, post_step=None):
    print(f"\n--- Training: {tag} ---")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            if post_step is not None:
                post_step(model)
            total += loss.item() * x.size(0)
        print(f"[{tag}] epoch {ep}/{EPOCHS} | loss {total/len(train_loader.dataset):.4f} "
              f"| test acc {evaluate(model, test_loader):.2f}%")


# ----------------------------- sign-conserved k-means -----------------------------
@torch.no_grad()
def sign_conserved_kmeans(weight, k=K, iters=100):
    """
    weight: (O, I, kh, kw). Each (kh x kw) kernel is clustered independently.
    Signs are kept; k-means (1-D) runs on magnitudes only; every magnitude is
    replaced by its cluster centroid; the original sign is re-applied.
    Returns clustered weight, cluster assignments (N, kh*kw), signs (N, kh*kw).
    """
    O, I, kh, kw = weight.shape
    flat = weight.reshape(O * I, kh * kw)
    sign = torch.where(flat >= 0, torch.ones_like(flat), -torch.ones_like(flat))
    mag = flat.abs()

    # deterministic init: min / middle / max of the sorted magnitudes of each kernel
    idx = torch.linspace(0, kh * kw - 1, k).round().long()
    cent = mag.sort(dim=1).values[:, idx]                        # (N, k)

    for _ in range(iters):
        assign = (mag.unsqueeze(2) - cent.unsqueeze(1)).abs().argmin(2)   # (N, 9)
        onehot = F.one_hot(assign, k).to(mag.dtype)                       # (N, 9, k)
        counts = onehot.sum(1)                                            # (N, k)
        sums = (onehot * mag.unsqueeze(2)).sum(1)                         # (N, k)
        new = torch.where(counts > 0, sums / counts.clamp(min=1), cent)   # empty cluster keeps old centroid
        if torch.allclose(new, cent, atol=1e-10, rtol=0):
            cent = new
            break
        cent = new

    assign = (mag.unsqueeze(2) - cent.unsqueeze(1)).abs().argmin(2)
    clustered = sign * cent.gather(1, assign)
    return clustered.reshape_as(weight), assign, sign


@torch.no_grad()
def project_to_clusters(weight, assign, sign, k=K):
    """Keep sign + cluster assignment fixed; set each cluster's magnitude to the mean magnitude of its members."""
    N = assign.shape[0]
    flat = weight.reshape(N, -1)
    mag = (sign * flat).clamp(min=0)
    onehot = F.one_hot(assign, k).to(flat.dtype)
    cent = (onehot * mag.unsqueeze(2)).sum(1) / onehot.sum(1).clamp(min=1)
    weight.copy_((sign * cent.gather(1, assign)).reshape_as(weight))


@torch.no_grad()
def kernel_magnitude_stats(w):
    """w: (O, I, 3, 3) tensor (float or int). Returns (max, mean) number of distinct |values| per kernel."""
    flat = w.reshape(-1, w.shape[-1] * w.shape[-2]).abs()
    u = torch.tensor([r.unique().numel() for r in flat], dtype=torch.float32)
    return int(u.max().item()), u.mean().item()


@torch.no_grad()
def report_quantised_conv(model, name):
    qw = getattr(model, name).quant_weight()
    w_int = qw.int().cpu()
    mx, mean = kernel_magnitude_stats(w_int)
    print(f"  [{name}] INT{int(qw.bit_width)} | scale={qw.scale.flatten()[0].item():.6f} "
          f"| int range [{w_int.min().item()}, {w_int.max().item()}] "
          f"| distinct |int| per kernel: max={mx}, mean={mean:.2f}")
    print(f"  [{name}] kernel (0,0) int values:\n{w_int[0, 0]}")


# ----------------------------- main -----------------------------
if __name__ == "__main__":
    train_loader, test_loader = get_loaders()
    results = {}

    # 1) FP32 baseline
    set_seed()
    fp = FloatLeNet().to(device)
    train(fp, train_loader, test_loader, "FP32 baseline")
    results["FP32 baseline"] = evaluate(fp, test_loader)

    # 2) INT8 QAT from scratch (reference)
    set_seed()
    q_ref = QuantLeNet().to(device)
    train(q_ref, train_loader, test_loader, "INT8 QAT (no clustering)")
    results["INT8 QAT, no clustering"] = evaluate(q_ref, test_loader)

    # 3) Sign-conserved k-means on the trained FP32 baseline (no retraining)
    fp_cl = copy.deepcopy(fp)
    cluster_info = {}
    print(f"\n--- Sign-conserved k-means (k={K}) on layers: {CLUSTER_LAYERS} ---")
    for name in CLUSTER_LAYERS:
        layer = getattr(fp_cl, name)
        cw, assign, sign = sign_conserved_kmeans(layer.weight.data, K)
        layer.weight.data.copy_(cw)
        cluster_info[name] = (assign, sign)
        mx, mean = kernel_magnitude_stats(layer.weight.data)
        print(f"  [{name}] distinct |w| per kernel after clustering: max={mx}, mean={mean:.2f}")
    results[f"FP32 clustered (k={K}), no retraining"] = evaluate(fp_cl, test_loader)
    print(f"Clustered FP32 test accuracy: {results[f'FP32 clustered (k={K}), no retraining']:.2f}%")

    # 4a) INT8 QAT initialised from clustered weights (free fine-tuning)
    set_seed()
    q_free = QuantLeNet().to(device)
    load_float_weights(q_free, fp_cl)
    train(q_free, train_loader, test_loader, "INT8 QAT from clustered init (free)")
    results["INT8 QAT, clustered init (free)"] = evaluate(q_free, test_loader)
    print("\nQuantised conv weights after free QAT:")
    for name in ["conv1", "conv2"]:
        report_quantised_conv(q_free, name)

    # 4b) INT8 QAT with the cluster structure enforced after every optimizer step
    set_seed()
    q_keep = QuantLeNet().to(device)
    load_float_weights(q_keep, fp_cl)

    def keep_clusters(model):
        for n, (a, s) in cluster_info.items():
            project_to_clusters(getattr(model, n).weight.data, a, s, K)

    train(q_keep, train_loader, test_loader, "INT8 QAT, cluster-preserving", post_step=keep_clusters)
    results["INT8 QAT, clustered (structure preserved)"] = evaluate(q_keep, test_loader)
    print("\nQuantised conv weights after cluster-preserving QAT:")
    for name in ["conv1", "conv2"]:
        report_quantised_conv(q_keep, name)
    torch.save(q_keep.state_dict(), "lenet_qat_keep.pth")
    torch.save(q_ref.state_dict(), "lenet_qat_plain.pth")   # optional, for baseline comparison

    # ----------------------------- summary -----------------------------
    print("\n" + "=" * 70 + "\nSUMMARY (MNIST test accuracy)\n" + "=" * 70)
    for k_, v in results.items():
        print(f"{k_:<48s} {v:6.2f}%")
