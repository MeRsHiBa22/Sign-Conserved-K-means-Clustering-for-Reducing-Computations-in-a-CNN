import os
import torch
import torch.nn as nn
import numpy as np

from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 128

# For a 3x3 kernel
N_CLUSTERS = 3

MODEL_PATH = "results/lenet_mnist.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs("results", exist_ok=True)


# ============================================================
# LeNet
# ============================================================

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

        self.fc1 = nn.Linear(
            16 * 7 * 7,
            120
        )

        self.fc2 = nn.Linear(
            120,
            84
        )

        self.fc3 = nn.Linear(
            84,
            10
        )

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x = self.fc3(x)

        return x


# ============================================================
# MNIST
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Accuracy function
# ============================================================

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predicted = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    return 100.0 * correct / total


# ============================================================
# SIGN-CONSERVED K-MEANS
# ============================================================

def sign_conserved_kmeans(kernel, n_clusters=3):

    """
    Apply K-means to the MAGNITUDES of a kernel while
    preserving the original sign of every weight.

    Example:

        Original:
            [-0.8, 0.2, -0.5]

        Magnitudes:
            [ 0.8, 0.2,  0.5]

        K-means:
            [0.65, 0.2, 0.65]

        Restore signs:
            [-0.65, 0.2, -0.65]

    Parameters
    ----------
    kernel : torch.Tensor
        A single 2D convolution kernel.

    n_clusters : int
        Number of magnitude clusters.

    Returns
    -------
    clustered_kernel : torch.Tensor
        Sign-conserved clustered kernel.

    centroids : numpy array
        Magnitude cluster centroids.
    """

    # --------------------------------------------------------
    # Original values
    # --------------------------------------------------------

    original = kernel.detach().cpu().numpy()

    # --------------------------------------------------------
    # Separate sign and magnitude
    # --------------------------------------------------------

    signs = np.sign(original)

    magnitudes = np.abs(original)

    # --------------------------------------------------------
    # Flatten magnitudes
    # --------------------------------------------------------

    magnitude_values = magnitudes.reshape(-1, 1)

    # --------------------------------------------------------
    # K-means on MAGNITUDES ONLY
    # --------------------------------------------------------

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        magnitude_values
    )

    centroids = kmeans.cluster_centers_.flatten()

    # --------------------------------------------------------
    # Replace each magnitude with its centroid
    # --------------------------------------------------------

    clustered_magnitudes = centroids[labels]

    clustered_magnitudes = clustered_magnitudes.reshape(
        magnitudes.shape
    )

    # --------------------------------------------------------
    # Restore original signs
    # --------------------------------------------------------

    clustered_kernel = (
        signs * clustered_magnitudes
    )

    # --------------------------------------------------------
    # Convert back to PyTorch tensor
    # --------------------------------------------------------

    clustered_kernel = torch.tensor(
        clustered_kernel,
        dtype=kernel.dtype
    )

    return clustered_kernel, centroids


# ============================================================
# LOAD ORIGINAL TRAINED MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING TRAINED MODEL")
print("=" * 70)

model = LeNet5().to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state_dict)

model.eval()


# ============================================================
# BASELINE ACCURACY
# ============================================================

baseline_accuracy = evaluate(
    model,
    test_loader
)

print(
    f"\nBaseline accuracy: "
    f"{baseline_accuracy:.2f}%"
)


# ============================================================
# SAVE ORIGINAL CONVOLUTION WEIGHTS
# ============================================================

original_conv1 = (
    model.conv1.weight.data.clone()
)

original_conv2 = (
    model.conv2.weight.data.clone()
)


# ============================================================
# SIGN-CONSERVED K-MEANS FOR CONV1
# ============================================================

print()
print("=" * 70)
print("SIGN-CONSERVED K-MEANS: CONV1")
print("=" * 70)

clustered_conv1 = torch.zeros_like(
    original_conv1
)

for out_channel in range(
    original_conv1.shape[0]
):

    for in_channel in range(
        original_conv1.shape[1]
    ):

        kernel = original_conv1[
            out_channel,
            in_channel
        ]

        clustered_kernel, centroids = (
            sign_conserved_kmeans(
                kernel,
                N_CLUSTERS
            )
        )

        clustered_conv1[
            out_channel,
            in_channel
        ] = clustered_kernel

        print(
            f"Kernel [{out_channel},{in_channel}] "
            f"centroids = "
            f"{np.round(centroids, 5)}"
        )


# ============================================================
# SIGN-CONSERVED K-MEANS FOR CONV2
# ============================================================

print()
print("=" * 70)
print("SIGN-CONSERVED K-MEANS: CONV2")
print("=" * 70)

clustered_conv2 = torch.zeros_like(
    original_conv2
)

for out_channel in range(
    original_conv2.shape[0]
):

    for in_channel in range(
        original_conv2.shape[1]
    ):

        kernel = original_conv2[
            out_channel,
            in_channel
        ]

        clustered_kernel, centroids = (
            sign_conserved_kmeans(
                kernel,
                N_CLUSTERS
            )
        )

        clustered_conv2[
            out_channel,
            in_channel
        ] = clustered_kernel

        print(
            f"Kernel [{out_channel},{in_channel}] "
            f"centroids = "
            f"{np.round(centroids, 5)}"
        )


# ============================================================
# SAVE CLUSTERED KERNELS
# ============================================================

torch.save(
    {
        "conv1": clustered_conv1,
        "conv2": clustered_conv2
    },
    "results/sign_conserved_clustered_kernels.pth"
)

print()
print(
    "Saved clustered kernels to:"
)

print(
    "results/sign_conserved_clustered_kernels.pth"
)


# ============================================================
# CREATE EXPERIMENTAL MODEL
# ============================================================

experiment_model = LeNet5().to(DEVICE)

# Start with EXACTLY the original trained model
experiment_model.load_state_dict(
    state_dict
)


# ============================================================
# LAYER-WISE ALTERNATION
#
# Conv1 -> ORIGINAL
# Conv2 -> SIGN-CONSERVED K-MEANS
# ============================================================

# Conv1 remains original
experiment_model.conv1.weight.data.copy_(
    original_conv1
)

# Conv2 becomes clustered
experiment_model.conv2.weight.data.copy_(
    clustered_conv2
)


# Biases remain ORIGINAL

experiment_model.conv1.bias.data.copy_(
    model.conv1.bias.data
)

experiment_model.conv2.bias.data.copy_(
    model.conv2.bias.data
)


# ============================================================
# EVALUATE EXPERIMENTAL MODEL
# ============================================================

experiment_accuracy = evaluate(
    experiment_model,
    test_loader
)


# ============================================================
# CALCULATE ACCURACY DIFFERENCE
# ============================================================

accuracy_difference = (
    baseline_accuracy
    - experiment_accuracy
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"Baseline accuracy       : "
    f"{baseline_accuracy:.2f}%"
)

print(
    f"Alternating accuracy    : "
    f"{experiment_accuracy:.2f}%"
)

print(
    f"Accuracy difference     : "
    f"{accuracy_difference:.2f} percentage points"
)

print("=" * 70)


# ============================================================
# SAVE EXPERIMENTAL MODEL
# ============================================================

torch.save(
    experiment_model.state_dict(),
    "results/lenet_sign_conserved_alternating.pth"
)

print()
print(
    "Saved experimental model to:"
)

print(
    "results/lenet_sign_conserved_alternating.pth"
)
