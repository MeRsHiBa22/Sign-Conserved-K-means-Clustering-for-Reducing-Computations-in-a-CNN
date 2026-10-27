# Sign Conserved-K-means Clustering and INT8 Quantization of CNNs

## Overview

This project investigates the combination of sign-conserved K-means (SC-K-means) clustering and INT8 quantization to explore computational and hardware-efficiency improvements in convolutional neural networks (CNNs).

The initial experiments use LeNet on MNIST. Experiments with VGG-11 on CIFAR-10 are maintained separately.

## Motivation

Convolutional layers perform a large number of multiply-accumulate operations. Reducing the computational cost of these operations is important for efficient CNN inference, particularly on hardware accelerators.

This project explores two complementary techniques:

1. **Sign conserved-K-means clustering:** K-means clustering is employed to group the weights in a kernel to a specific number of clusters and make them take on the centroid value of the cluster they belong to, while retaining their original signs. By doing so, say a kernel of 3*3 dimensions having 9 distinct values will now have 'k' distinct values ( k = number of clusters) and retain their original sign and sign of the values plays quite an importance when the CNN has to learn features pertaining to edges and sharp boundaries. Now initially in the convolution operation 9 multiplications would be needed, but with 3 distinct kernel values and appropriate algebraic arrangement we can perform only 3 multiplications to get the results.


2. **INT8 quantization:** Represent weights and activations using 8-bit integer values to investigate low-precision inference.

Clustering alone does not automatically reduce the number of multiplications in a conventional convolution. Actual reductions depend on whether the implementation exploits the resulting repeated weight values.

## Alternating-Layer SC-K-means

SC-K-means clustering is applied to alternating convolutional layers rather than every convolutional layer.

The objective is to investigate whether retaining the original weights in intervening layers helps preserve classification accuracy while still enabling potential computational savings in the clustered layers.

The clustered model is evaluated against the original floating-point baseline.

## Quantization and Brevitas

Brevitas is used to implement quantization-aware training (QAT).

The quantization experiments investigate INT8 representations of weights and activations, with the aim of making the model more suitable for hardware implementations supporting low-precision arithmetic.

Two approaches were investigated:

- **Post-training quantization (PTQ):** Quantization applied after training.
- **Quantization-aware training (QAT):** Training that incorporates simulated quantization so the model can adapt to quantization effects.

Initial PTQ experiments showed poor accuracy under the tested configurations. These results motivated further investigation using QAT.

The results of both approaches are retained to document the experimental process.

## Experiments

### MNIST + LeNet

The initial experiment covers:

- Baseline LeNet training.
- Alternating-layer SC-K-means clustering.
- Brevitas-based quantization-aware training.
- Evaluation of classification accuracy.

### CIFAR-10 + VGG-11

Experiments using VGG-11 on CIFAR-10 are maintained separately and will be documented in a subsequent update.

## Repository Structure

```text
experiments/
├── mnist_lenet/
│   ├── baseline/
│   ├── sc_kmeans/
│   └── qat/
└── cifar10_vgg11/

results/
├── ptq/
├── qat/
└── sc_kmeans/

docs/
```

## Evaluation

Classification accuracy is the primary evaluation metric.

Where available, additional metrics include:

- Number of distinct weight values per kernel.
- Estimated multiply-accumulate operations.
- Inference latency.
- Hardware resource utilization.

Estimated computational savings are distinguished from measured hardware improvements.

## Current Status

This repository documents an ongoing investigation into SC-K-means clustering and quantization for CNNs.
Results are reported according to the configurations actually tested.
