# Adaptive-Rank SVD (ARSVD) Neural Network Compression

Code, experimental notebooks, and research documentation from a 1-month Edge AI research internship at the National Institute of Technology Calicut (NITC), conducted under the guidance of Prof. Ameer P.M. and research scholar Vijeesh.

## Overview

Deploying deep neural networks on resource-constrained edge devices requires reducing memory footprint and multiply-accumulate (MAC) operations while preserving predictive accuracy. This project implements **Adaptive-Rank Singular Value Decomposition (ARSVD)** in PyTorch to factorize dense weight matrices and 4D convolutional kernels into low-rank bottleneck sub-layers.

## Key Methods

- **Spectral Entropy Rank Selection**: Computes the normalized singular-value energy distribution (spectral entropy) of each target layer to select the truncation rank `k` that retains a target variance threshold (for example, `tau = 0.80`).
- **Linear and Conv2d Layer Decomposition**: Unfolds 4D convolutional weight tensors `(C_out, C_in, K_h, K_w)` into 2D matrices, applies truncated SVD, and reconstructs them as a two-stage sequential bottleneck (reduced-rank spatial convolution followed by a `1x1` pointwise projection).
- **Hardware FLOPs Guardrail**: Evaluates the analytical parameter count and MAC cost before replacing a layer, skipping decomposition if the chosen rank `k` exceeds the break-even threshold where factorization would increase parameters.
- **Ablation Sweeps**: Evaluates accuracy, parameter count, and FLOPs across standard SVD, Energy-threshold SVD (ESVD), and Adaptive-Rank SVD (ARSVD) on FC1 alone and combined Conv2d + FC1 configurations.

## Experimental Results (MNIST CNN)

Targeting the highest-density layers (Convolutional Layer 3 and Fully Connected Layer 1) yielded the following compression trade-off on the MNIST test set:

| Metric | Baseline CNN | ARSVD Compressed CNN | Change |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | 98.86% | 98.65% | -0.21% |
| **Total Parameters** | 421,642 | 349,161 | -17.19% |
| **Compute (MACs)** | 4,278,922 | 3,425,658 | -19.94% |

Peak baseline training accuracy reached 99.10%, beyond which additional epochs or higher learning rates produced diminishing returns (`docs/OBSERVATIONS.txt`).

## Repository Structure

```text
arsvd-cnn-compression/
├── src/
│   ├── CNN/CNN.ipynb                    # Baseline MNIST CNN training and evaluation
│   ├── SVD/SVD.ipynb                    # Fixed-rank SVD matrix factorization baseline
│   ├── ESVD/ESVD_FC1_CNN.ipynb          # Energy-threshold SVD on Fully Connected Layer 1
│   ├── ESVD/ESVD_CNN.ipynb              # Energy-threshold SVD across Conv2d and Linear layers
│   └── ArSVD/ArSVD.ipynb                # Adaptive-Rank SVD with spectral entropy thresholding
├── Refernece/
│   └── CNN_MNIST_ARSVD_modified (1).ipynb
├── models/
│   └── cnn_model.pth                    # Pretrained baseline CNN weights
├── docs/
│   ├── OBSERVATIONS.txt                 # Training convergence notes
│   ├── Research Paper/                  # Reference papers on pruning and low-rank approximation
│   ├── Reviews/                         # Manual rank and FLOPs calculation derivations (PDF)
│   └── Linkedin/                        # Result summary plots and visual snapshots
├── presentation/
│   ├── Comprehensive Guide.pdf          # Technical guide to SVD, ESVD, and ARSVD
│   ├── DPRP.pptx                        # Dynamic Parameter Rank Pruning presentation
│   ├── Deep Learning.pptx               # Introductory neural network compression deck
│   └── ESVD-Compression-of-Fully-Connected-Layers.pptx
├── .gitignore
├── LICENSE
└── README.md
```

## Getting Started

1. Install Python 3.9+ and the required dependencies:

```bash
pip install torch torchvision numpy matplotlib jupyter
```

2. Launch Jupyter Notebook from the repository root:

```bash
jupyter notebook
```

3. Open `src/CNN/CNN.ipynb` to train or inspect the baseline model, then run `src/ArSVD/ArSVD.ipynb` to execute the spectral-entropy compression pipeline and benchmark table generation.

## License

MIT License. See [LICENSE](LICENSE) for details.
