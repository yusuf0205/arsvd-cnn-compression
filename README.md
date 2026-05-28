# Adaptive-Rank SVD (ARSVD) Neural Network Compression 🧠⚡

This repository contains the code, research, and presentation materials for my **1-Month Research Internship** at the **National Institute of Technology Calicut (NITC)**, conducted under the guidance of **Dr. Ameer PM** Professor and Research scholar **vijeesh**.

### 📌 Project Overview
As Deep Learning models grow exponentially in size, deploying them on edge devices with limited memory and compute power has become a major engineering bottleneck. This research focuses on mathematically compressing Neural Networks without sacrificing their predictive accuracy. 

By implementing **Adaptive-Rank Singular Value Decomposition (ARSVD)**, we can identify and mathematically prune redundant weights inside fully connected (Linear) and Convolutional layers, replacing massive parameter blocks with highly efficient "Hourglass Bottleneck" sub-layers.

### 🔬 Key Features & Research Highlights
* **Spectral Entropy Thresholding:** Instead of blindly deleting weights, the algorithm calculates the spectral entropy of each layer to dynamically select the optimal Rank (*k*) to preserve a specific percentage of data variance (e.g., *tau* = 0.80).
* **FC1 & Conv2d Targeted Compression:** Custom PyTorch surgery scripts that safely unfold 4D Convolutional tensors into 2D matrices, perform SVD, and refold them into spatially compressed sub-layers (Standard Conv + 1x1 Pointwise Conv).
* **Hardware Safety Net:** An algorithmic failsafe that calculates projected FLOPs and parameter counts *before* splitting a layer, ensuring the compression physically accelerates the hardware.
* **Ablation Studies:** Extensive hyperparameter sweeps evaluating the trade-offs between parameter reduction, inference latency, and accuracy.

---

### 📊 Performance Results (Custom MNIST CNN)
By strictly targeting the highest-density layers (Conv2d Layer 3 and Fully Connected Layer 1), we successfully accelerated the forward pass and reduced the memory footprint with negligible impact on the model's accuracy.

| Evaluation Metric | Original CNN | ARSVD Compressed CNN | Overall Impact |
| :--- | :--- | :--- | :--- |
| **Model Accuracy** | 98.86% | 98.65% | **- 0.21%** (Negligible Drop) |
| **Total Parameters** | 421,642 | 349,161 | **17.19%** Memory Saved |
| **Compute FLOPs (MACs)** | 4,278,922 | 3,425,658 | **19.94%** Compute Saved |

---

### 📂 Repository Structure
```text
arsvd-cnn-compression/
├── docs/                   # Research papers and literature review
├── presentation/           # Final slide decks and visual assets
├── src/                    # Source code, Jupyter notebooks, and PyTorch scripts
├── models/                 # Saved state_dicts for baseline and compressed models
├── .gitignore              # Ignores __pycache__, large datasets, and heavy .pth files
└── README.md               # Project documentation
