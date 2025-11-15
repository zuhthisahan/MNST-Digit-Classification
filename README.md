# MNIST Digit Classification with PyTorch

This project demonstrates how to build, train, and evaluate two different neural network architectures for handwritten digit classification using the MNIST dataset. It includes both a simple Fully Connected Network (FCN) and a Convolutional Neural Network (CNN), implemented in PyTorch.

All steps are shown in the `MNIST_CNN.ipynb` notebook, covering an end-to-end deep learning workflow.

---

## Project Overview

### 1. Data Loading & Preparation
- The MNIST dataset is loaded using `torchvision.datasets`.
- The 60,000 training images are split into:
  - **50,000-image training set**
  - **10,000-image validation set**
- The **10,000-image test set** is loaded separately.
- `DataLoader` is used to create batch iterators (batch size = 64).
- A grid of 36 sample images is visualized to inspect the data.

---

## Model 1: Fully Connected Network (FCN)

A simple fully connected neural network (`MINST`) is defined:
- The 28×28 pixel input is flattened.
- Multiple linear layers map the input to **10 output logits**.
- After **10 training epochs**, the model achieves approximately **97.0% test accuracy**.

---

## Model 2: Convolutional Neural Network (CNN)

A more advanced convolutional model (`MNST_CNN`) is implemented.

### Architecture:
 - Conv2d (1 → 24) → ReLU → MaxPool2d
 - Conv2d (24 → 36) → ReLU → MaxPool2d
 - Flatten
 - Linear (900 → 128) → ReLU
 -  Linear (128 → 10)
   
After **10 epochs**, this model achieves a significantly higher accuracy of approximately **99.0%** on the test set.


## Key Technologies

- **PyTorch** (nn, optim, DataLoader)  
- **Torchvision** (MNIST dataset, transforms)  
- **NumPy & Matplotlib** for visualization  
- **tqdm** for progress bars  
