# Image Classification using CNN (PyTorch)

This project implements a Convolutional Neural Network (CNN) using PyTorch to perform image classification on the CIFAR-10 dataset.

The goal of this project is to demonstrate a complete deep learning workflow including data preprocessing, model training, evaluation, and saving trained model weights.

---

## Dataset
- **CIFAR-10**
- Developed by the University of Toronto
- 60,000 RGB images (32×32)
- 10 classes

The dataset is loaded using `torchvision.datasets.CIFAR10` and downloaded automatically.

---

## Model Architecture
- Convolutional Neural Network (CNN)
- Two convolutional layers with ReLU activation
- Max Pooling layers for spatial reduction
- Fully connected layers for classification
- Final output layer with 10 classes

---

## Training Details
- Framework: PyTorch
- Loss Function: Cross-Entropy Loss
- Optimizer: Adam
- Batch Size: 64
- Epochs: 10
- Device: CPU / GPU (automatically selected)

---

## Results
- **Test Accuracy:** 71.24%
- The model was evaluated on unseen test data to measure generalization performance.

---


