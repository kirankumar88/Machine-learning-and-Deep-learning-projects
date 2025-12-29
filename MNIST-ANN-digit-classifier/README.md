# MNIST ANN Digit Classifier

## Overview
This project implements a **baseline Artificial Neural Network (ANN)** to classify handwritten digits using the MNIST dataset. The objective is to demonstrate core deep learning concepts, model evaluation, error analysis, and practical inference using a fully connected neural network.
This project prioritizes **clarity, interpretability, and rigorous evaluation** over architectural complexity.

---

## Dataset
- **MNIST handwritten digits**
- 60,000 training images, 10,000 test images
- Image size: 28 × 28 (grayscale)
- Classes: Digits 0–9

---

## Model Architecture
- Flatten layer (28×28 → 784)
- Dense layer (128 neurons, ReLU)
- Dropout (0.3)
- Dense layer (64 neurons, ReLU)
- Dropout (0.3)
- Output layer (10 neurons, Softmax)

**Optimizer:** Adam (learning rate = 0.001)  
**Loss function:** Categorical Cross-Entropy  

---

## Training Strategy
- Validation split: 20%
- Batch size: 32
- Epochs: Up to 20
- **EarlyStopping** applied to prevent overfitting

---

## Performance
- **Test Accuracy:** ~97.5–97.7%
- **Macro F1-score:** ~0.974
- Training and validation curves show stable convergence with minimal overfitting

---

## Evaluation & Error Analysis
- Confusion matrix for class-wise performance
- Precision, recall, and F1-score for each digit
- Most errors occur between visually similar digits (e.g., 4 vs 9, 3 vs 5)
- Error patterns highlight the **limitations of ANN architectures for image data**

---

## Hyperparameter Tuning
Explored:
- Learning rate
- Dropout rate
- Batch size

Result:
- Marginal improvements observed
- Confirms that performance is **architecture-limited**, not hyperparameter-limited

---

## Local Image Inference
Supports inference on custom handwritten digit images using OpenCV-based preprocessing.

