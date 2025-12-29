#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout,Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the MNIST dataset
(x_train, y_train_orig), (x_test, y_test_orig) = mnist.load_data()
x_train.shape, y_train_orig.shape, x_test.shape, y_test_orig.shape


# In[3]:


# Normalize the pixel values (x values) to [0, 1] 
# Pixel intensity values in the MNIST dataset range from 0 to 255 and dividing by 255 scales all feature values to the [0, 1] range.
x_train = x_train / 255.0
x_test = x_test / 255.0


# In[4]:


y_test_orig


# In[5]:


# Convert labels (y) to one-hot encoding

y_train = to_categorical(y_train_orig, 10)
y_test = to_categorical(y_test_orig, 10)


# In[6]:


y_test


# # Build the model

# In[7]:


# Build the ANN model
ann_model = Sequential([
    Flatten(input_shape=(28, 28)),      # Flatten 28x28 images to 784 vector
    Dense(128, activation='relu'),      # Hidden layer with 128 neurons
    Dropout(0.3),                       # To randomly deactivate (30%) neurons during training, reducing overfitting and improving generalization.
    Dense(64, activation='relu'),       # Another hidden layer
    Dropout(0.3),
    Dense(10, activation='softmax')     # Output layer for 10 classes
])


# In[8]:


# Compile the model

ann_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# # Model training

# In[9]:


from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)


# In[10]:


history = ann_model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)


# # Inference : 
# The model demonstrates rapid convergence with stable training and validation performance, reaching approximately 97.6% validation accuracy by epoch 9. The close alignment between training and validation metrics indicates effective regularization and strong generalization, with diminishing returns observed beyond epoch 7.

# In[11]:


# Evaluate the model on test data

test_loss, test_accuracy = ann_model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")


# # Inference :
# The trained ANN achieved a test accuracy of 97.69% with low cross-entropy loss, demonstrating strong generalization and effective regularization. Performance is consistent with the expected upper bound for fully connected neural networks on the MNIST dataset.

# #### Visualize the network

# In[12]:


history_main = history  # fallback if history_main was not defined

epochs_ran = len(history_main.history['loss'])
epochs = range(1, epochs_ran + 1)

best_epoch = np.argmin(history_main.history['val_loss']) + 1

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, history_main.history['accuracy'], label='Train Accuracy')
plt.plot(epochs, history_main.history['val_accuracy'], label='Validation Accuracy')
plt.axvline(best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history_main.history['loss'], label='Train Loss')
plt.plot(epochs, history_main.history['val_loss'], label='Validation Loss')
plt.axvline(best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()

plt.suptitle(f"Training Curves (Early Stopping at Epoch {best_epoch})")
plt.tight_layout()

plt.savefig("Accuracy_loss_epochs.jpeg",dpi=300)
plt.show()


# # Inference : 
# 
# 1. Training accuracy increases from ~86.7% (epoch 1) to ~97.6% (epoch 11), indicating effective learning of discriminative features.
# 
# 2. Validation accuracy improves from ~95.3% to ~97.7%, closely tracking training accuracy and demonstrating strong generalization.
# 
# 3. The maximum validation accuracy (~97.7%) is achieved around epoch 11, after which performance plateaus.
# 
# 4. Training loss decreases sharply from ~0.44 to ~0.08, reflecting rapid convergence during early epochs.
# 
# 5. Validation loss reduces from ~0.16 to a minimum of ~0.09 at epoch 11, confirming optimal model selection by EarlyStopping.
# 
# 6. The minimal gap between training and validation curves across epochs indicates negligible overfitting.
# 
# 7. Early stopping at epoch 11 prevents unnecessary training and preserves the best-performing model weights.
# 
# 8. Additional epochs beyond epoch 11 do not yield significant improvement, indicating the model has reached its performance ceiling.

# # Model testing - evaluating performance on unseen data

# In[13]:


y_pred = ann_model.predict(x_test)


# In[14]:


np.round(y_pred)


# In[15]:


# Converting Probabilities to Predicted Classes

y_pred_classes = np.argmax(y_pred, axis=1)
y_pred_classes


# In[16]:


# The confusion matrix compares the true class labels (y_test_orig) with the model’s predicted labels (y_pred_classes)

cm = confusion_matrix(y_test_orig, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='mako',
    xticklabels=range(10),
    yticklabels=range(10)
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.jpeg",dpi=300)
plt.show()


# # Inference
# 
# 1. The confusion matrix exhibits strong diagonal dominance, indicating a high proportion of correct classifications across all digit classes.
# 
# 2. The model achieves balanced class-wise performance, with no significant bias toward any particular digit.
# 
# 3. Digits such as 0, 1, and 2 are classified with near-perfect accuracy, reflecting their distinct visual patterns.
# 
# 4. Most misclassifications occur between visually similar digits (e.g., 4 vs 9, 3 vs 5, 7 vs 9), which is expected for a fully connected ANN that does not preserve spatial information.
# 
# 5. Off-diagonal values are relatively small, confirming that errors are limited and systematic rather than random.
# 
# 6. The confusion matrix aligns with the overall ~97.5% test accuracy, validating the reliability and generalization capability of the model.
# 
# Residual classification errors highlight the architectural limitation of ANNs for image data, suggesting that convolutional neural networks would further reduce these confusions.

# In[17]:


from sklearn.metrics import classification_report

print(
    classification_report(
        y_test_orig,
        y_pred_classes,
        digits=4
    )
)


# # Inference : 
# 
# 1. Overall Performance:
# The model achieved a test accuracy of 97.4% with a macro F1-score of 0.9738, indicating strong and balanced classification across all digit classes.
# 
# 2. Class Balance:
# The macro-averaged (0.9738) and weighted F1-scores (0.9740) are nearly identical, confirming that performance is not biased by class frequency.
# 
# 3. High-Confidence Digits:
# Digits 0, 1, and 6 show the strongest performance with F1-scores ≥ 0.978, reflecting clear separability and stable recognition.
# 
# 4. Moderate Variability Digits:
# Digits 2, 3, 4, 5, 7, and 8 maintain F1-scores around 0.97, indicating reliable recognition with minor sensitivity to handwriting style variations.
# 
# 5. Weakest Class (Digit 9):
# Digit 9 records the lowest recall (94.75%) despite high precision (98.15%), suggesting missed detections likely due to confusion with visually similar digits such as 4 or 7.
# 
# 6. Error Characteristics:
# Misclassifications are localized and class-specific rather than systemic, supporting the conclusion that errors arise from spatial pattern ambiguity rather than poor generalization.
# 
# 7. Model Limitation Insight:
# The observed error patterns align with known limitations of fully connected ANN architectures, which lack explicit spatial feature extraction.
# 
# 8. Deployment Readiness:
# With >97% accuracy on a clean benchmark dataset, the model is suitable for baseline digit recognition tasks, but performance on real-world handwritten inputs may degrade without spatial modeling.

# In[18]:


# ANN-Specific Error Analysis

import numpy as np

# Copy confusion matrix and zero diagonal
cm_offdiag = cm.copy()
np.fill_diagonal(cm_offdiag, 0)

# Get top confusion pairs
top_confusions = np.dstack(np.unravel_index(
    np.argsort(cm_offdiag.ravel())[::-1][:5],
    cm_offdiag.shape
))[0]

for true, pred in top_confusions:
    print(f"True {true} → Predicted {pred} : {cm[true, pred]} samples")


# # Inference :
# 
# 1. Digit 9 → 4 (17 samples)
# This is the most frequent error, confirming that the ANN struggles to distinguish loop–stem combinations when spatial context is flattened. This directly explains the lower recall for digit 9 (94.75%) observed in class-wise metrics.
# 
# 2. Digit 7 → 2 (11 samples)
# Misclassification likely arises from similar diagonal stroke patterns, particularly in handwritten styles where the crossbar of 7 is absent.
# 
# 3. Digit 3 → 2 (11 samples)
# Indicates sensitivity to curvature and stroke closure, where incomplete or open loops cause ambiguity in flattened representations.
# 
# 4. Digit 7 → 1 (9 samples)
# Reflects confusion when digit 7 is written without a horizontal bar, making it visually similar to a straight vertical stroke.
# 
# 5. Digit 5 → 3 (9 samples)
# Highlights ANN difficulty in differentiating curved versus semi-looped structures when pixel spatial relationships are not explicitly modeled

# # Hyperparameter tuning 

# In[19]:


# Define a Model Builder Function

def build_model(learning_rate=1e-3, dropout_rate=0.3):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# In[20]:


# Define the Hyperparameter Grid and run the model

learning_rates = [1e-3, 5e-4, 1e-4]
dropout_rates = [0.2, 0.3, 0.4]
batch_sizes = [32, 64]

results = []

early_stop = EarlyStopping( monitor='val_loss', patience=3,restore_best_weights=True)

for lr in learning_rates:
    for dr in dropout_rates:
        for bs in batch_sizes:
            print(f"\nTraining with LR={lr}, Dropout={dr}, BatchSize={bs}")

            model = build_model(learning_rate=lr, dropout_rate=dr)

            history = model.fit(
                x_train,
                y_train,
                epochs=20,
                batch_size=bs,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )

            best_val_acc = max(history.history['val_accuracy'])

            results.append({
                "learning_rate": lr,
                "dropout": dr,
                "batch_size": bs,
                "best_val_accuracy": best_val_acc
            })

            print(f"Best Val Accuracy: {best_val_acc:.4f}")



# # Hyperparameter Tuning – Point-Wise Inference
# 
# 1. The model shows highest validation accuracy (≈97.6%) with a learning rate of 0.001, indicating that the default Adam learning rate is optimal for th is ANN on MNIST.
# 
# 2. Reducing the learning rate to 0.0005 or 0.0001 results in slower convergence and underfitting, leading to a consistent drop in validation accuracy.
# 
# 3. A dropout rate of 0.2 provides the best trade-off between regularization and learning capacity; higher dropout values (0.3–0.4) introduce excessive regularization and degrade performance.
# 
# 4. A batch size of 32 consistently outperforms larger batch sizes, suggesting improved gradient updates and better generalization.
# 
# 5. The overall improvement from hyperparameter tuning is marginal, confirming that the baseline ANN was already near its optimal configuration.
# 
# The tuning results indicate that model performance is architecture-limited, and further gains would require structural changes (e.g., CNN) rather than additional hyperparameter adjustments.

# # Sample digit Prediction Visualization

# In[22]:


plt.figure(figsize=(10, 8))

plot_idx = 1
for digit in range(10):
    idx = np.where(y_test_orig == digit)[0][0]  # first sample of each digit

    plt.subplot(3, 4, plot_idx)   # 3 rows, 4 columns
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"True: {digit}\nPred: {y_pred_classes[idx]}", fontsize=10)
    plt.axis('off')

    plot_idx += 1

# Hide unused subplots (11 and 12)
for i in range(plot_idx, 13):
    plt.subplot(3, 4, i)
    plt.axis('off')

plt.suptitle("Predictions for Digits 0–9 (3×4 Grid)", fontsize=14)
plt.savefig("sample_digits.jpeg",dpi=300)
plt.tight_layout()
plt.show()


# # Inference :
# 
# 1. The grid displays one representative test sample for each digit class (0–9), arranged in a compact 3×4 layout for clear visual comparison.
# 
# 2. For all displayed samples, the predicted labels match the true labels, demonstrating correct class-wise recognition across all digit categories.
# 
# 3. The model accurately captures distinct digit shapes (e.g., 1, 7) as well as more complex patterns (e.g., 8, 9), indicating effective feature learning from pixel intensities.
# 
# 4. The correct predictions across all digits confirm balanced performance, with no observable class bias.
# 
# 5. This qualitative visualization is consistent with the confusion matrix and ~97.5% test accuracy, reinforcing the model’s strong generalization on unseen data.
# 
# 6. The results further validate that remaining errors (seen elsewhere) are case-specific and due to visual ambiguity, not systematic failure.

# # Test Digits from a Local Image

# # Load & Preprocess Local Image

# In[23]:


#!pip install opencv-python


# In[24]:


import cv2

def preprocess_digit(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image not found")

    # Threshold to binary
    img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

    # Resize
    img = cv2.resize(img, (28, 28))

    # Normalize
    img = img.astype("float32") / 255.0

    # CNN shape
    img = img.reshape(1, 28, 28)

    return img


# # Run Inference

# In[25]:


image_path = "image2.jpg"   # your local image
x_test = preprocess_digit(image_path)

prediction = ann_model.predict(x_test)
predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)


# # Visual Sanity Check

# In[26]:


plt.imshow(x_test.reshape(28,28), cmap="gray")
plt.title(f"Predicted: {predicted_digit}")
plt.axis("off")
plt.show()


# # Summary :
# 
# 1. Objective
# Develop a baseline Artificial Neural Network (ANN) for handwritten digit classification on the MNIST dataset to demonstrate core deep learning concepts.
# 
# 2. Model & Training
# A simple fully connected ANN was used with the Adam optimizer (learning rate = 0.001) and categorical cross-entropy loss, prioritizing stability and interpretability over architectural complexity.
# 
# 3. Performance Metrics
# • Training Accuracy: ~97–98%
# • Validation Accuracy: ~96–97%
# • Loss curves showed smooth convergence with no significant overfitting.
# 
# 4. Outcome
# The model effectively learned non-linear pixel relationships and delivered strong baseline performance for an ANN, meeting the objectives of the project.
