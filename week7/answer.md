# Hand-In Lab 07 - Answers

---

## Task 2a: Dropout (2 Points)

**Dropout rate used:** 0.5

**Learning curve comparison (with vs. without dropout):**

- **Without dropout:** The training loss decreases rapidly and reaches very low values (~0.15 by epoch 20). The validation loss also decreases but a noticeable gap develops between training and validation loss, indicating the model starts to overfit.
- **With dropout (rate=0.5):** The training loss is higher than without dropout (~0.15 at epoch 20 vs. lower without), because during training 50% of the hidden neurons are randomly deactivated, making the training task harder. However, the gap between training loss and validation loss is smaller, meaning the model generalizes better and overfits less.

**Why?** Dropout acts as a regularizer. By randomly dropping neurons during training, it prevents the network from relying too heavily on any single neuron (co-adaptation). This forces the network to learn more robust features, reducing overfitting and improving generalization to unseen data.

**Test accuracy:** ~97.24% (without dropout) vs. ~96.90% (with dropout). The slightly lower test accuracy with dropout is expected since the model is more regularized and the base model was not severely overfitting on this dataset.

---

## Task 2b: Early Stopping "by hand" (2 Points)

**Validation loss per epoch (12 epochs, no dropout):**

| Epoch | val_loss | val_accuracy |
|-------|----------|-------------|
| 1     | 0.2645   | 92.77%      |
| 2     | 0.2085   | 94.43%      |
| 3     | 0.1779   | 95.20%      |
| 4     | 0.1578   | 95.98%      |
| 5     | 0.1397   | 96.42%      |
| 6     | 0.1330   | 96.42%      |
| 7     | 0.1211   | 96.83%      |
| 8     | 0.1142   | 96.98%      |
| 9     | 0.1090   | 96.98%      |
| 10    | 0.1032   | 97.25%      |
| 11    | 0.0996   | 97.37%      |
| 12    | 0.0950   | 97.43%      |

**Should we use early stopping?**
No, early stopping is **not necessary** in this case. Both the training loss and validation loss are consistently decreasing throughout all 12 epochs and converging toward each other. There is no sign of overfitting (the validation loss never starts increasing). The model is still improving at epoch 12.

**If yes, at what epoch would you stop?**
Since both losses are still decreasing, there is no clear epoch at which to stop early. If forced to choose, one could stop around **epoch 10-11** where the rate of improvement becomes marginal, but this is not a strong case for early stopping.

---

## Task 3: Data Augmentation (2 Points)

**Which transformation does NOT make sense for digit classification?**

**RandomFlip** (horizontal and vertical flipping) does NOT make sense for digit classification.

**Reason:** Flipping handwritten digits changes their identity. For example:
- A "6" flipped vertically becomes a "9" (and vice versa)
- A "2" flipped horizontally becomes unrecognizable
- A "7" flipped vertically or horizontally no longer looks like a "7"

Digits have a fixed orientation, so flipping them would create misleading training examples that could confuse the classifier. In contrast, small rotations and zooming are reasonable augmentations since handwritten digits can naturally appear slightly rotated or at different scales.

---

## Task 4: Boston Housing - The Deep End (4 Points)

**Network architecture:**
- Input layer: 13 features (the 13 Boston Housing features)
- Hidden layer 1: 64 neurons, ReLU activation
- Hidden layer 2: 64 neurons, ReLU activation
- Output layer: 1 neuron, linear activation (no activation function)

**Data preprocessing:** Features were standardized (zero mean, unit variance) using training set statistics to ensure stable training.

**Loss function:** Mean Squared Error (MSE)
- **Why:** This is a **regression** task (predicting continuous house prices), not classification. MSE is the standard loss function for regression that penalizes larger errors more heavily.

**Metric:** Mean Absolute Error (MAE)
- **Why:** MAE gives an interpretable measure of average prediction error in the same units as the target variable (thousands of dollars). It is less sensitive to outliers than MSE and provides a clear understanding of model performance.

**Optimizer:** Adam (adaptive learning rate optimizer, more robust than plain SGD)

**Training setup:** batch_size=32, epochs=100, validation_split=0.2

**Final performance on the test set:**
- **MSE (loss): ~14.99**
- **MAE: ~2.46** (meaning on average, predictions are off by about $2,460)

The learning curve shows that both training and validation loss decrease over epochs, confirming that the model learned meaningful patterns from the data.
