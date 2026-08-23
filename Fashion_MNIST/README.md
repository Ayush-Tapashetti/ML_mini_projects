# Fashion-MNIST Classifier (NumPy Implementation)

## Overview
A robust, optimized 2-layer neural network built entirely from scratch to classify the Fashion-MNIST dataset. `Fashion_MNIST.py` demonstrates how to handle vectorization, dynamic bias gradients, and image visualization using Matplotlib without relying on heavy deep learning libraries.

## Technical Details
- **Architecture:** 2 Layers (784 Input -> 128 Hidden -> 10 Output)
- **Activations:** ReLU (Hidden), Corrected Softmax (Output)
- **Gradients:** Properly managed dimensional axes for bias updates.
- **Dataset:** `fashion-mnist_train.csv` (28x28 pixel grayscale clothing images)
- **Evaluation:** Includes Dev-set validation and Matplotlib visual inspection functions.

## Usage
Ensure you have `fashion-mnist_train.csv` in the same directory.
```bash
python Fashion_MNIST.py
```
*Note: Ensure the return statement in `gradient_Descent` is correctly unindented outside the loop to complete the 1000 epochs! Cause,I have fumbled plenty of times right at this step.*

## Visualization
The script includes a `test_prediction` function that plots the predicted vs actual clothing label alongside the actual 28x28 image using `matplotlib.pyplot`.

## Future Extensions & Subtopics
Want to take this project to the next level? Try implementing these subtopics:
* **Mini-Batch Gradient Descent:** Transitioning from full-batch to mini-batches (e.g., 32 or 64 samples at a time) for faster, more stable training.
* **Model Persistence (Save/Load Weights):** Writing the trained `W1, b1, W2, b2` matrices to a `.npy` or `.npz` file so you don't have to retrain from scratch every time you run the script.
* **Advanced Optimizers:** Upgrading from standard Gradient Descent to Adam or RMSprop.
* **Regularization (Dropout & L2):** Implementing dropout masks during forward propagation to prevent the 128-node hidden layer from overfitting on the training set.
