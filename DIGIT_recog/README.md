# MNIST Neural Network from Scratch 

## Overview
This repository contains a foundational, built-from-scratch implementation of a 2-layer Multi-Layer Perceptron (MLP) using only NumPy and Pandas. The file `MNIST_DIGIT_RECOGNISER.py` is designed as a raw learning exercise to understand the deep math behind neural networks without relying on frameworks like TensorFlow or PyTorch.

## Technical Details
- **Architecture:** 2 Layers (Input -> Hidden Layer -> Output)
- **Hidden Nodes:** 10 (A deliberate constraint to observe capacity limits)
- **Activations:** ReLU (Hidden), Softmax (Output)
- **Dataset:** Standard MNIST (or similar tabular pixel data) via `train.csv`
- **Loss Optimization:** Gradient Descent via custom Backpropagation

## Usage
Ensure you have `train.csv` in the same directory.
```bash
python MNIST_DIGIT_RECOGNISER.py
```

## Known Limitations
This is a v1 tutorial script. The `tomax` (Softmax) function currently calculates the sum across the entire matrix instead of the specific axis, which is a great debugging exercise for learners. 

## Future Extensions & Subtopics
Here are some great subtopics to explore to extend this project:
* **Softmax Axis Correction:** Debugging the probability distribution generation.
* **Dynamic Hidden Layer Sizing:** Parameterizing the hidden nodes to easily scale from 10 to 128 or 256.
* **Loss Tracking:** Implementing Cross-Entropy Loss calculation to plot a loss curve over the 500 iterations.
* **Learning Rate Decay:** Gradually lowering the alpha value during training for better convergence.

## REFERENCE
*Samson Zhang's video: Buidling a NEURAL NETWORK FROM SCRATCH
