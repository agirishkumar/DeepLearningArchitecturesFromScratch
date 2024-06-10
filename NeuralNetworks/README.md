# Neural Networks from Scratch

This folder contains an implementation of neural networks from scratch using NumPy. The implementation includes features like dropout, L2 regularization, hyperparameter tuning, and visualization of training/validation loss and accuracy.

## Introduction

This project demonstrates how to build and train neural networks from scratch using NumPy. It includes various features to enhance the training process and improve the performance of the neural networks.

## Installation

To run the code, you need to have Python installed along with the following packages:
- NumPy
- scikit-learn
- matplotlib

You can install the required packages using pip/conda.

## Usage

1. **Training a Neural Network**: Modify the parameters in the script and run it to train a neural network on your dataset.
2. **Hyperparameter Tuning**: Use the hyperparameter tuning function to find the best set of parameters for your neural network.
3. **Plotting Results**: The script generates plots for training/validation loss and accuracy.

## Features

- **Multiple Hidden Layers**: Support for neural networks with multiple hidden layers.
- **Dropout Regularization**: Prevents overfitting by randomly setting a fraction of input units to 0 at each update during training.
- **L2 Regularization**: Adds a penalty on the layer weights to encourage smaller weights.
- **Hyperparameter Tuning**: Automated tuning of various hyperparameters using grid search.
- **Cross-Validation**: Evaluate the performance of the model using k-fold cross-validation.
- **Data Normalization**: Normalize input data to improve training stability.
- **Visualization**: Plot training and validation loss and accuracy over epochs.

## Examples

### XOR Problem

The script includes an example of training a neural network to solve the XOR problem. It demonstrates how to define the network architecture, train the model, and generate predictions.

### Hyperparameter Tuning

The script provides a function for hyperparameter tuning, allowing you to find the optimal set of hyperparameters for your neural network. This includes tuning the learning rate, activation functions, dropout rate, L2 regularization, and batch size.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss improvements.

