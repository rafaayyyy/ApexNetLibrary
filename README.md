# ApexNetLibrary

ApexNetLibrary is a Python library designed to provide a comprehensive yet straightforward interface for building and training neural networks entirely from scratch. This library is aimed at educational purposes, allowing users to grasp the underlying mechanics of neural networks without relying on high-level frameworks. It is a custom deep learning neural network library built for flexibility and ease of use. This library allows you to create, train, and evaluate neural network models with simple and intuitive syntax.

## Features

- **Layer Abstraction:** Easily define layers and ANN Objects (e.g., Dense, Dropout Filters) with activation functions.
- **Activation Functions:** Includes implementations of popular activation functions like Linear, ReLU, Sigmoid, and Tanh.
- **Loss Functions:** Supports various loss functions such as Mean Squared Error (MSE), Mean Absolute Error (MAE) for regression tasks and Cross-Entropy for evaluating model performance in classification tasks.
- **Optimization Algorithms:** Implements all the well know optimizers like Stochastic Gradient Descent (SGD), Adagrad, RMSProp and Adam for efficient training.
- **Model Evaluation:** Tools for model evaluation including accuracy calculation.

## Table of Contents

- [Installation](#installation)
- [Main Imports](#main-imports)
- [Usage Example](#usage-example)
  - [Instantiate the Model](#instantiate-the-model)
  - [Add Layers](#add-layers)
  - [Set Loss, Optimizer, and Accuracy Objects](#set-loss-optimizer-and-accuracy-objects)
  - [Train the Model](#train-the-model)
  - [Evaluate the Model](#evaluate-the-model)
  - [Make Predictions](#make-predictions)
- [Experiments](#experiments)

## Installation

To use ApexNetLibrary, you can install the library directly from the source code i.e. download zip folder, or you can clone the Github Repo:

```bash
git clone https://github.com/rafaayyyy/ApexNetLibrary.git
```

## Main Imports


To use the library, you need to have NumPy installed. If you haven't installed it yet, you can do so using the following command:

```bash
pip install numpy
```

After that, import the necessary components as shown below:
```python
import numpy as np
from ApexNetLibrary import ApexNetModel
import ApexNetLibrary as apex
```

## Usage Example

### Instantiate the Model

First, create an instance of the model:

```python
model = ApexNetModel()
```

### Add Layers

Add layers to the model using the `add` method:

```python
model.add(apex.DenseLayer([Number of Features in one Sample], [Hidden Layer Neurons Count]))
model.add(apex.ReLUActivation())  # Activation of your choice
model.add(apex.DenseLayer([Hidden Layer Neurons Count], [Output Layer Neurons Count]))
model.add(apex.SoftmaxActivation())  # Activation of your choice
```

### Set Loss, Optimizer, and Accuracy Objects

Set the loss function, optimizer, and accuracy metric for the model. Given below is one usage example:

```python
model.set(
    loss=apex.CategoricalCrossEntropyLoss(),
    optimizer=apex.AdamOptimizer(lrDecay=1e-3),
    accuracy=apex.categoricalAccuracy()
)
```

### Train the Model

Train the model using the `train` method:

```python
model.train(X, y, epochs=10, batch_size=128, printFrequency=100)
```

### Evaluate the Model

Evaluate the model on the test or validation set:

```python
model.evaluate(X_test, y_test)
```

### Make Predictions

Make predictions on new data:

```python
confidences = model.predict([Your input data])
```

## Experiments

In order to validate the working and performance of the library, Fashion MNIST dataset is used which can be found in `TrainTest.ipynb` for classification and `TrainTest-Reg.py` for regression.
