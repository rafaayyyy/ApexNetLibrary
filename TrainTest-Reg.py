import numpy as np
import matplotlib.pyplot as plt

import nnfs # type: ignore
from nnfs.datasets import spiral_data # type: ignore
from nnfs.datasets import sine_data # type: ignore
nnfs.init()

from ApexNetLibrary import ApexNetModel
import ApexNetLibrary as apex

# Create dataset
X, y = sine_data()
X_test, y_test = sine_data()

# Instantiate the model
model = ApexNetModel()

# Add layers

model.add(apex.DenseLayer(1, 64))
model.add(apex.ReLUActivation())
model.add(apex.DenseLayer(64, 64))
model.add(apex.ReLUActivation())
model.add(apex.DenseLayer(64, 1))
model.add(apex.LinearActivation())

# Set loss, optimizer and accuracy objects
model.set(
    loss=apex.MeanSquaredErrorLoss(),
    optimizer=apex.AdamOptimizer(learningRate=0.005, lrDecay=1e-3),
    accuracy=apex.regressionAccuracy()
)

# Train the model
model.train(X, y, epochs=10000, printFrequency=100)

model.evaluate(X_test, y_test)