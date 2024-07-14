import numpy as np
import matplotlib.pyplot as plt

class DenseLayer():
    def __init__(self, numOfInputNeurons, numOfOutputNeurons, weight_L1Regularization=0, weight_L2Regularization=0, bias_L1Regularization=0, bias_L2Regularization=0):
        """
        Initializes the Dense Layer with given parameters, setting up weights, biases, and regularization factors.
        Weights are initialized with a normal distribution scaled by 0.1, and biases are initialized to zero.
        """
        self.weights = 0.1 * np.random.randn(numOfInputNeurons, numOfOutputNeurons)
        self.biases = np.zeros((1, numOfOutputNeurons))
        
        # Regularization factors for weights and biases
        self.weight_L1Regularization = weight_L1Regularization
        self.weight_L2Regularization = weight_L2Regularization
        self.bias_L1Regularization = bias_L1Regularization
        self.bias_L2Regularization = bias_L2Regularization

    def forward(self, inputs, training):
        """
        Computes the forward pass of the dense layer by multiplying inputs with weights and adding biases.
        Stores inputs for use in the backward pass.
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        """
        Computes the backward pass of the dense layer.
        Calculates gradients for weights, biases, and inputs.
        Applies L1 and L2 regularization adjustments to the gradients of weights and biases if their respective regularization factors are greater than 0.
        """
        # Gradient on weights and biases
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Apply L1 regularization if factor > 0
        if self.weight_L1Regularization > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_L1Regularization * dL1
        # Apply L2 regularization if factor > 0
        if self.weight_L2Regularization > 0:
            self.dweights += 2 * self.weight_L2Regularization * self.weights
            
        # Apply L1 regularization to biases if factor > 0
        if self.bias_L1Regularization > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_L1Regularization * dL1
        # Apply L2 regularization to biases if factor > 0
        if self.bias_L2Regularization > 0:
            self.dbiases += 2 * self.bias_L2Regularization * self.biases

        # Gradient on inputs for further propagation
        self.dinputs = np.dot(dvalues, self.weights.T)


class FirstLayerInput:
    def forward(self, inputs, training):
        """
        Forwards the inputs directly to the output. This layer acts as a placeholder for the first layer in a network, where no modification of inputs is necessary.
        """
        self.output = inputs

class DropoutLayerFilter():
    def __init__(self, rate):
        """
        Initializes the Dropout layer with a specified dropout rate, determining the fraction of inputs to randomly set to zero during training.
        """
        self.rate = rate

    def forward(self, inputs, training):
        """
        Applies dropout to the inputs if in training mode, by creating a mask that randomly sets a portion of inputs to zero.
        If not in training mode meaning we are calling it on testing dataset, outputs are the same as inputs.
        """
        self.inputs = inputs
        
        if not training:
            self.output = inputs.copy()
            return
        
        # Create dropout mask
        self.dropOutMask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)
        self.output = inputs * self.dropOutMask

    def backward(self, dvalues):
        """
        Backward pass for the dropout layer. Adjusts the gradients passed to the previous layer by the same dropout mask applied during the forward pass.
        """
        self.dinputs = dvalues * self.dropOutMask

# Activation functions

# Linear Activation Function
# Formula: f(x) = x
# Output range: (-∞, ∞)
# This activation function simply returns the input as output. It's useful for problems where we want the output to not be bounded.
class LinearActivation:
    def forward(self, inputs, training):
        # Store inputs and directly pass them as output
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        # Gradient of linear function is 1, so it just passes the gradient unchanged
        self.dinputs = dvalues.copy()
        
    # Returns the outputs directly without any modification
    def predictions(self, outputs):
        return outputs
        

# Sigmoid Activation Function
# Formula: f(x) = 1 / (1 + e^(-x))
# Output range: (0, 1)
# This function squashes the input values between 0 and 1, making it useful for binary classification problems.
class SigmoidActivation():
    def forward(self, inputs, training):
        # Apply the sigmoid function to the inputs and store the output
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        # Gradient of sigmoid function: f'(x) = f(x) * (1 - f(x))
        self.dinputs = dvalues * self.output * (1 - self.output)
        
    # Converts outputs to binary predictions based on a threshold of 0.5
    def predictions(self, outputs):
        return (outputs > 0.5) * 1
        
        
# Hyperbolic Tangent (Tanh) Activation Function
# Formula: f(x) = tanh(x) = (2 / (1 + e^(-2x))) - 1
# Output range: (-1, 1)
# Similar to sigmoid but outputs values between -1 and 1, making it zero-centered.
class TanhActivation():
    def forward(self, inputs, training):
        # Apply the tanh function to the inputs and store the output
        self.inputs = inputs
        self.output = np.tanh(inputs)

    def backward(self, dvalues):
        # Gradient of tanh function: f'(x) = 1 - (f(x))^2
        self.dinputs = dvalues * (1 - self.output ** 2)


# Rectified Linear Unit (ReLU) Activation Function
# Formula: f(x) = max(0, x)
# Output range: [0, ∞)
# It returns 0 for any negative input, while for positive input, it returns that value itself.
class ReLUActivation():
    def forward(self, inputs, training):
        # Apply the ReLU function to the inputs and store the output
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Gradient of ReLU function: 1 for x > 0, otherwise 0
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        
    # For ReLU, predictions are the same as the outputs, ALTHOUGH IT WONT BE USED ON THE OUTPUT LAYER
    def predictions(self, outputs):
        return outputs

# Softmax Activation Function
# Formula: f(xi) = e^(xi) / Σ(e^(xj)) for all j
# Output range: (0, 1) for each class, and the outputs sum to 1
# This function is used for multi-class classification problems. It converts scores to probabilities.
class SoftmaxActivation():
    def forward(self, inputs, training):
        # Subtract the max input value for numerical stability, then apply the softmax function
        # This subtraction does not affect the softmax probabilities but helps in preventing potential numerical overflow.
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Calculate the softmax probabilities. The division is done row-wise.
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
    def backward(self, dvalues):
        # Compute the gradient of softmax function using the Jacobian matrix
        # Initialize an array to hold the gradients of the inputs (dinputs)
        self.dinputs = np.empty_like(dvalues)
        # Iterate through each output and corresponding derivative
        for index, (singleOutput, singleDvalues) in enumerate(zip(self.output, dvalues)):
            # Reshape singleOutput to make it a column vector
            singleOutput = singleOutput.reshape(-1, 1)
            # Compute the Jacobian matrix for the current output
            # The Jacobian matrix for a softmax function is a square matrix where each element is
            # the partial derivative of the output with respect to each input.
            # For a softmax output 's' for classes [s1, s2, ..., sn], the Jacobian matrix J is:
            # J = diag(s) - s * s^T
            # diag(s) is a diagonal matrix with the softmax probabilities on the diagonal.
            # s * s^T is the outer product of the softmax vector with itself.
            jacobianMatrix = np.diagflat(singleOutput) - np.dot(singleOutput, singleOutput.T)
            # Multiply the Jacobian matrix by the derivative of the loss with respect to the outputs (dvalues)
            # to get the derivative of the loss with respect to the inputs.
            self.dinputs[index] = np.dot(jacobianMatrix, singleDvalues)
        
    # Returns the class index with the highest probability
    # This function is typically used during inference, to convert the softmax output probabilities to class labels.
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
        
class Loss:
    def regularizationLoss(self):
        regularizationLoss = 0
        for layer in self.trainableLayers:
            if layer.weight_L1Regularization > 0:
                regularizationLoss += layer.weight_L1Regularization * np.sum(np.abs(layer.weights))
            if layer.weight_L2Regularization > 0:
                regularizationLoss += layer.weight_L2Regularization * np.sum(layer.weights ** 2)
            if layer.bias_L1Regularization > 0:
                regularizationLoss += layer.bias_L1Regularization * np.sum(np.abs(layer.biases))
            if layer.bias_L2Regularization > 0:
                regularizationLoss += layer.bias_L2Regularization * np.sum(layer.biases ** 2)
        
        return regularizationLoss
    
    def rememberTrainableLayers(self, trainableLayers):
        self.trainableLayers = trainableLayers
    
    def calculate(self, predictions, targets, *, includeRegularization=False):
        sampleLosses = self.forward(predictions, targets)
        meanLoss = np.mean(sampleLosses)
        
        self.accumulatedLoss += np.sum(sampleLosses)
        self.accumulatedCount += len(sampleLosses)
        
        if not includeRegularization:
            return meanLoss
        return meanLoss, self.regularizationLoss()
    
    def calculateAccumulated(self, *, includeRegularization=False):
        meanLoss = self.accumulatedLoss / self.accumulatedCount
        
        if not includeRegularization:
            return meanLoss
        return meanLoss, self.regularizationLoss()
    
    def newPass(self):
        self.accumulatedLoss = 0
        self.accumulatedCount = 0

class MeanAbsoluteErrorLoss(Loss):
    # Formula: MAE = (1/N) * sum(abs(y_true - y_pred))
    # Summary: Measures the average magnitude of errors in predictions, without direction.
    def forward(self, predictions, targets):
        return np.mean(np.abs(predictions - targets), axis=-1)

    def backward(self, dvalues, targets):
        # Calculate gradient of loss w.r.t. predictions for Mean Absolute Error Loss
        # It is 1/N * sign(y_true - y_pred) for each prediction.
        # Meaning 1 for positive values and -1 for negative values.
        # This gives us the direction to move towards the target value.
        # The division by numOfOutputs normalizes the gradient across all outputs.
        # Finally, we average the gradients over all samples in the batch.
        numOfSamples = len(dvalues)
        numOfOutputs = len(dvalues[0])
        self.dinputs = np.sign(targets - dvalues) / numOfOutputs
        self.dinputs = self.dinputs / numOfSamples


class MeanSquaredErrorLoss(Loss):
    # Formula: MSE = (1/N) * sum((y_true - y_pred)^2)
    # Summary: Measures the average of the squares of the errors.
    def forward(self, predictions, targets):
        return np.mean((predictions - targets) ** 2, axis=-1)

    def backward(self, dvalues, targets):
        # Calculate gradient of loss w.r.t. predictions for Mean Squared Error Loss
        # It is 2/N * (y_pred - y_true) for each prediction.
        # This gradient points in the direction of the prediction error.
        numOfSamples = len(dvalues)
        numOfOutputs = len(dvalues[0])
        self.dinputs = -2 * (targets - dvalues) / numOfOutputs
        self.dinputs = self.dinputs / numOfSamples


class BinaryCrossEntropyLoss(Loss):
    # Formula: BCE = - (1/N) * sum(y_true*log(y_pred) + (1-y_true)*log(1-y_pred))
    # Summary: Measures the performance of a classification model whose output is a probability value between 0 and 1.
    def forward(self, predictions, targets):
        predictionsClipped = np.clip(predictions, 1e-15, 1 - 1e-15)
        allSamplesIndividualLosses = -(targets * np.log(predictionsClipped) + (1 - targets) * np.log(1 - predictionsClipped))
        return np.mean(allSamplesIndividualLosses, axis=-1)

    def backward(self, dvalues, targets):
        # Calculate gradient of loss w.r.t. predictions for Binary Cross Entropy Loss
        # The derivative of BCE with respect to the predictions is more complex due to the log function:
        # It involves the formula -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) for each prediction.
        # This gradient effectively measures how the change in prediction probability affects the loss.
        numOfSamples = len(dvalues)
        numOfOutputs = len(dvalues[0])
        predictionsClipped = np.clip(dvalues, 1e-15, 1 - 1e-15)
        self.dinputs = -(targets / predictionsClipped - (1 - targets) / (1 - predictionsClipped)) / numOfOutputs
        self.dinputs = self.dinputs / numOfSamples


class CategoricalCrossEntropyLoss(Loss):
    # Formula: CCE = - sum(y_true*log(y_pred))
    # Summary: Used when there are two or more label classes.
    def forward(self, predictions, targets):
        predictionsClipped = np.clip(predictions, 1e-15, 1 - 1e-15)
        numOfSamples = len(predictions)
        # Handle targets being provided as scalars (for single-label classification)
        if len(targets.shape) == 1:
            correctConfidences = predictionsClipped[range(numOfSamples), targets]
        # Handle one-hot encoded targets (for multi-label classification)
        elif len(targets.shape) == 2:
            correctConfidences = np.sum(predictionsClipped * targets, axis=1)
        # Calculate negative log likelihood, which quantifies the difference between
        # the predicted probabilities and the actual distribution (targets).
        return -np.log(correctConfidences)

    def backward(self, dvalues, targets):
        # Calculate gradient of loss w.r.t. predictions for Categorical Cross Entropy Loss
        # The formula -targets / predictions for each class probability in the predictions.
        # This gradient indicates how a small change in the predicted probability of the correct class
        # affects the overall loss, guiding the model to adjust its predictions.
        numOfSamples = len(dvalues)
        numOfLabels = len(dvalues[0])
        # Convert scalar targets to one-hot encoded format if they aren't already.
        # This is necessary for gradient calculation as it simplifies the process
        # of applying the derivative of the loss function.
        if len(targets.shape) == 1:
            targets = np.eye(numOfLabels)[targets]
        # Calculate the gradient of the loss function with respect to the inputs (dvalues).
        # This gradient will be used to update the weights in the opposite direction
        # of the gradient to minimize the loss.
        self.dinputs = -targets / dvalues
        # Normalize the gradients by the number of samples to avoid the scale of the gradients
        # being influenced by the batch size. This makes the learning rate more manageable.
        self.dinputs = self.dinputs / numOfSamples

class SoftmaxWithCategoricalCrossEntropyLoss():
    # The combination of Softmax activation and Categorical Cross Entropy loss simplifies backpropagation:
    # 1. During backpropagation, the derivative of the loss with respect to the input of the Softmax (z_i) is straightforward.
    # 2. The combined gradient formula of Softmax + Categorical Cross Entropy Loss is simplified to: dL/dz_i = y_hat_i - y_i.
    # 3. This simplification means that we can easily compute the gradients needed for updating the weights,
    #    making the training process more efficient and stable.

    # def __init__(self):
    #     self.activation = SoftmaxActivation()
    #     self.loss = CategoricalCrossEntropyLoss()

    # def forward(self, inputs, targets):
    #     self.activation.forward(inputs)
    #     self.output = self.activation.output
    #     return self.loss.calculate(self.output, targets)

    # Simplifies backpropagation by adjusting predictions towards correct classes
    def backward(self, dvalues, targets):
        numOfSamples = len(dvalues)
        # Convert one-hot encoded targets to class indices if necessary
        if len(targets.shape) == 2: targets = np.argmax(targets, axis=1)
        # Copy softmax output, then subtract 1 from the probabilities of the correct classes
        self.dinputs = dvalues.copy()
        self.dinputs[np.arange(numOfSamples), targets] -= 1
        # Normalize the gradient by the number of samples
        self.dinputs /= numOfSamples
        
# Optimizers Classes
class SGDOptimizer:
    """
    Stochastic Gradient Descent (SGD) optimizer with optional momentum.
    
    Parameters:
    - learningRate (float): Initial learning rate for weight updates.
    - lrDecay (float): Decay rate for the learning rate, applied each iteration.
    - momentum (float): Momentum factor, combining gradients from previous steps for smoother updates.
    
    Update Logic:
    - Adjusts the learning rate based on decay (if applicable).
    - Applies momentum to the update (if momentum > 0), otherwise directly uses the gradient.
    - Updates weights and biases based on the calculated update amounts.
    
    Update Formula:
    - If momentum > 0:
        weightUpdate = momentum * previousWeightUpdate - currLearningRate * dweights
        biasUpdate = momentum * previousBiasUpdate - currLearningRate * dbiases
    - Else:
        weightUpdate = -currLearningRate * dweights
        biasUpdate = -currLearningRate * dbiases
    
    """
    def __init__(self, learningRate=1.0, lrDecay=0, momentum=0):
        self.learningRate = learningRate
        self.currLearningRate = learningRate
        self.lrDecay = lrDecay
        self.iterations = 0
        self.momentum = momentum

    def beforeParamsUpdate(self):
        if self.lrDecay:
            self.currLearningRate = self.learningRate * (1.0 / (1.0 + self.lrDecay * self.iterations))
    
    def updateParameters(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weightMomentum'):
                layer.weightMomentum = np.zeros_like(layer.weights)
                layer.biasMomentum = np.zeros_like(layer.biases)
            weightsUpdateAmount = self.momentum * layer.weightMomentum - self.currLearningRate * layer.dweights
            layer.weightMomentum = weightsUpdateAmount
            biasesUpdateAmount = self.momentum * layer.biasMomentum - self.currLearningRate * layer.dbiases
            layer.biasMomentum = biasesUpdateAmount
        else:
            weightsUpdateAmount = -self.currLearningRate * layer.dweights
            biasesUpdateAmount = -self.currLearningRate * layer.dbiases
            
        layer.weights += weightsUpdateAmount
        layer.biases += biasesUpdateAmount
        
    def afterParamsUpdate(self):
        self.iterations += 1
        
class AdagradOptimizer():
    """
    Adagrad optimizer, adapting the learning rate to parameters.
    This introduces a per-parameter learning rate that improves learning for sparse data.
    Learning rate is decreased based on commulative squared gradients.
    
    Parameters:
    - learningRate (float): Initial learning rate.
    - lrDecay (float): Decay rate for the learning rate, applied each iteration.
    - epsilon (float): Small value to prevent division by zero.
    
    Update Logic:
    - Adjusts the learning rate based on decay (if applicable).
    - Accumulates squared gradients in a cache.
    - Adjusts updates based on the accumulated cache, allowing for an adaptive learning rate.
    
    Update Formula:
    - weightCache = weightCache + dweights ** 2
    - weightUpdate = -currLearningRate * dweights / (sqrt(weightCache) + epsilon)
    
    - biasCache = biasCache + dbiases ** 2
    - biasUpdate = -currLearningRate * dbiases / (sqrt(biasCache) + epsilon)
    """
    
    def __init__(self, learningRate=1 , lrDecay=0, epsilon=1e-7):
        self.learningRate = learningRate
        self.currLearningRate = learningRate
        self.lrDecay = lrDecay
        self.iterations = 0
        self.epsilon = epsilon
    
    def beforeParamsUpdate(self):
        if self.lrDecay:
            self.currLearningRate = self.learningRate * (1.0 / (1.0 + self.lrDecay * self.iterations))
    
    def updateParameters(self, layer):
        if not hasattr(layer, 'weightCache'):
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasCache = np.zeros_like(layer.biases)

        layer.weightCache += layer.dweights ** 2
        layer.biasCache += layer.dbiases ** 2
        
        layer.weights += -self.currLearningRate * layer.dweights / (np.sqrt(layer.weightCache) + self.epsilon)
        layer.biases += -self.currLearningRate * layer.dbiases / (np.sqrt(layer.biasCache) + self.epsilon)

    def afterParamsUpdate(self):
        self.iterations += 1
        
class RMSPropOptimizer():
    """
    RMSProp optimizer, improving upon Adagrad by using a moving average of squared gradients.
    This is helpful in increasing as well as decreasing the learning rate based on the gradient magnitudes.
    
    Parameters:
    - learningRate (float): Initial learning rate.
    - lrDecay (float): Decay rate for the learning rate, applied each iteration.
    - rho (float): Decay rate for the moving average of squared gradients.
    - epsilon (float): Small value to prevent division by zero.
    
    Update Logic:
    - Adjusts the learning rate based on decay (if applicable).
    - Updates the cache with a moving average of squared gradients.
    - Uses the cache to adaptively adjust learning rates for each parameter.
    
    Update Formula:
    - weightCache = rho * weightCache + (1 - rho) * dweights ** 2
    - weightUpdate = -currLearningRate * dweights / (sqrt(weightCache) + epsilon)
    
    - biasCache = rho * biasCache + (1 - rho) * dbiases ** 2
    - biasUpdate = -currLearningRate * dbiases / (sqrt(biasCache) + epsilon)
    """
    def __init__(self, learningRate=0.001 , lrDecay=0, rho=0.9, epsilon=1e-7):
        self.learningRate = learningRate
        self.currLearningRate = learningRate
        self.lrDecay = lrDecay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    
    def beforeParamsUpdate(self):
        if self.lrDecay:
            self.currLearningRate = self.learningRate * (1.0 / (1.0 + self.lrDecay * self.iterations))
    
    def updateParameters(self, layer):
        if not hasattr(layer, 'weightCache'):
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasCache = np.zeros_like(layer.biases)

        layer.weightCache = self.rho * layer.weightCache + (1 - self.rho) * layer.dweights ** 2
        layer.biasCache = self.rho * layer.biasCache + (1 - self.rho) * layer.dbiases ** 2
        
        layer.weights += -self.currLearningRate * layer.dweights / (np.sqrt(layer.weightCache) + self.epsilon)
        layer.biases += -self.currLearningRate * layer.dbiases / (np.sqrt(layer.biasCache) + self.epsilon)

    def afterParamsUpdate(self):
        self.iterations += 1
        
class AdamOptimizer():
    """
    Adam optimizer, combining momentum and adaptive learning rates.
    It combines the momentum from SGD and the adaptive learning rate from RMSProp.
    
    Parameters:
    - learningRate (float): Initial learning rate.
    - lrDecay (float): Decay rate for the learning rate, applied each iteration.
    - epsilon (float): Small value to prevent division by zero.
    - beta_1 (float): Exponential decay rate for the first moment estimates.
    - beta_2 (float): Exponential decay rate for the second moment estimates.
    
    Update Logic:
    - Adjusts the learning rate based on decay (if applicable).
    - Updates momentum and cache with bias-corrected first and second moment estimates.
    - Applies updates using the Adam algorithm for adaptive learning rates with momentum.
    
    Update Formula:
    - weightMomentum = beta_1 * weightMomentum + (1 - beta_1) * dweights
    - correctedWeightMomentum = weightMomentum / (1 - beta_1 ** (iterations + 1))
    - biasMomentum = beta_1 * biasMomentum + (1 - beta_1) * dbiases
    - correctedBiasMomentum = biasMomentum / (1 - beta_1 ** (iterations + 1))
    
    - weightCache = beta_2 * weightCache + (1 - beta_2) * dweights ** 2
    - correctedWeightCache = weightCache / (1 - beta_2 ** (iterations + 1))
    - biasCache = beta_2 * biasCache + (1 - beta_2) * dbiases ** 2
    - correctedBiasCache = biasCache / (1 - beta_2 ** (iterations + 1))
    
    - weightUpdate = -currLearningRate * correctedWeightMomentum / (sqrt(correctedWeightCache) + epsilon)
    - biasUpdate = -currLearningRate * correctedBiasMomentum / (sqrt(correctedBiasCache) + epsilon)
    """
    
    def __init__(self, learningRate=0.001, lrDecay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learningRate = learningRate
        self.currLearningRate = learningRate
        self.lrDecay = lrDecay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def beforeParamsUpdate(self):
        if self.lrDecay:
            self.currLearningRate = self.learningRate * (1. / (1. + self.lrDecay * self.iterations))

    def updateParameters(self, layer):
        if not hasattr(layer, 'weightCache'):
            layer.weightMomentum = np.zeros_like(layer.weights)
            layer.biasMomentum = np.zeros_like(layer.biases)

        layer.weightMomentum = self.beta_1 * layer.weightMomentum + (1 - self.beta_1) * layer.dweights
        layer.biasMomentum = self.beta_1 * layer.biasMomentum + (1 - self.beta_1) * layer.dbiases

        weightMomentum_corrected = layer.weightMomentum / (1 - self.beta_1 ** (self.iterations + 1))
        biasMomentum_corrected = layer.biasMomentum / (1 - self.beta_1 ** (self.iterations + 1))

        if not hasattr(layer, 'weightCache'):
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasCache = np.zeros_like(layer.biases)
        
        layer.weightCache = self.beta_2 * layer.weightCache + (1 - self.beta_2) * layer.dweights**2
        layer.biasCache = self.beta_2 * layer.biasCache + (1 - self.beta_2) * layer.dbiases**2

        weightCache_corrected = layer.weightCache / (1 - self.beta_2 ** (self.iterations + 1))
        biasCache_corrected = layer.biasCache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.currLearningRate * weightMomentum_corrected / (np.sqrt(weightCache_corrected) + self.epsilon)
        layer.biases += -self.currLearningRate * biasMomentum_corrected / (np.sqrt(biasCache_corrected) + self.epsilon)
        
    def afterParamsUpdate(self):
        self.iterations += 1
    
class Accuracy:
    # This class provides methods to calculate accuracy for models.
    def calculate(self, predictions, y):
        # Compares predictions with actual values to determine accuracy.
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        
        # Updates accumulated correct and total predictions.
        self.accumulatedCorrectPredictions += np.sum(comparisons)
        self.accumulatedTotalPredictions += len(comparisons)

        return accuracy
    
    def calculateAccumulated(self):
        # Calculates overall accuracy percentage from accumulated values.
        accuracy = self.accumulatedCorrectPredictions / self.accumulatedTotalPredictions
        return accuracy

    def newPass(self):
        # Resets accumulated predictions for a new evaluation pass.
        self.accumulatedCorrectPredictions = 0
        self.accumulatedTotalPredictions = 0

class categoricalAccuracy(Accuracy):
    # Handles accuracy calculation for classification tasks.
    def __init__(self, *, binary=False):
        self.binary = binary  # Indicates if the task is binary classification or not.

    def init(self, y):
        pass

    def compare(self, predictions, y):
        # Adjusts labels for non-binary tasks and compares with predictions.
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

class regressionAccuracy(Accuracy):
    # Handles accuracy calculation for regression tasks.
    def __init__(self):
        self.precision = None  # Precision threshold for considering predictions accurate.

    def init(self, y, reinit=False):
        # Determines precision based on the standard deviation of y.
        # We can change factor to a different value if needed.
        # Increasing the factor will make precision more strict.
        
        factor = 250
        if self.precision is None or reinit:
            self.precision = np.std(y) / factor

    def compare(self, predictions, y):
        # Compares predictions with actual values within a precision threshold.
        return np.absolute(predictions - y) < self.precision

class ApexNetModel:
    def __init__(self):
        # Initialize the model with an empty list of layers and softmax and categorical cross-entropy loss combination to none
        self.layers = []
        self.softmaxCategoricalLossCombo = None

    def add(self, layer):
        # Add a new layer or object to the model's layer list
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        # Set the model's loss function, optimizer, and accuracy metric
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        # Prepare the model for training or inference by setting up layer connections and trainable layers
        
        # Initialize the input layer because the loop is designed in such a way that it works on previous layers output
        # So, we need to have a placeholder for the input layer which will pass the inputs as output to first hidden layer
        self.input_layer = FirstLayerInput()

        # Calculate the total number of layers
        numberOfLayers = len(self.layers)
        # List to keep track of layers with weights (trainable layers)
        self.trainableLayers = []

        # Connect layers sequentially
        for i in range(numberOfLayers):
            if i == 0:
                # First layer connects to the input layer and the next layer
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < numberOfLayers - 1:
                # Middle layers connect to their previous and next layers
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                # Last layer connects to its previous layer and the loss function
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                # Keep a reference to the last layer's activation function
                # We do this because the last layer could be a Softmax activation
                self.lastLayerActivation = self.layers[i]

            # Identify trainable layers (layers with weights)
            if hasattr(self.layers[i], 'weights'):
                self.trainableLayers.append(self.layers[i])

        # Inform the loss function about the trainable layers
        self.loss.rememberTrainableLayers(self.trainableLayers)

        # Optimize if the last layer is a Softmax and the loss is Categorical Cross-Entropy
        # and the combined Softmax and Categorical Cross-Entropy loss is used for efficiency
        if isinstance(self.layers[-1], SoftmaxActivation) and isinstance(self.loss, CategoricalCrossEntropyLoss):
            self.softmaxCategoricalLossCombo = SoftmaxWithCategoricalCrossEntropyLoss()

    def train(self, X, y, *, epochs=1, batch_size=None, printFrequency=1, validationData=None):
        self.finalize()
        # Initialize accuracy object
        # We do this here because we need to pass the y value to the accuracy object in case we need to 
        # define precision based on the standard deviation of y in regression tasks
        self.accuracy.init(y)

        # Default value if batch size is not being set
        trainingSteps = 1

        # Calculate number of steps needed for training
        if batch_size is not None:
            trainingSteps = len(X) // batch_size
            # In case there are some remaining data but not a full batch
            # Add `1` to include this incomplete batch
            if trainingSteps * batch_size < len(X):
                trainingSteps += 1

        # Main training loop
        for epoch in range(1, epochs + 1):
            print(f'epoch: {epoch}')

            # Reset accumulated values in loss and accuracy objects
            self.loss.newPass()
            self.accuracy.newPass()

            # Iterate over steps
            for step in range(trainingSteps):
                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                else:
                    # we set the batch_X and batch_y to the current batch
                    # by using formula: start = step * batch_size, end = (step + 1) * batch_size
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                # Perform the forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                dataLoss, regularization_loss = self.loss.calculate(output, batch_y, includeRegularization=True)
                loss = dataLoss + regularization_loss

                # Get predictions and calculate an accuracy
                predictions = self.lastLayerActivation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.beforeParamsUpdate()
                for layer in self.trainableLayers:
                    self.optimizer.updateParameters(layer)
                self.optimizer.afterParamsUpdate()

                # Print a summary every `printFrequency` steps or if it's the last step
                if not step % printFrequency or step == trainingSteps - 1:
                    print(
                        f"Step: {step}\n"
                        f"Accuracy: {accuracy * 100:.3f}\n"
                        f"Total Loss: {loss:.3f}\n"
                        f"  - Data Loss: {dataLoss:.3f}\n"
                        f"  - Regularization Loss: {regularization_loss:.3f}\n"
                        f"Learning Rate: {self.optimizer.currLearningRate}"
                    )
                    print("----------------------------------------")

            # Get and print epoch loss and accuracy values
            epoch_data_loss, epoch_regularization_loss = self.loss.calculateAccumulated(includeRegularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculateAccumulated()

            print(
                f"Training:\n"
                f"Accuracy: {epoch_accuracy * 100:.3f}\n"
                f"Total Loss: {epoch_loss:.3f}\n"
                f"  - Data Loss: {epoch_data_loss:.3f}\n"
                f"  - Regularization Loss: {epoch_regularization_loss:.3f}\n"
                f"Learning Rate: {self.optimizer.currLearningRate}"
            )
            print("-----------------------------------------------------------------------------------------")

            # If there is the validation data
            if validationData is not None:
                # Evaluate the model:
                self.evaluate(*validationData, batch_size=batch_size) 
                
            print("=========================================================================================")

    def evaluate(self, X_val, y_val, *, batch_size=None):
        # Default value if batch size is not being set
        validationSteps = 1

        # Calculate number of steps for validation loop
        if batch_size is not None:
            validationSteps = len(X_val) // batch_size
            # In case there are some remaining data but not a full batch
            # Add `1` to include this incomplete batch
            if validationSteps * batch_size < len(X_val):
                validationSteps += 1

        # Reset accumulated values in the accuracy object
        self.accuracy.newPass()
        self.loss.newPass()

        # Iterate over steps
        for step in range(validationSteps):
            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
                
            # Otherwise slice a batch using the formula: start = step * batch_size, end = (step + 1) * batch_size
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            # Perform the forward pass
            output = self.forward(batch_X, training=False)

            self.loss.calculate(output, batch_y)
            # Get predictions and calculate an accuracy using Activation object saved during finalize function
            predictions = self.lastLayerActivation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        # Get and print validation loss and accuracy values
        validation_loss = self.loss.calculateAccumulated()
        validation_accuracy = self.accuracy.calculateAccumulated()
        # Print a summary
        print(
            f"Validation:\n"
            f"Accuracy: {validation_accuracy * 100:.3f}\n"
            f"Loss: {validation_loss:.3f}"
        )
        print("-----------------------------------------------------------------------------------------")
    
    # Predicts on the Test samples
    def predict(self, X, *, batch_size=None):
        # Default value if batch size is not being set
        predictionSteps = 1

        # Calculate number of steps for testing loop
        if batch_size is not None:
            predictionSteps = len(X) // batch_size
            # In case there are some remaining data but not a full batch
            # Add `1` to include this incomplete batch
            if predictionSteps * batch_size < len(X):
                predictionSteps += 1

        # Model outputs
        output = []

        # Iterate over steps
        for step in range(predictionSteps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)
            # Append batch prediction to the list of predictions
            output.append(batch_output)
        # Stack and return results
        return np.vstack(output)
    
    # Performs forward pass
    def forward(self, X, training):
        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain meaning output of the previous layer is input for the next layer
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # "layer" is now the last object from the list,
        # This will be the output of the model after the forward pass
        return layer.output

    # Performs backward pass
    def backward(self, output, y):
        # If combined Softmax and Categorical Cross-Entropy loss is used
        if self.softmaxCategoricalLossCombo is not None:
            self.softmaxCategoricalLossCombo.backward(output, y)
            # Skip the backward method of the last layer because it is Softmax activation
            self.layers[-1].dinputs = self.softmaxCategoricalLossCombo.dinputs

            # Call backward method in reverse order going through all the objects in the layers
            # except the last one (Softmax activation)
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            # Return from the function
            return


        # First call backward method on the loss function
        # this will set dinputs property, that the last layer will try to access
        self.loss.backward(output, y)

        # Call backward method in reverse order going through all the objects in the layers
        # with parameters (dinputs) from the next object in the list
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)