######################################
# Assignement 2 for CSC420
# MNIST clasification example
# Author: Jun Gao
######################################

import numpy as np

def cross_entropy_loss_function(prediction, label):
    #TODO: compute the cross entropy loss function between the prediction and ground truth label.
    # prediction: the output of a neural network after softmax. It can be an Nxd matrix, where N is the number of samples,
    #           and d is the number of different categories
    # label: The ground truth labels, it can be a vector with length N, and each element in this vector stores the ground truth category for each sample.
    # Note: we take the average among N different samples to get the final loss.
    return - np.sum(np.dot(label, np.log(prediction))) / label.shape[0]

def sigmoid(x):
    # TODO: compute the softmax with the input x: y = 1 / (1 + exp(-x))
    return 1 / (1 + np.exp(-x))

def softmax(x):
    # TODO: compute the softmax function with input x.
    #  Suppose x is Nxd matrix, and we do softmax across the last dimention of it.
    #  For each row of this matrix, we compute x_{j, i} = exp(x_{j, i}) / \sum_{k=1}^d exp(x_{j, k})
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

class OneLayerNN():
    def __init__(self, num_input_unit, num_output_unit):
        #TODO: Random Initliaize the weight matrixs for a one-layer MLP.
        # the number of units in each layer is specified in the arguments
        # Note: We recommend using np.random.randn() to initialize the weight matrix: zero mean and the variance equals to 1
        #       and initialize the bias matrix as full zero using np.zeros()
        self.weights = np.random.randn(num_input_unit, num_output_unit)
        # Initialize bias vector with shape (1, num_output_units) as zeros
        self.biases = np.zeros((1, num_output_unit))

    def forward(self, input_x):
        #TODO: Compute the output of this neural network with the given input.
        # Suppose input_x is an Nxd matrix, where N is the number of samples and d is the number of dimension for each sample.
        # Compute output: z = softmax (input_x * W_1 + b_1), where W_1, b_1 are weights, biases for this layer
        # Note: If we only have one layer in the whole model and we want to use it to do classification,
        #       then we directly apply softmax **without** using sigmoid (or relu) activation
        self.N = input_x.shape[0]
        self.z = np.dot(input_x, self.weights) + self.biases
        self.y = softmax(self.z)
        return self.y


    def backpropagation_with_gradient_descent(self, loss, learning_rate, input_x, label):
        #TODO: given the computed loss (a scalar value), compute the gradient from loss into the weight matrix and running gradient descent
        # Note that you may need to store some intermidiate value when you do forward pass, such that you don't need to recompute them
        # Suggestions: you need to first write down the math for the gradient, then implement it to compute the gradient

        labels_one_hot = np.zeros((self.N, self.y.shape[1]))
        labels_one_hot[np.arange(self.N), label] = 1

        # Compute the gradient of the loss with respect to logits (z)
        dL_dz = self.y - labels_one_hot

        # Compute gradients with respect to weights and biases
        dL_dW = np.dot(input_x.T, dL_dz) / self.N
        dL_db = np.sum(dL_dz, axis=0, keepdims=True) / self.N

        # Update weights and biases
        self.weights -= learning_rate * dL_dW
        self.biases -= learning_rate * dL_db


# [Bonus points] This is not necessary for this assignment
class TwoLayerNN():
    def __init__(self, num_input_unit, num_hidden_unit, num_output_unit):
        #TODO: Random Initliaize the weight matrixs for a two-layer MLP with sigmoid activation,
        # the number of units in each layer is specified in the arguments
        # Note: We recommend using np.random.randn() to initialize the weight matrix: zero mean and the variance equals to 1
        #       and initialize the bias matrix as full zero using np.zeros()
        self.W1 = np.random.randn(num_input_unit, num_hidden_unit)
        self.b1 = np.zeros((1, num_hidden_unit))
        self.W2 = np.random.randn(num_hidden_unit, num_output_unit)
        self.b2 = np.zeros((1, num_output_unit))

    def forward(self, input_x):
        #TODO: Compute the output of this neural network with the given input.
        # Suppose input_x is Nxd matrix, where N is the number of samples and d is the number of dimension for each sample.
        # Compute: first layer: z = sigmoid (input_x * W_1 + b_1) # W_1, b_1 are weights, biases for the first layer
        # Compute: second layer: o = softmax (z * W_2 + b_2) # W_2, b_2 are weights, biases for the second layer
        self.N = input_x.shape[0]
        self.z1 = np.dot(input_x, self.W1) + self.b1
        self.h1 = sigmoid(self.z1)
        self.z2 = np.dot(self.h1, self.W2) + self.b2
        self.y = softmax(self.z2)
        return self.y

    def backpropagation_with_gradient_descent(self, loss, learning_rate, input_x, label):
        #TODO: given the computed loss (a scalar value), compute the gradient from loss into the weight matrix and running gradient descent
        # Note that you may need to store some intermidiate value when you do forward pass, such that you don't need to recompute them
        # Suggestions: you need to first write down the math for the gradient, then implement it to compute the gradient
        labels_one_hot = np.zeros((self.N, self.y.shape[1]))
        labels_one_hot[np.arange(self.N), label] = 1

        # Compute the gradient of the loss with respect to logits (z)
        dL_dz2 = self.y - labels_one_hot
        dL_dW2 = np.dot(self.h1.T, dL_dz2) / self.N
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True) / self.N
        dL_dz1 = np.dot(dL_dz2, self.W2.T) * self.h1 * (1 - self.h1)
        dL_dW1 = np.dot(input_x.T, dL_dz1) / self.N
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True) / self.N

        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1


