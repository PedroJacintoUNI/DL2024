import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def forward(x, weights, biases):
    num_layers = len(weights)
    g = np.tanh
    hiddens = []
    
    # compute hidden layers
    for i in range(num_layers):
        h = x if i == 0 else hiddens[i-1]
        z = weights[i].dot(h) + biases[i]
        if i < num_layers-1:  
            hiddens.append(g(z))
    #compute output
    output = z
    
    return output, hiddens

def compute_loss(output, y):
    loss = 0.5 * (output - y).dot(output - y)
    return loss 

def backward(x, y, output, hiddens, weights):
    num_layers = len(weights)
    g = np.tanh
    z = output
    
    grad_z = z - y  # Grad of loss wrt last z.
    
    grad_weights = []
    grad_biases = []
    for i in range(num_layers-1, -1, -1):
        # Gradient of hidden parameters.
        h = x if i == 0 else hiddens[i-1]
        
        grad_weights.append(grad_z[:, None].dot(h[:, None].T))
        grad_biases.append(grad_z)

        # Gradient of hidden layer below.
        grad_h = weights[i].T.dot(grad_z)

        # Gradient of hidden layer below before activation.
        assert(g == np.tanh)
        grad_z = grad_h * (1-h**2)   # Grad of loss wrt z3.

    grad_weights.reverse()
    grad_biases.reverse()
    return grad_weights, grad_biases

def MLP_train_epoch(inputs, labels, weights, biases):
    num_layers = len(weights)
    total_loss = 0.0
    eta = 1
    
    # For each observation and target
    for x, y in zip(inputs, labels):
        # Compute forward pass
        output, hiddens = forward(x, weights, biases)
        
        # Compute Loss and update total loss
        loss = compute_loss(output, y)
        total_loss += loss
        # Compute backpropagation
        grad_weights, grad_biases = backward(x, y, output, hiddens, weights)      
        # Update weights
        for i in range(num_layers):
            weights[i] -= eta*grad_weights[i]
            biases[i] -= eta*grad_biases[i]
            
    return weights, biases, total_loss

def MLP_predict(inputs, weights, biases):
    predicted_labels = []
    for x in inputs:
        # Compute forward pass and get the class with the highest probability
        output, _ = forward(x, weights, biases)
        y_hat = np.zeros_like(output)
        y_hat[np.argmax(output)] = 1
        predicted_labels.append(y_hat)
    predicted_labels = np.array(predicted_labels)
    return predicted_labels

inputs = np.array([[1, 0, 1, 0]])
labels = np.array([[0, 1, 0]])

# First is input size, last is output size.
units = [4, 4, 3, 3]

# Initialize weights with correct shapes 
W1 = 0.1 * np.ones((units[1], units[0]))
b1 = 0.1 * np.ones(units[1])

W2 = 0.1 * np.ones((units[2], units[1]))
b2 = 0.1 * np.ones(units[2])

W3 = 0.1 * np.ones((units[3], units[2]))
b3 = 0.1 * np.ones(units[3])

weights = [W1, W2, W3]
biases = [b1, b2, b3]


# Empty loss list
loss = []
# Learning rate.
eta = 0.1
    
# Run epochs and append loss to list
for epoch in range(10):
    print("Epoch:", epoch)
    weights, biases, current_loss = MLP_train_epoch(inputs, labels, weights, biases)
    loss.append(current_loss)
    predicted_labels = MLP_predict(inputs, weights, biases)
    
    # Calculate accuracy
    correct_predictions = (np.argmax(predicted_labels, axis=1) == np.argmax(labels, axis=1)).sum()
    total_samples = labels.shape[0]
    accuracy = correct_predictions / total_samples
    print("Accuracy:", accuracy)
    print("Loss:", current_loss)
