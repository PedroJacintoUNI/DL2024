import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def forward(x, weights, biases):
    num_layers = len(weights)
    g = np.tanh
    hiddens = []
    for i in range(num_layers):
        h = x if i == 0 else hiddens[i-1]
        z = weights[i].dot(h) + biases[i]
        if i < num_layers-1:  # Assume the output layer has no activation.
            hiddens.append(g(z))
    output = z
    # For classification this is a vector of logits (label scores).
    # For regression this is a vector of predictions.
    output = np.exp(z) / np.sum(np.exp(z), axis=0)
    return output, hiddens

def compute_label_probabilities(output):
    # softmax transformation.
    probs = np.exp(output) / np.sum(np.exp(output))
    return probs

def compute_loss(output, y, loss_function='squared'):
    if loss_function == 'squared':
        y_pred = output
        loss = .5*(y_pred - y).dot(y_pred - y)
    elif loss_function == 'cross_entropy':
        # softmax transformation.
        probs = compute_label_probabilities(output)
        loss = -y.dot(np.log(probs))
    return loss

def predict_label(output):
    # The most probable label is also the label with the largest logit.
    y_hat = np.zeros_like(output)
    y_hat[np.argmax(output)] = 1
    return y_hat

def backward(x, y, output, hiddens, weights, loss_function='squared'):
    num_layers = len(weights)
    g = np.tanh
    z = output
    if loss_function == 'squared':
        grad_z = z - y  # Grad of loss wrt last z.
    elif loss_function == 'cross_entropy':
        # softmax transformation.
        probs = compute_label_probabilities(output)
        grad_z = probs - y  # Grad of loss wrt last z.
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

def update_parameters(weights, biases, grad_weights, grad_biases, eta):
    num_layers = len(weights)
    for i in range(num_layers):
        weights[i] -= eta*grad_weights[i]
        biases[i] -= eta*grad_biases[i]
        
def MLP_train_epoch(inputs, labels, weights, biases, eta=1, loss_function='squared'):
    total_loss = 0
    for x, y in zip(inputs, labels):
        output, hiddens = forward(x, weights, biases)
        loss = compute_loss(output, y, loss_function=loss_function)
        total_loss += loss
        grad_weights, grad_biases = backward(x, y, output, hiddens, weights, loss_function=loss_function)
        update_parameters(weights, biases, grad_weights, grad_biases, eta=eta)
    print("Total loss: %f" % total_loss)

def MLP_predict(inputs, weights, biases):
    predicted_labels = []
    for x in inputs:
        output, _ = forward(x, weights, biases)
        y_hat = predict_label(output)
        predicted_labels.append(y_hat)
    predicted_labels = np.array(predicted_labels)
    return predicted_labels

def evaluate(predicted_labels, gold_labels):
    print("Accuracy: %f" % np.mean(np.argmax(predicted_labels, axis=1) == np.argmax(gold_labels, axis=1)))
    
data = load_digits()

inputs = data.data  
labels = data.target  
n, p = np.shape(inputs)
n_classes = len(np.unique(labels))
print(n_classes)


X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

# Encode labels as one-hot vectors.
one_hot = np.zeros((np.size(y_train, 0), n_classes))
for i in range(np.size(y_train, 0)):
    one_hot[i, y_train[i]] = 1
y_train_ohe = one_hot
one_hot = np.zeros((np.size(y_test, 0), n_classes))
for i in range(np.size(y_test, 0)):
    one_hot[i, y_test[i]] = 1
y_test_ohe = one_hot

units =[64, 50, 10]

# Initialize weights
W1 = 0.1 * np.ones((units[1], units[0]))
b1 = 0.1 * np.ones(units[1])

W2 = 0.1 * np.ones((units[2], units[1]))
b2 = 0.1 * np.ones(units[2])


weights = [W1, W2]
biases = [b1, b2]

# Learning rate.
eta = 0.001  
    
# Run 10 epochs of SGD to train the MLP.
for epoch in range(50):
    MLP_train_epoch(X_train, y_train_ohe, weights, biases, eta=eta, loss_function='squared')
    predicted_labels = MLP_predict(X_train, weights, biases)
    evaluate(predicted_labels, y_train_ohe)