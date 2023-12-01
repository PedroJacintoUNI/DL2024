
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
'''
x: single observation of shape (n,)
weights: list of weight matrices [W1, W2, ...]
biases: list of biases matrices [b1, b2, ...]

y: final output
hiddens: list of computed hidden layers [h1, h2, ...]
'''

def forward(x, weights, biases):
    num_layers = len(weights)
    g = np.tanh
    hiddens = []
    # compute hidden layers
    for i in range(num_layers):
            h = x if i == 0 else hiddens[i-1]
            z = weights[i].dot(h) + biases[i]
            if i < num_layers-1:  # Assuming the output layer has no activation.
                hiddens.append(g(z))
    #compute output
    output = z
    
    return output, hiddens

def compute_loss(output, y):
    # compute loss
    probs = np.exp(output) / np.sum(np.exp(output))
    loss = -y.dot(np.log(probs))
    
    return loss   

def backward(x, y, output, hiddens, weights):
    num_layers = len(weights)
    g = np.tanh
    z = output

    probs = np.exp(output) / np.sum(np.exp(output))
    grad_z = probs - y  
    
    grad_weights = []
    grad_biases = []
    
    # Backpropagate gradient computations 
    for i in range(num_layers-1, -1, -1):
        
        # Gradient of hidden parameters.
        h = x if i == 0 else hiddens[i-1]
        grad_weights.append(grad_z[:, None].dot(h[:, None].T))
        grad_biases.append(grad_z)
        
        # Gradient of hidden layer below.
        grad_h = weights[i].T.dot(grad_z)

        # Gradient of hidden layer below before activation.
        grad_z = grad_h * (1-h**2)   # Grad of loss wrt z3.

    # Making gradient vectors have the correct order
    grad_weights.reverse()
    grad_biases.reverse()
    return grad_weights, grad_biases



data = load_digits()

inputs = data.data  
labels = data.target  
n, p = np.shape(inputs)
n_classes = len(np.unique(labels))


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

'''
Outputs:
    - weights: list of updated weights
    - biases: list of updated 
    - loss: scalar of total loss (sum for all observations)

'''

def MLP_train_epoch(inputs, labels, weights, biases):
    num_layers = len(weights)
    total_loss = 0
    # For each observation and target
    for x, y in zip(inputs, labels):
        # Comoute forward pass
        output, hiddens = forward(x, weights, biases)
        
        # Compute Loss and Update total loss
        loss = compute_loss(output, y)
        total_loss+=loss
        # Compute backpropagation
        grad_weights, grad_biases = backward(x, y, output, hiddens, weights)
        
        # Update weights
        
        for i in range(num_layers):
            weights[i] -= eta*grad_weights[i]
            biases[i] -= eta*grad_biases[i]
            
    return weights, biases, total_loss

# Initialize weights

# Single hidden unit MLP with 50 hidden units.
# First is input size, last is output size.
units = [64, 50, 10]

# Initialize all weights and biases randomly.
W1 = .1 * np.random.randn(units[1], units[0])
b1 = .1 * np.random.randn(units[1])
W2 = .1 * np.random.randn(units[2], units[1])
b2 = .1 * np.random.randn(units[2])

weights = [W1, W2]
biases = [b1, b2]

# Learning rate.
eta = 0.001  
    
# Run epochs

losses = []

for epoch in range(100):
    weights, biases, loss = MLP_train_epoch(X_train, y_train_ohe, weights, biases)
    losses.append(loss)
    
plt.plot(losses)
plt.show()

# %% [markdown]
# ❓ Complete function `MLP_predict` to get array of predictions from your trained MLP:

# %%
def MLP_predict(inputs, weights, biases):
    predicted_labels = []
    for x in inputs:
        # Compute forward pass and get the class with the highest probability
        output, _ = forward(x, weights, biases)
        y_hat = np.argmax(output)
        predicted_labels.append(y_hat)
    predicted_labels = np.array(predicted_labels)
    return predicted_labels

y_train_pred = MLP_predict(X_train, weights, biases)
y_test_pred = MLP_predict(X_test, weights, biases)

print(f'Train accuracy: {(y_train_pred==y_train).mean()}')
print(f'Test accuracy: {(y_test_pred==y_test).mean()}')

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(50),
                    activation='tanh',
                    solver='sgd',
                    learning_rate='constant',
                    learning_rate_init=0.001,
                    nesterovs_momentum=False,
                    random_state=1,
                    max_iter=1000)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))


