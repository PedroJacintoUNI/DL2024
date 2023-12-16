#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        y_hat_i = self.predict(x_i)
        
        if (y_hat_i != y_i):
            self.W[y_i] += x_i
            self.W[y_hat_i] -= x_i
        #raise NotImplementedError

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b
        #Z
        Z = np.sum(np.exp(np.dot(self.W, x_i)))
        
        #get probabilities
        prob = np.exp(np.dot(self.W, x_i)) / Z
        
        #gradient
        ey = np.identity(len(self.W))
        grad = np.outer(ey[y_i] - prob, x_i)
        
        #update weights
        self.W = self.W + learning_rate * grad
        #raise NotImplementedError

def Relu(x):
    return (np.maximum(0,x))

class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        W1 = np.random.normal(0.1,0.1,(hidden_size,n_features))
        W2 = np.random.normal(0.1,0.1,(n_classes,hidden_size))
        b1 = np.zeros(hidden_size)
        b2 = np.zeros(n_classes)
        
        self.weights = [W1,W2]
        self.biases = [b1,b2]
        
        #raise NotImplementedError
    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.     
        predicted_labels = []
        for x_i in X:
            output , _= self.forward(x_i)
            probs = np.exp(output[1]) / np.sum(np.exp(output[1]))
            predicted_labels.append(np.argmax(probs, axis = 0))
        predicted_labels = np.array(predicted_labels)
            
        return predicted_labels
        #raise NotImplementedError
    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible
    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Dont forget to return the loss of the epoch.
        """
        #Encode labels as one-hot
        n_classes = np.unique(y).size
        one_hot = np.zeros((np.size(y, 0), n_classes))
        for i in range(np.size(y, 0)):
            one_hot[i, y[i]] = 1
        y_hot = one_hot
        
        total_loss = 0.0
        
        for x_i, y_i in zip (X,y_hot):
            output, hiddens = self.forward(x_i)
            loss = self.compute_loss(y_i, output)
            grad_weights, grad_bias = self.backwards(x_i, y_i, output, hiddens)
            self.update_parameters(grad_weights, grad_bias, learning_rate)
            
            total_loss += loss
            
        return total_loss
    #def forward
    def forward(self, x):
        num_layers = len(self.weights)
        #hidden layers
        hiddens = []
        #z values from start to finish
        output = []
        
        for i in range(num_layers):
            h = x if i == 0 else hiddens[i -1]
            z = self.weights[i].dot(h) + self.biases[i]
            if i < num_layers -1:
                hiddens.append(Relu(z))
            if i == num_layers-1:
                z -= np.max(z)
            output.append(z)
        return output, hiddens   
    #def backward
    def backwards(self, x, y, z_values, hiddens):
        num_layers = len(self.weights)
        
        #softmax calculation
        probs = np.exp(z_values[num_layers-1]) / np.sum(np.exp(z_values[num_layers-1]))
        #gradient calculation
        grad_z = probs - y
        
        grad_weights = []
        grad_bias = []
        for i in range(num_layers-1, -1, -1):
            # Gradient of hidden parameters.
            h = x if i == 0 else hiddens[i-1]
        
            grad_weights.append(grad_z[:, None].dot(h[:, None].T))
            grad_bias.append(grad_z)

            # Gradient of hidden layer below.
            grad_h = self.weights[i].T.dot(grad_z)
            
            if i >= 1:
                mask = z_values[i-1]>0
                grad_z = grad_h*mask
             
        grad_weights.reverse()
        grad_bias.reverse()
        return grad_weights, grad_bias
             
    #def update parameters
    def update_parameters(self, grad_weights, grad_bias, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate*grad_weights[i]
            self.biases[i] -= learning_rate*grad_bias[i]
           
    def compute_loss(self, y, y_prob):
        num_layers = len(self.weights)

        
        #softmax calculation
        probs = np.exp(y_prob[num_layers-1]) / np.sum(np.exp(y_prob[num_layers-1]), axis=0)
        
        loss = - y.dot(np.log(probs))
        
        return loss     
        
def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
