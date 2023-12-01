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
        scores = np.dot(self.W, x_i)
        predicted_class = np.argmax(scores)

        # Compute the gradient only for the predicted class
        gradient = np.zeros_like(self.W)
        gradient[predicted_class, :] = x_i

        # Update weights using SGD
        self.W -= self.learning_rate * gradient
        
        #raise NotImplementedError

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
        y_hat = self.predict(x_i)
        
        if y_hat != y_i:
            #addition for the corrected class
            self.W[y_i] += x_i
            #subtraction for the predicted class
            self.W[y_hat] -= x_i
            
        
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


def ReLU(x):
    return(np.maximum(0,x))

class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        W1 = np.random.normal(0.1, 0.1, (hidden_size, n_features))
        W2 = np.random.normal(0.1,0.1, (n_classes, hidden_size))
        
        b1 = np.zeros(hidden_size)
        b2 = np.zeros(n_classes)
        
        self.weights = [W1, W2]
        self.bias = [b1, b2]
        
        #raise NotImplementedError

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        prediction = []
        for x_i in X:
            _ , output, _ = self.forward(x_i)
            probs = np.exp(output) / np.sum(np.exp(output))
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
        total_loss = 0.0
        num_samples = len(X)
        for x_i, y_i in zip(X, y):
            #forward pass
            z1, z2, h = self.forward(x_i)
            #backward pass
            grad_weights, grad_biases = self.backward(x_i,y_i,z1,z2,h)
            #update parameters
            self.update_parameters(grad_weights,grad_biases,learning_rate)
            
            #calculate loss
            loss_i = np.mean((y_i - h)**2)  
            total_loss += loss_i

        # Calculate average loss for the epoch
        avg_loss = total_loss / num_samples
        return avg_loss
        

    def forward(self, x):
        z1 = (self.weights[0]).dot(x) + self.bias[0]
        h1 = ReLU(z1)
        z2 = (self.weights[1]).dot(h1) + self.bias[1]
        z2 -= np.max(z2)
        
        return z1, z2, h1
    
    def backward(self, x, y, z1, z2, h1):
        probs = np.exp(z2) / np.sum(np.exp(z2))
        output = np.zeros(self.weights[1].shape[0])
        output[y] = 1
        
        grad_z2 = probs - output
        grad_weights = []
        grad_biases = []
        
        grad_weights.append(grad_z2[:,None].dot(h1[:,None].T) )
        grad_biases.append( grad_z2 ) #grad_weights/biases[1]
        
        grad_h1 = self.weights[1].T.dot(grad_z2)      
        mask = z1>0
        grad_z1 = grad_h1*mask
    
        grad_weights.append(grad_z1[:, None].dot(x[:, None].T))
        grad_biases.append(grad_z1)
        
        grad_weights.reverse()
        grad_biases.reverse()  

        return grad_weights, grad_biases
    
    def update_parameters(self,grad_weights,grad_biases,learning_rate):
      
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate*grad_weights[i]
            self.bias[i] -= learning_rate*grad_biases[i]

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
    parser.add_argument('-learning_rate', type=float, default=0.001,
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
