import numpy as np

# Load the dataset
data = np.load('octmnist.npz')

# Check for class labels or relevant attributes
class_labels = data['class_labels']
num_classes = len(np.unique(class_labels))

print("Number of classes:", num_classes)