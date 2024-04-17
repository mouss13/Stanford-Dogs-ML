import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.weights = None

        

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        ##

        nb_features = training_data.shape[1]
        nb_classes = get_n_classes(training_labels)
        self.weights = np.random.normal(0,0.01,(nb_features, nb_classes))

   
        y_onehot = label_to_onehot(training_labels, nb_classes)
        
        for _ in range(self.max_iters):
            #softmax
            a = np.dot(training_data, self.weights)
            probs = np.exp(a) / np.sum(np.exp(a), axis=1, keepdims=True)
            gradient = np.dot(training_data.T, (probs - y_onehot))

            self.weights -= self.lr * gradient
        pred_labels = self.predict(training_data)
        # Return predicted labels
       
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ##
        if self.weights is None:
            raise ValueError("Model not trained yet")
        a = np.dot(test_data, self.weights)
        probs = np.exp(a) / np.sum(np.exp(a), axis=1, keepdims=True)
        pred_labels = np.argmax(probs, axis=1)
        ##
        return pred_labels