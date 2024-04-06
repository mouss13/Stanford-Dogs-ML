import numpy as np
import sys

class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """

        N, D = training_data.shape # N = 2938, D = 2048
        print("N, D:", N, D)
        #Formule du cours
        A = training_data.T @ training_data + self.lmda * np.eye(D)
        print("A shape:", A.shape)
        B = training_data.T @ training_labels
        print("B shape:", B.shape)
        self.weights = np.linalg.solve(A, B) # Resoulution pour w optimal
        return training_data @ self.weights
    
    def predict(self, test_data): 
        return test_data @ self.weights

    