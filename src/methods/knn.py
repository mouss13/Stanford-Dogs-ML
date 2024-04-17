import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "center_locating"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind =task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        
        self.training_data = training_data
        self.training_labels = training_labels
        pred_labels = training_labels

        #just return training labels, because KNN does not train!
        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        test_labels = []
        for test_point in test_data:
            # Compute distances between test_point and all training points
            distances = np.linalg.norm(test_point - self.training_data, axis=1)  
            # indides of the k smallest distances  
            k_indices = np.argsort(distances)[:self.k]

            if self.task_kind == "breed_identifying":
                #voting for classification task
                labels, counts = np.unique(self.training_labels[k_indices], return_counts=True)
                test_labels.append(labels[np.argmax(counts)])
                
            else:
                #calculate mean for regression task
                mean_values = np.mean(self.training_labels[k_indices], axis=0)
                test_labels.append(mean_values)
                
        return np.array(test_labels)