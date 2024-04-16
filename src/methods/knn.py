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
        return training_labels

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
            distances = [np.linalg.norm(test_point - point) for point in self.training_data]
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]

            if self.task_kind == "breed_identifying":
                # For classification, vote the most frequent label
                label_count = {}
                for idx in k_indices:
                    label = self.training_labels[idx]
                    if isinstance(label, np.ndarray) and label.size == 1:
                        label = label[0]
                    elif isinstance(label, np.ndarray):
                        raise ValueError("Labels must be scalar or 1D array with one element for classification")
                    label_count[label] = label_count.get(label, 0) + 1
                most_frequent_label = max(label_count, key=label_count.get)
                test_labels.append(most_frequent_label)
            else:
                # For regression, calculate the mean of the neighbors' values
                neighbor_values = [self.training_labels[idx] for idx in k_indices]
                neighbor_values_array = np.array(neighbor_values)
                # Calculate the mean across the rows (axis=0), not across the columns
                average_value = np.mean(neighbor_values_array, axis=0)
                test_labels.append(average_value)

        return np.array(test_labels)