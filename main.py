import argparse

import numpy as np
import matplotlib.pyplot as plt

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression 
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os
np.random.seed(100)

def plot_predictions_vs_actual(predictions, actual, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(actual[:, 0], predictions[:, 0], alpha=0.5, label='Dimension 1')
    plt.scatter(actual[:, 1], predictions[:, 1], alpha=0.5, label='Dimension 2')
    plt.title(title)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors

    ##EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load('../features.npz',allow_pickle=True)
        xtrain, xtest, ytrain, ytest, ctrain, ctest =feature_data['xtrain'],feature_data['xtest'],\
        feature_data['ytrain'],feature_data['ytest'],feature_data['ctrain'],feature_data['ctest']
    
    ##ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path,'dog-small-64')
        xtrain, xtest, ytrain, ytest, ctrain, ctest = load_data(data_dir)

    ##TODO: ctrain and ctest are for regression task. (To be used for Linear Regression and KNN)
    ##TODO: xtrain, xtest, ytrain, ytest are for classification task. (To be used for Logistic Regression and KNN)


    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        ### WRITE YOUR CODE HERE

        # Splitting the data into 80% training and 20% validation
        t_set_ratio = 0.8 # adjust training set ratio
        np.random.seed(0)
        rinds = np.random.permutation(xtrain.shape[0]) # randomize indices
        n_train = int(t_set_ratio * xtrain.shape[0]) # size of training set
        xval = xtrain[rinds[n_train:]] # validation set
        xtrain, ytrain = xtrain[rinds[:n_train]], ytrain[rinds[:n_train]] 

        cval = ctrain[rinds[n_train:]]
        ctrain = ctrain[rinds[:n_train]]

    means, stds = np.mean(xtrain, axis=0), np.std(xtrain, axis=0)
    xtrain = normalize_fn(xtrain, means, stds)
    xtest = normalize_fn(xtest, means, stds)
    xtrain, xtest = append_bias_term(xtrain), append_bias_term(xtest)
    
    if not args.test:
        xval = normalize_fn(xval, means, stds)
        xval = append_bias_term(xval)

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)
    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)
    elif args.method == "linear_regression":
        method_obj = LinearRegression(lmda=args.lmda)
    elif args.method == "knn":
        method_obj = KNN(k=args.K)
    else:
        raise ValueError(f"Unsupported method: {args.method}")


    ## 4. Train and evaluate the method

    if args.task == "center_locating":
        # Fit parameters on training data
        print("============ Center Locating Task ============")
        print("ctrain shape:", ctrain.shape)
        preds_train = method_obj.fit(xtrain, ctrain)
        print("preds_train shape:", preds_train.shape)
        
        # Perform inference for training and test data
        training_preds = method_obj.predict(xtrain)
        test_preds = method_obj.predict(xtest)

        print("\ntraining shapes: ", training_preds.shape, ctrain.shape)
        print ("test shapes : ", test_preds.shape, ctest.shape)
        print("\n")

        ## Report results: performance on train and valid/test sets
        training_loss = mse_fn(training_preds, ctrain)
        loss = mse_fn(test_preds, ctest)

        print(f"\nTrain loss = {training_loss:.3f}% || Test loss = {loss:.3f}%")

        plot_predictions_vs_actual(training_preds, ctrain, "Training Predictions vs Actual")
        plot_predictions_vs_actual(test_preds, ctest, "Test Predictions vs Actual")

    elif args.task == "breed_identifying":

        # Fit (:=train) the method on the training data for classification task
        preds_train = method_obj.fit(xtrain, ytrain)

        # Predict on unseen data
        test_preds = method_obj.predict(xtest)

        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(test_preds, ytest)
        macrof1 = macrof1_fn(test_preds, ytest)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    else:
        raise Exception("Invalid choice of task! Only support center_locating and breed_identifying!")
    
    plt.figure(figsize=(10, 5))
    plt.bar(['Train Accuracy', 'Train Macro F1', 'Test Accuracy', 'Test Macro F1'], [acc, macrof1, acc, macrof1])
    plt.title("Performance Metrics for Breed Identifying Task")
    plt.ylabel("Score")
    plt.show()

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="center_locating", type=str, help="center_locating / breed_identifying")
    parser.add_argument('--method', default="linear_regression", type=str, help="dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
    parser.add_argument('--data_path', default="data", type=str, help="path to your dataset")
    parser.add_argument('--data_type', default="features", type=str, help="features/original(MS2)")
    parser.add_argument('--lmda', type=float, default=10, help="lambda of linear/ridge regression")
    parser.add_argument('--K', type=int, default=1, help="number of neighboring datapoints used for knn")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--nn_type', default="cnn", help="which network to use, can be 'Transformer' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)


