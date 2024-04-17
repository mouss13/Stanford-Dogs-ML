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

def plot_results(predictions, actuals, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5, label='Predicted vs Actual')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_hyperparameter_tuning(hyperparam_values, train_metrics, val_metrics, title, xlabel, ylabel, labels):
    plt.figure(figsize=(12, 6))
    
    for metric, label in zip([train_metrics, val_metrics], labels):
        plt.plot(hyperparam_values, metric, label=label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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
        xtrain, xtest, ytrain, ytest, ctrain, ctest = feature_data['xtrain'],feature_data['xtest'],\
                                                      feature_data['ytrain'],feature_data['ytest'],\
                                                      feature_data['ctrain'],feature_data['ctest']

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
        validation_size = int(0.2 * len(xtrain)) # 20% of the training data
        indices = np.random.permutation(len(xtrain))
        train_indices, val_indices = indices[validation_size:], indices[:validation_size]
        xval, yval, cval = xtrain[val_indices],\
                           ytrain[val_indices],\
                           ctrain[val_indices]
        xtrain, ytrain, ctrain = xtrain[train_indices],\
                                 ytrain[train_indices],\
                                 ctrain[train_indices]
    
    means, stds = np.mean(xtrain, axis=0), np.std(xtrain, axis=0)
    xtrain, xtest = normalize_fn(xtrain, means, stds), normalize_fn(xtest, means, stds)
    xtrain, xtest = append_bias_term(xtrain), append_bias_term(xtest)
    
    if not args.test:
        xval = normalize_fn(xval, means, stds)
        xval = append_bias_term(xval)

    ## 3. Initialize the method you want to use.

    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.") # for MS2
    
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)
        if args.graph:
            #lr_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5]  # Example range for learning rates
            lr_values = np.linspace(1e-5, 0.2, 10)
            val_acc = []
            val_f1 = []
            
            for lr in lr_values:
                method_obj = LogisticRegression(lr=lr, max_iters=args.max_iters)
                method_obj.fit(xtrain, ytrain)
                val_pred = method_obj.predict(xval)
                val_acc.append(accuracy_fn(val_pred, yval))
                val_f1.append(macrof1_fn(val_pred, yval))
            plot_hyperparameter_tuning(lr_values, val_acc, val_f1,
                                    "Logistic Regression Performance Over Learning Rate", 
                                    "Learning Rate", "Accuracy/F1", 
                                    ["Validation Accuracy",
                                        "Validation F1"])
                                    


    elif args.method == "linear_regression":
        method_obj = LinearRegression(lmda=args.lmda)
        
        if args.graph:
            #lambda_values = [1, 2, 3, 5, 8, 13, 21, 34]
            #lambda_values = np.logspace(0, 2, base=100, num=10)
            lambda_values = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597] # fibonacci (j'ai essayé d'autres séquences et c'est moins beau lol)
            train_mse = []
            val_mse = []
            
            # Split your dataset into training and validation here, if not already split

            for lmda in lambda_values:
                method_obj = LinearRegression(lmda=lmda)
                # Fit the model on training data
                method_obj.fit(xtrain, ytrain)
                # Predict on training and validation data
                train_pred = method_obj.predict(xtrain)
                val_pred = method_obj.predict(xval)
                
                # Calculate and append the MSE for train and validation sets
                train_mse.append(mse_fn(train_pred, ytrain))
                val_mse.append(mse_fn(val_pred, yval))
            
            # Plot the MSE against lambda values for both training and validation sets
            plot_hyperparameter_tuning(lambda_values, train_mse, val_mse, "Linear Regression Performance Over Lambda",
                                    "Lambda", "MSE", ["Train MSE", "Validation MSE"])
    
    elif args.method == "knn":
        method_obj = KNN(k=args.K)
        #k_values = np.logspace(0, 2, base=10, num=10)
        k_values = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        val_acc = []
        val_f1 = []

        if args.graph:
            for k in k_values:
                method_obj = KNN(k=k)
                method_obj.fit(xtrain, ytrain)
                val_pred = method_obj.predict(xval)

                val_acc.append(mse_fn(val_pred, yval))
                val_f1.append(macrof1_fn(val_pred, yval))

            plot_hyperparameter_tuning(k_values, val_acc, val_f1, "KNN Validation Scores over K", "K", "Score", ["Test error", "Validation F1"])
    
    else:
        raise ValueError("Invalid method!")

    ## 4. Train and evaluate the method

    if args.task == "center_locating":
        # Fit parameters on training data
        preds_train = method_obj.fit(xtrain, ctrain)

        # Perform inference for training and test data
        train_pred = method_obj.predict(xtrain)
        preds = method_obj.predict(xtest)

        ## Report results: performance on train and valid/test sets
        train_loss = mse_fn(train_pred, ctrain)
        loss = mse_fn(preds, ctest)
        print(f"\nTrain loss = {train_loss:.3f}% - Test loss = {loss:.3f}")

        if not args.test:
            val_pred = method_obj.predict(xval)
            val_loss = mse_fn(val_pred, cval)
            print(f"Validation loss = {val_loss:.3f}")
        
        #plot_results(preds, ctest, "Linear Regression Predictions vs Actual", "Actual Values", "Predicted Values")

    
    elif args.task == "breed_identifying":
        
        # Fit (:=train) the method on the training data for classification task
        preds_train = method_obj.fit(xtrain, ytrain)

        # Predict on unseen data
        preds = method_obj.predict(xtest)

        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        
        if not args.test:
            val_pred = method_obj.predict(xval)
            val_acc = accuracy_fn(val_pred, yval)
            print(f"Validation set accuracy = {val_acc:.3f}%")
        
        #plot_results(preds, ytest, "Logistic Regression Predictions vs Actual", "Actual Labels", "Predicted Labels")
    else:
        raise Exception("Invalid choice of task! Only support center_locating and breed_identifying!")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=int, default=0, help="Toggle to visualize performance metrics over hyperparameters")
    parser.add_argument('--task', default="center_locating", type=str, help="center_locating / breed_identifying")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
    parser.add_argument('--data_path', default="data", type=str, help="path to your dataset")
    parser.add_argument('--data_type', default="features", type=str, help="features/original(MS2)")
    parser.add_argument('--lmda', type=float, default=10, help="lambda of linear/ridge regression")
    parser.add_argument('--K', type=int, default=1, help="number of neighboring datapoints used for knn")
    parser.add_argument('--lr', type=float, default=0.0671, help="learning rate for methods with learning rate")
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


