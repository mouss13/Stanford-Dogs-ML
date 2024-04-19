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

def k_fold_logReg(X, y, num_folds, lr_values, iter_values):
    fold_size = len(X) // num_folds
    best_acc = -1
    best_f1 = -1
    best_lr = None
    results = []
    
    for lr in lr_values:
        for iters in iter_values:
            acc_scores = []
            f1_scores = []
            for fold in range(num_folds):
                start, end = fold * fold_size, (fold + 1) * fold_size
                X_train = np.concatenate((X[:start], X[end:]))
                y_train = np.concatenate((y[:start], y[end:]))
                X_val = X[start:end]
                y_val = y[start:end]

                model = LogisticRegression(lr=lr, max_iters=iters)
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                acc = accuracy_fn(predictions, y_val)
                f1 = macrof1_fn(predictions, y_val)

                acc_scores.append(acc)
                f1_scores.append(f1)

            avg_acc = np.mean(acc_scores)
            avg_f1 = np.mean(f1_scores)
            results.append((lr, iters, avg_acc, avg_f1))

            if avg_acc > best_acc:
                best_acc = avg_acc
                best_lr = lr
                best_iters_acc = iters

            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_iters_f1 = iters

    return results, best_lr, best_iters_acc, best_f1, best_iters_f1


def cross_val_score(X, y, num_folds, k):
    """
    Performs k-fold cross validation on the given data and labels.

    Arguments:

        X (np.array): training data of shape (N,D)
        y (np.array): regression target of shape (N,)
        num_folds (int): number of folds for cross validation
        k (int): number of neighbors for KNN

    Returns:

        val_accuracies (float): average validation accuracy over all folds
    """
    
    fold_size = len(X) // num_folds
    val_accuracies = []
    f1_scores = []
    val_mses = []
    train_mses = []
    
    for fold in range(num_folds):
        start, end = fold * fold_size, (fold + 1) * fold_size
        X_val_fold = X[start:end]
        y_val_fold = y[start:end]
        X_train_fold = np.concatenate((X[:start], X[end:]))
        y_train_fold = np.concatenate((y[:start], y[end:]))

        model = KNN(k=k)
        model.fit(X_train_fold, y_train_fold)
        predictions = model.predict(X_val_fold)

        acc = accuracy_fn(predictions, y_val_fold)
        f1 = macrof1_fn(predictions, y_val_fold)
        mse = mse_fn(predictions, y_val_fold)
        train_mse = mse_fn(model.predict(X_train_fold), y_train_fold)

        val_accuracies.append(acc)
        f1_scores.append(f1)
        val_mses.append(mse)
        train_mses.append(train_mse)

    return np.mean(val_accuracies), np.mean(f1_scores), np.mean(val_mses), np.mean(train_mses)


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
        feature_data = np.load('features.npz',allow_pickle=True)
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
            
            lr_values = np.logspace(-6, 0, 20)  # Learning rate values
            iter_values = [10, 50, 100, 500, 1000]  # Max iteration values

            print("\nPerforming K-Fold Cross Validation for Logistic Regression with {} values of learning rate and {} values of max iterations...".format(len(lr_values), len(iter_values)))
            results, best_lr, best_iters_acc, best_f1, best_iters_f1 = k_fold_logReg(xtrain, ytrain, args.folds, lr_values, iter_values)

            # Extract results for plotting
            acc_by_lr = [result[2] for result in results if result[1] == best_iters_acc]
            acc_by_iter = [result[2] for result in results if result[0] == best_lr]

            print("\n========== Logistic Regression Results =========\n")
            print(f"Optimal learning rate for accuracy: {best_lr} with max_iters: {best_iters_acc}")
            print(f"Optimal learning rate for F1: {best_lr} with max_iters: {best_iters_f1}")
            print("\n========================================================\n")

            # Plotting for learning rates
            plt.figure(figsize=(12, 6))
            plt.semilogx(lr_values, acc_by_lr, label='Validation Accuracy')
            plt.title("Logistic Regression Performance Over Learning Rate")
            plt.xlabel("Learning Rate")
            plt.ylabel("Performance Metrics")
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plotting for iterations
            plt.figure(figsize=(12, 6))
            plt.plot(iter_values, acc_by_iter, label='Validation Accuracy')
            plt.title("Logistic Regression Performance Over Max Iterations")
            plt.xlabel("Max Iterations")
            plt.ylabel("Performance Metrics")
            plt.legend()
            plt.grid(True)
            plt.show()
            
            
    elif args.method == "linear_regression":
        method_obj = LinearRegression(lmda=args.lmda)
    
        if args.graph:
            
            lambda_values = np.logspace(-3, 3, 50)
            train_mse = []
            val_mse = []
            
            for lmda in lambda_values:
                method_obj = LinearRegression(lmda=lmda)
                method_obj.fit(xtrain, ytrain)
                train_pred = method_obj.predict(xtrain)
                val_pred = method_obj.predict(xval)
                train_mse.append(mse_fn(train_pred, ytrain))
                val_mse.append(mse_fn(val_pred, yval))
            
            plt.figure(figsize=(12, 6))
            plt.semilogx(lambda_values, train_mse, label='Training MSE')
            plt.semilogx(lambda_values, val_mse, label='Validation MSE')
            plt.title("Linear Regression Performance Over Lambda")
            plt.xlabel("Lambda (Regularization parameter)")
            plt.ylabel("Mean Squared Error")
            plt.legend()
            plt.grid(False)
            plt.show()


    elif args.method == "knn":

        method_obj = KNN(k=args.K, task_kind=args.task)

        if args.graph:
            k_values = list(range(1, 101)) # adjust to 21 if computation is too long 
            acc_results, f1_results, val_mse_results, train_mse_results = [], [], [], []
            best_k_acc = best_k_f1 = best_k_mse = 1
            best_score_acc = best_score_f1 = best_score_mse = float('-inf')

            print("\nPlotting KNN performance over K for {} values of K and {}-folds...".format(len(k_values), args.folds))
            print("/!\\ Disclaimer: This may take 2-3 minutes for 100 values of K and 5-fold\n")

            for k in k_values:
                method_obj = KNN(k=k)
                acc, f1, val_mse, train_mse = cross_val_score(xtrain, ytrain, args.folds, k)
                acc_results.append(acc)
                f1_results.append(f1)
                val_mse_results.append(val_mse)
                train_mse_results.append(train_mse)

                if acc > best_score_acc:
                    best_score_acc = acc
                    best_k_acc = k

                if f1 > best_score_f1:
                    best_score_f1 = f1
                    best_k_f1 = k

                if val_mse < best_score_mse or best_score_mse == float('-inf'):
                    best_score_mse = val_mse
                    best_k_mse = k
            
            #printing the results for optimal K for different metrics
            print("\n========== K-Fold Cross Validation Results =========\n")
            print("\nBest k for accuracy ({}-Fold CV): {}\nBest CV accuracy: {:.5f}%\n".format(args.folds, best_k_acc, best_score_acc))
            print("Best k for F1 ({}-Fold CV): {}\nBest CV F1: {:.3f}\n".format(args.folds, best_k_f1, best_score_f1))
            print("Best k for MSE ({}-Fold CV): {}\nBest CV MSE: {:.3f}\n".format(args.folds, best_k_mse, best_score_mse))
            print("\n========================================================\n")
            
            #plot Accuracy
            plt.figure(figsize=(12, 6))
            plt.plot(k_values, acc_results, label='Validation Accuracy')
            plt.title("KNN Performance Variation with K")
            plt.xlabel("Number of Neighbors (K)")
            plt.ylabel("Performance Metrics")
            plt.legend()
            plt.grid(True)
            plt.show()

            #plot MSE for validation set and training set
            plt.figure(figsize=(12, 6))
            plt.plot(k_values, val_mse_results, label='Validation MSE')
            plt.plot(k_values, train_mse_results, label='Training MSE')
            plt.title("KNN Performance Variation with K")
            plt.xlabel("Number of Neighbors (K)")
            plt.ylabel("Mean Squared Error")
            plt.legend()
            plt.grid(True)
            plt.show()
        
    else:
        raise ValueError("Invalid model! Only support linear_regression, logistic_regression, knn")

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
        print(f"\nTrain loss = {train_loss:.4f}% - Test loss = {loss:.4f}")

        if not args.graph and not args.test:
            val_pred = method_obj.predict(xval)
            val_loss = mse_fn(val_pred, cval)
            print(f"Validation loss = {val_loss:.4f}")
        
    
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

    else:
        raise Exception("Invalid choice of task! Only support center_locating and breed_identifying!")


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5, help="Number of folds for cross-validation") # Added argument for K-fold cross-validation
    parser.add_argument('--graph', type=int, default=0, help="Toggle to visualize performance metrics over hyperparameters") # Added argument for plotting and finding optimal hyperparameters
    parser.add_argument('--task', default="center_locating", type=str, help="center_locating / breed_identifying")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
    parser.add_argument('--data_path', default="data", type=str, help="path to your dataset")
    parser.add_argument('--data_type', default="features", type=str, help="features/original(MS2)")
    parser.add_argument('--lmda', type=float, default=10, help="lambda of linear/ridge regression")
    parser.add_argument('--K', type=int, default=1, help="number of neighboring datapoints used for knn")
    parser.add_argument('--lr', type=float, default=0.00297635144, help="learning rate for methods with learning rate")
    # better accuracy lr : 0.0006951927961775605, better max_iters = 1000
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


