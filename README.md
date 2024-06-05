# Introduction to Machine Learning Course Project

## Project Overview
This project applies linear regression, logistic regression, and K-nearest neighbors (KNN) to the Stanford Dogs dataset. The tasks include locating the center of a dog in an image and identifying dog breeds using supervised machine learning techniques.

## Methodology

### Data Preparation
Preprocessed the Stanford Dogs dataset through shuffling, normalizing, and bias augmentation.

### Model Implementations

- **Linear Regression**: Implemented Ridge Regression with dynamic lambda adjustment.
- **Logistic Regression**: Used gradient descent with tuned learning rates and iterations.
- **K-Nearest Neighbors (KNN)**: Optimized k value for classification and regression tasks.

## Results

### Linear Regression
- Robust performance across a range of lambda values.

### Logistic Regression
- Optimal learning rate at \(1.438 \cdot 10^{-3}\) and 500 iterations.

### K-Nearest Neighbors (KNN)
- Optimal k values: \( k_r=10 \) for regression and \( k_c=13 \) for classification.

## Conclusion
Combining these models achieved a robust solution with minimal overfitting and good performance across all metrics.
