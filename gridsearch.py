import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np

# K-Nearest Neighbors (KNN):
#
# Simple and effective for classification tasks.
# Can capture local patterns in the data.
# Gradient Boosting (e.g., XGBoost, LightGBM):
#
# Ensemble methods that combine weak learners to form a strong classifier.
# Can handle complex relationships in the data.
# Convolutional Neural Networks (CNN):
#
# Especially effective for analyzing spatial patterns in EEG data.
# Can automatically learn hierarchical features.
# Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM):
#
# Suitable for capturing temporal dependencies in time-series EEG data.
# Effective for tasks involving sequential information.
# Gaussian Naive Bayes:
#
# Assumes features are normally distributed.
# Can be computationally efficient and work well for certain types of data.
# Linear Discriminant Analysis (LDA):
#
# Assumes normally distributed classes and equal covariance matrices.
# Useful when the assumption holds, and it can perform well.
# Logistic Regression:
#
# Simple and interpretable.
# Works well when the relationship between features and classes is approximately linear.
# Support Vector Machines (SVM) with Other Kernels:
#
# In addition to the linear kernel, SVMs can be used with radial basis function (RBF), polynomial, and sigmoid kernels.
# The choice of the kernel depends on the characteristics of your data.
# Ensemble Methods (Voting Classifier, Bagging, AdaBoost):
#
# Combine multiple classifiers to improve overall performance.
# Can be beneficial when there is diversity among the base classifiers.
# EEGNet:
#
# A deep learning architecture specifically designed for EEG data.
# Takes into account the spatial and temporal aspects of EEG signals.



# param_grid_rf = {
#     'n_estimators': [5, 10, 20, 50, 100, 150, 200],
#     'max_depth': [None, 5, 10, 20, 50, 100, 200],
#     'max_features': [None, 'sqrt', 'log2', ] + list(np.arange(0.5, 1, 0.1)),
#     'min_samples_split': [2, 5, 10, 15, 20]
# }
param_grid_rf = {
    'n_estimators': [5, 50, 100],
    'max_depth': [None, 5, 50, 100],
    'max_features': [None, 'sqrt', 'log2', ] + list(np.arange(0.7, 1, 0.1)),
    'min_samples_split': [2, 10, 20]
}


# param_grid_svm = {
#     'C': [0.1, 1, 10, 100, 1000], # Controls the trade-off between achieving a low training error and a low testing error. Larger values of C lead to a smaller-margin hyperplane but better training accuracy.
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], # Specifies the kernel type to be used in the algorithm. Common choices include 'linear', 'poly', 'rbf' (radial basis function), and 'sigmoid'.
#     'gamma': ['scale', 'auto', 0.1, 1, 10], # Controls the influence of input samples. The 'auto' option uses 1/n_features.
#     'class_weight': [None, 'balanced'], # Adjusts the penalty for different classes. Use 'balanced' to automatically adjust weights inversely proportional to class frequencies.
#     'shrinking': [True, False], # Whether to use the shrinking heuristic. It can be set to True or False.
#     'tol': [1e-3, 1e-4, 1e-5], # Tolerance for stopping criterion.
# }

param_grid_svm = {
    'C': [0.1, 1, 10, 100, 1000], # Controls the trade-off between achieving a low training error and a low testing error. Larger values of C lead to a smaller-margin hyperplane but better training accuracy.
    'kernel': ['sigmoid'], # Specifies the kernel type to be used in the algorithm. Common choices include 'linear', 'poly', 'rbf' (radial basis function), and 'sigmoid'.
    'gamma': ['scale', 'auto', 0.1, 1, 10], # Controls the influence of input samples. The 'auto' option uses 1/n_features.
    'class_weight': [None, 'balanced'], # Adjusts the penalty for different classes. Use 'balanced' to automatically adjust weights inversely proportional to class frequencies.
    'shrinking': [True, False], # Whether to use the shrinking heuristic. It can be set to True or False.
    'tol': [1e-3, 1e-4, 1e-5], # Tolerance for stopping criterion.
}

param_grid_svm_poly = {
    'C': [0.1, 1, 10, 100, 1000], # Controls the trade-off between achieving a low training error and a low testing error. Larger values of C lead to a smaller-margin hyperplane but better training accuracy.
    'kernel': ['poly'], # Specifies the kernel type to be used in the algorithm. Common choices include 'linear', 'poly', 'rbf' (radial basis function), and 'sigmoid'.
    'degree': [2, 3, 4], # Only relevant for the 'poly' kernel. It specifies the degree of the polynomial.
    'gamma': ['scale', 'auto', 0.1, 1, 10], # Controls the influence of input samples. The 'auto' option uses 1/n_features.
    'class_weight': [None, 'balanced'], # Adjusts the penalty for different classes. Use 'balanced' to automatically adjust weights inversely proportional to class frequencies.
    'shrinking': [True, False], # Whether to use the shrinking heuristic. It can be set to True or False.
    'tol': [1e-3, 1e-4, 1e-5], # Tolerance for stopping criterion.
}

param_dict = {'random_forest': param_grid_rf,
              'svm': param_grid_svm,
              'svm_poly': param_grid_svm_poly}



def gridsearch(estimator, data, labels, classifier_keys, scoring, num_folds=5):
    """
    Performs a grid search cross validation procedure using the estimator and data over the
    given parameter space. Can be used with any estimators from the sklearn library.

    Returns the most succesful model.
    """
    scores = []
    best_models = []
    best_model_names = []

    for classifier_key in classifier_keys:
        best_model = GridSearchCV(estimator, param_dict[classifier_key], scoring=scoring, cv=num_folds, n_jobs=-1, verbose=5)
        best_model.fit(data, labels)
        print(f"[{classifier_key}] The cross-validation {best_model.scorer_} of the chosen 'best model' is {best_model.best_score_}")
        best_models.append(best_model)
        scores.append(best_model.best_score_.copy())
        best_model_names.append(classifier_key)

    max_value = max(scores)
    max_index = scores.index(max_value)
    best_model = best_models[max_index]
    best_model_name = best_model_names[max_index]

    return best_model.best_estimator_, max_value, best_model_name


def randomgridsearch(estimator, data, labels, classifier_keys, r_seed, n_iter, scoring, num_folds=5):
    """
    Performs a grid search over a random subset of combinations from the parameter space.

    Returns the most successful model.
    """
    scores = []
    best_models = []
    best_model_names = []

    for classifier_key in classifier_keys:
        best_model = RandomizedSearchCV(estimator, param_dict[classifier_key], n_iter=n_iter,
                                        scoring=scoring, cv=num_folds, n_jobs=-1,
                                        random_state=r_seed, verbose=5)
        best_model.fit(data, labels)
        print(f"[{classifier_key}] The cross-validation {best_model.scorer_} of the chosen 'best model' is {best_model.best_score_}")
        best_models.append(best_model)
        scores.append(best_model.best_score_.copy())
        best_model_names.append(classifier_key)

    max_value = max(scores)
    max_index = scores.index(max_value)
    best_model = best_models[max_index]
    best_model_name = best_model_names[max_index]

    return best_model.best_estimator_, max_value, best_model_name