import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def undersample_classes(X_train, y_train):
    # Instantiate RandomUnderSampler
    under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)

    # Fit and transform the training data using the under sampler
    X_resampled, y_resampled = under_sampler.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


def oversample_classes(train_data, train_labels, strategy='SMOTE', ratio=0.5):
    """
    Performs oversampling to help the classifier learn more about the minority class.
    Either the SMOTE or random oversampling techniques can be used.

    NOTE: Oversampling MUST be used on the training dataset AFTER splitting the data.
    If not, the results will be overly optimistic.

    ratio: the goal proportion of the minority class compared to the majority class.
    """

    print("Oversampling:")
    # print(train_labels.value_counts())

    if strategy == "SMOTE":
        oversampler = SMOTE(sampling_strategy=ratio)
    elif strategy == "random":
        oversampler = RandomOverSampler(sampling_strategy=ratio)
    else:
        print("{} is an invalid oversampling strategy. No oversampling will be performed.")
        return train_data, train_labels

    over_data, over_labels = oversampler.fit_resample(train_data, train_labels)

    # print("Label value counts after oversampling:")
    # print(over_labels.value_counts())
    # print()

    return over_data, over_labels