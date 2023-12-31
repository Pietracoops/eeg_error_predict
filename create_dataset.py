import mne
import numpy as np
from pathlib import Path
from flanker import Participant, FlankerData, load_flanker_data_from_pickle
import matplotlib.pyplot as plt
import os
from tkinter import filedialog
from tkinter import *
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

from preprocessing import oversample_classes, undersample_classes

from utils import check_class_distribution
from sklearn.metrics import (f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay,
                            precision_recall_curve, PrecisionRecallDisplay, roc_curve, RocCurveDisplay, roc_auc_score, auc)


from MachineLearning import MLAnalysis

def find_files_by_name(directory, file_name):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == file_name:
                file_list.append(os.path.join(root, file))
    return file_list

def process_files(filename):
    flanker_data = FlankerData()

    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()

    files = find_files_by_name(folder_selected, "Flanker.set")

    print(files)

    id = 1
    for file in files:
        print(f"Processing participant {id}")
        flanker_data.add_participant_path(file, id)
        id += 1

    flanker_data.save_to_pickle(os.getcwd() + "\\" + filename)
    print("Complete processing")

def process_brainvision_files():
    flanker_data = FlankerData()

    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()

    files = find_files_by_name(folder_selected, "Flanker.set")

    print(files)

    id = 1
    for file in files:
        print(f"Processing participant {id}")
        flanker_data.add_participant_path(file, id)
        id += 1

    flanker_data.save_to_pickle(os.getcwd() + "\\flanker_data_2.pkl")
    print("Complete processing")


def machine_learning_svm(filename):
    data_path = os.getcwd() + "\\" + filename
    parameter_path = os.getcwd() + "\\parameters.yaml"

    ml_obj = MLAnalysis(parameter_path)
    ml_obj.prepare_data(data_path)

    # Create SVM
    # svm_params = {
    #     'C': 1.0,          # Regularization parameter
    #     'kernel': 'rbf',   # Kernel type (e.g., 'linear', 'rbf', 'poly')
    #     'gamma': 'scale',  # Kernel coefficient ('scale' uses 1 / (n_features * X.var()))
    #     'probability': True,  # Enable probability estimates
    #     'shrinking': True,    # Use the shrinking heuristic
    #     'tol': 0.001          # Tolerance for stopping criterion
    # }
    
    svm_params = {
        'C': 1000.0,  # Regularization parameter
        'class_weight': None,
        'kernel': 'poly',  # Kernel type (e.g., 'linear', 'rbf', 'poly')
        'gamma': 'scale',  # Kernel coefficient ('scale' uses 1 / (n_features * X.var()))
        'probability': True,  # Enable probability estimates
        'shrinking': True,  # Use the shrinking heuristic
        'tol': 0.001  # Tolerance for stopping criterion
    }
    
    svm = SVC(**svm_params)

    ml_obj.launch_model(svm, 'svm_1', svm_params)


def machine_learning_model_search(filename):
    data_path = os.getcwd() + "\\" + filename
    parameter_path = os.getcwd() + "\\parameters.yaml"

    ml_obj = MLAnalysis(parameter_path)
    ml_obj.prepare_data(data_path)

    # ml_obj.search_model_domain()

def machine_learning_rf(filename):
    data_path = os.getcwd() + "\\" + filename
    parameter_path = os.getcwd() + "\\parameters.yaml"

    ml_obj = MLAnalysis(parameter_path)
    ml_obj.prepare_data(data_path)

    # ml_obj.search_model_domain()

    random_forest = RandomForestClassifier()
    rf_params = {
        "bootstrap": True, "ccp_alpha": 0.0, "class_weight": None, "criterion": "gini", "max_depth": 50,
         "max_features": 0.7, "max_leaf_nodes": None, "max_samples": None, "min_impurity_decrease": 0.0,
         "min_samples_leaf": 1, "min_samples_split": 10, "min_weight_fraction_leaf": 0.0, "n_estimators": 5,
         "n_jobs": None,
         "oob_score": False, "random_state": None, "verbose": 0, "warm_start": False
    }
    ml_obj.launch_model(random_forest, 'rf_1', rf_params)

def machine_learning_nn(filename):
    print("Running machine learning for Neural Network...")
    data_path = os.getcwd() + "\\" + filename
    parameter_path = os.getcwd() + "\\parameters.yaml"

    ml_obj = MLAnalysis(parameter_path)
    ml_obj.prepare_data_nn(data_path)

    ml_obj.run_nn_model()
    print("Done")

now = datetime.now()
now_string = now.strftime("%d_%m_%y_%H_%M_%S")
# process_files(f"flanker_data_7_{now_string}.pkl")
machine_learning_nn("flanker_data_7_24_12_23_17_35_04.pkl")







