from flanker import Participant, FlankerData, load_flanker_data_from_pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from utils import check_class_distribution

from preprocessing import oversample_classes, undersample_classes
import yaml
from gridsearch import gridsearch, randomgridsearch
import pprint
from datetime import datetime
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import make_scorer

from utils import save_params, load_params, check_pos_int_arg, check_pos_float_arg, save_plot, save_model_stats
from plotting_metrics import (get_f1_score, plot_confusion_matrix_display,
    plot_roc_curve, get_accuracy, get_roc_score, get_precision_recall, get_feature_importance)
import numpy as np
import random


# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# Define the neural network model
class EEGClassifier(nn.Module):

    def __init__(self):
        super(EEGClassifier, self).__init__()
        # Check if a GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            print("Using CUDA")
        else:
            print("Using CPU")
        
        self.params = self.__load_parameters()

        self.data_sample_size = 152
        self.K = self.params['K'] # Size of output layer - Play with this value
        self.in_channels = 62
        self.out_channels = 1
        self.kernel_size_1 = 3
        self.kernel_size_2, self.kernel_stride_2 = self.compute_kernel_stride(self.data_sample_size, self.K)
        self.fc_hidden_size = 128
        self.linear_input = (152 - self.kernel_size_2)//self.kernel_stride_2 + 1

        self.to(self.device) # move model to device

        
        self.model_name = self.params['model_name']
        self.batch_size = self.params['batch_size']
        self.epochs = self.params['epochs']
        self.learning_rate = self.params['learning_rate']

        self.conv1d_1 = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size_1, stride=1, padding=1).to(self.device)
        self.conv1d_2 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size_2, stride=self.kernel_stride_2).to(self.device)
        self.fc1 = nn.Linear(self.linear_input, self.fc_hidden_size).to(self.device)
        self.fc2 = nn.Linear(self.fc_hidden_size, 1).to(self.device)

        # Print the values
        print("============ Neural Network Parameters ===============")
        print(f"Kernel Size 1: {self.kernel_size_1}")
        print(f"Kernel Size 2: {self.kernel_size_2}")
        print(f"Kernel Stride 2: {self.kernel_stride_2}")
        print(f"K value: {self.K}")
        print(f"Fully Connected hidden layer size: {self.fc_hidden_size}")
        print("%%%%%%%%%%%%%%% Tuning %%%%%%%%%%%%%%%%%%")
        print(f"Learning Rate: {self.params['learning_rate']}")
        print(f"Batch Size: {self.params['batch_size']}")
        print(f"Epochs: {self.params['epochs']}")
        print("============ Neural Network Parameters ===============")

    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # binary classification (1 output node)
        x = x.squeeze()
        return x
    
    def __load_parameters(self):
        self.parameter_filepath = os.getcwd() + "\\parameters.yaml"
        with open(self.parameter_filepath, 'r') as file:
            params = yaml.safe_load(file)
        return params

    def compute_kernel_stride(self, input_size, output_size):
        # Trying different kernel sizes and strides
        for kernel_size in range(1, input_size + 1):
            for stride in range(1, input_size + 1):
                computed_output_size = (
                    input_size - kernel_size + 1 + 2 * 0
                ) // stride + 1  # Assuming padding is 0

                if (
                    computed_output_size == output_size
                    and (input_size - kernel_size) % stride == 0
                ):
                    return int(kernel_size), int(stride)

        # If no match is found, return default values
        return 3, 1  # You can adjust these default values based on your specific needs

    def load_and_split_data(self, eeg_data, labels, split_ratio=0.8):
        # Split the data into training and testing sets
        eeg_data = torch.Tensor(eeg_data).to(self.device)
        labels = torch.Tensor(labels).to(self.device)

        dataset = TensorDataset(eeg_data, labels)
        train_size = int(split_ratio * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

    def load_data(self, X_train, y_train, X_test, y_test):
        # Convert NumPy arrays to PyTorch tensors
        X_train_tensor = torch.Tensor(X_train)
        y_train_tensor = torch.Tensor(y_train)
        X_test_tensor = torch.Tensor(X_test)
        y_test_tensor = torch.Tensor(y_test)

        # Create PyTorch datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Create PyTorch data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def train_model(self):
        # Define the loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Train the model
        for epoch in range(self.epochs):
            running_loss = 0.0
            all_predictions = []
            all_labels = []
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Convert outputs to binary predictions
                predictions = torch.round(outputs)
                all_predictions.extend(predictions.cpu().detach().numpy())
                all_labels.extend(labels.cpu().detach().numpy())

            
            # Calculate F1 score
            f1 = get_f1_score(all_labels, all_predictions)
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(self.train_loader)}, F1 Score: {f1}")

    # Evaluate the model on the test set
    def evaluate_model(self):
        # Set the model to evaluation mode
        self.eval()

        # Initialize variables to keep track of predictions and labels
        all_predictions = []
        all_probabilities = []
        all_labels = []

        threshold = 0.5

        # Iterate over the test loader
        for inputs, labels in self.test_loader:
            # Forward pass through the model
            outputs = self(inputs)

            # Convert the outputs to probabilities
            probabilities = torch.sigmoid(outputs)

            # Convert the outputs to predictions by taking the argmax
            binary_predictions = (outputs > threshold).int()

            # Append the predictions and labels to the respective lists
            all_predictions.extend(binary_predictions.tolist())
            all_probabilities.extend(probabilities.tolist())
            all_labels.extend(labels.tolist())
        
        f1 = get_f1_score(all_labels, all_predictions)
        print(f"[{self.model_name}]: F1 Macro Score: {f1}")

        acc = get_accuracy(all_labels, all_predictions)
        print(f"[{self.model_name}]: Accuracy: {acc}")

        roc_auc = get_roc_score(all_labels, all_probabilities)
        print(f"[{self.model_name}]: Area under ROC curve: {roc_auc}")

        precision, recall = get_precision_recall(all_labels, all_predictions)
        print(f"[{self.model_name}]: Precision: {precision}")
        print(f"[{self.model_name}]: Recall: {recall}")

        conf_mat = plot_confusion_matrix_display(all_labels, all_predictions)
        roc_curve = plot_roc_curve(all_labels, all_probabilities)
        
        graphs_dict = {'conf_mat': conf_mat, 'roc_curve': roc_curve}
        results_dict = {'f1': f1,
                        'accuracy': acc,
                        'roc_auc': roc_auc,
                        'precision': precision,
                        'recall': recall}

        return results_dict, graphs_dict


class MLAnalysis:
    def __init__(self, paramater_filepath=None):
        self.r_seed = 10
        if paramater_filepath == None:
            self.parameter_filepath = os.getcwd() + "\\parameters.yaml"
        self.params = self.__load_parameters(paramater_filepath)
        self.__set_seed()

        # self.models = {'svm': SVC(probability=True),
        #           'random_forest': RandomForestClassifier()}

        self.models = {'random_forest': RandomForestClassifier()}
        # self.model_gs_keys = {'svm': ['svm', 'svm_poly'],
        #                  'random_forest': ['random_forest']}
        self.model_gs_keys = {'svm': ['svm'],
                         'random_forest': ['random_forest']}
        
        self.nn_model = EEGClassifier()


    def __load_parameters(self, parameter_path):
        with open(parameter_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    
    def __set_seed(self):
        random.seed(self.r_seed)
        np.random.seed(self.r_seed)
        torch.manual_seed(self.r_seed)

    def run_nn_model(self):
        print("Building data loaders...", end="")
        self.nn_model.load_and_split_data(self.X, self.y, 0.8)
        print("Done", end="\n")
        # self.nn_model.load_data(self.X_train, self.y_train, self.X_test, self.y_test)

        print("Starting model training...", end="")
        self.nn_model.train_model()
        print("Done", end="\n")

        print("Evaluating model...", end="")
        self.nn_model.evaluate_model()
        print("Done", end="\n")

    def prepare_data_nn(self, filepath):
        flankerdata = load_flanker_data_from_pickle(filepath)
        self.X, self.y = flankerdata.concatenate_data()
        print("Data prepared successfully for Neural Network")
        

    def prepare_data(self, filepath):
        flankerdata = load_flanker_data_from_pickle(filepath)

        if self.params['no_participant_leakage']:
            self.X_train, self.y_train, self.X_test, self.y_test = flankerdata.split_data()
        else:
            X, y = flankerdata.concatenate_data()
            X_reshaped = X.reshape((X.shape[0], -1))
            self.data = X
            self.labels = y
            check_class_distribution(y)
            # Split the data into training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_reshaped, y, test_size=0.2,
                                                                                    random_state=40, stratify=y)

        if self.params['undersampling'] == 1:
            print("Performing under sampling:")
            # Apply sampling strategy
            self.X_train, self.y_train = undersample_classes(self.X_train, self.y_train)
            check_class_distribution(self.y_train)
        elif self.params['oversampling'] == 1:
            if self.params['os_strategy'] == 'SMOTE':
                self.X_train, self.y_train = oversample_classes(self.X_train, self.y_train, strategy="SMOTE",
                                                                ratio=self.params['smote_ratio'])
            elif self.params['os_strategy'] == 'random':
                self.X_train, self.y_train = oversample_classes(self.X_train, self.y_train, strategy="random",
                                                                ratio=self.params['smote_ratio'])

    def grid_search(self, classifier, classifier_key):
        print(f"\nPerforming a random hyperparameter grid search with {self.params['gs_search_iter']}"
              f" iterations and {self.params['gs_num_folds']} folds per iteration. This may take a minute...\n")

        if self.params['gs_random'] == 1:
            estimator, max_value, best_model_name = randomgridsearch(classifier,
                                                    self.X_train,
                                                    self.y_train,
                                                    classifier_key,
                                                    self.r_seed,
                                                    self.params['gs_search_iter'],
                                                    'f1_macro',
                                                    num_folds=self.params['gs_num_folds'])
        else:
            estimator, max_value, best_model_name = gridsearch(classifier,
                                                    self.X_train,
                                                    self.y_train,
                                                    classifier_key,
                                                    'f1_macro',
                                                    num_folds=self.params['gs_num_folds'])

        print(f"\nModel Parameters for {estimator.__class__.__name__}:")
        print("-------------------------------------------------------")
        pprint.pprint(estimator.get_params())
        print("-------------------------------------------------------\n")

        if self.params['gs_save_params']:
            print(f"Saving parameters to params for {estimator.__class__.__name__} model")
            save_params(estimator.get_params(), f"{best_model_name}_{max_value}")

        return estimator, max_value, best_model_name

    def search_model_domain(self):
        best_models = []
        best_scores = []
        best_gs_keys = []
        for model_name, model in self.models.items():
            print(f"Searching {model_name}...")
            best_model, best_score, best_gs_key = self.grid_search(model, self.model_gs_keys[model_name])
            best_models.append(best_model)
            best_scores.append(best_score)
            best_gs_keys.append(best_gs_key)
            print(f"{model_name} - {best_gs_key}: Best Score: {best_score}")

        # Get the results on the held out 10% of data
        for i in range(len(best_models)):
            model = best_models[i]
            print(f"Starting Validation Set for {best_gs_keys[i]}")
            test_preds = model.predict(self.X_test)
            test_probs = model.predict_proba(self.X_test)

            f1 = get_f1_score(self.y_test, test_preds)
            print(f"[{best_gs_keys[i]}]: F1 Macro Score: {f1}")

            acc = get_accuracy(self.y_test, test_preds)
            print(f"[{best_gs_keys[i]}]: Accuracy: {acc}")

            roc_auc = get_roc_score(self.y_test, test_probs)
            print(f"[{best_gs_keys[i]}]: Area under ROC curve: {roc_auc}")

            precision, recall = get_precision_recall(self.y_test, test_preds)
            print(f"[{best_gs_keys[i]}]: Precision: {precision}")
            print(f"[{best_gs_keys[i]}]: Recall: {recall}")

            conf_mat = plot_confusion_matrix_display(self.y_test, test_preds)
            roc_curve = plot_roc_curve(self.y_test, test_probs)

            if self.params['save_plots']:
                # Get the date and time
                now = datetime.now()
                now_string = now.strftime("%d_%m_%y_%H_%M_%S")
                save_plot(conf_mat, f"{best_gs_keys[i]}_{f1}_conf_mat_{now_string}.png")
                save_plot(roc_curve, f"{best_gs_keys[i]}_{f1}_roc_curve_{now_string}.png")


    def launch_model(self, model, model_name, model_params):

        print("Fitting model...")
        model.fit(self.X_train, self.y_train)

        print("Done Fitting...")
        if self.params['cross_validation']:
            # Perform cross-validation
            cv_preds = cross_val_predict(model, self.X_train, self.y_train, cv=self.params['cv_count'])

            # Calculate metrics on the training set (cross-validation predictions)
            cv_f1 = get_f1_score(self.y_train, cv_preds)
            print(f"[{model_name}]: Training F1 Macro Score (CV): {cv_f1}")

            cv_acc = get_accuracy(self.y_train, cv_preds)
            print(f"[{model_name}]: Training Accuracy (CV): {cv_acc}")

        test_preds = model.predict(self.X_test)
        test_probs = model.predict_proba(self.X_test)

        f1 = get_f1_score(self.y_test, test_preds)
        print(f"[{model_name}]: F1 Macro Score: {f1}")

        acc = get_accuracy(self.y_test, test_preds)
        print(f"[{model_name}]: Accuracy: {acc}")

        roc_auc = get_roc_score(self.y_test, test_probs)
        print(f"[{model_name}]: Area under ROC curve: {roc_auc}")

        precision, recall = get_precision_recall(self.y_test, test_preds)
        print(f"[{model_name}]: Precision: {precision}")
        print(f"[{model_name}]: Recall: {recall}")

        conf_mat = plot_confusion_matrix_display(self.y_test, test_preds)
        roc_curve = plot_roc_curve(self.y_test, test_probs)

        results_dict = {'f1': f1,
                        'accuracy': acc,
                        'roc_auc': roc_auc,
                        'precision': precision,
                        'recall': recall}

        save_model_stats(f"{model_name}_{f1}", model_params, results_dict, conf_mat, roc_curve)





