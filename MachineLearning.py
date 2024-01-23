# Standard Libraries
import os
from datetime import datetime as dt
import random
import csv
import time
import logging

# Third-party Libraries
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.exceptions import UndefinedMetricWarning

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.init as init

# External Libraries
import yaml
import optuna
import warnings
import pprint

# Custom Modules
from flanker import Participant, FlankerData, load_flanker_data_from_pickle
from utils import (
    check_class_distribution, load_and_split_data,
    CustomDataset, RESULTS_PATH,
    save_plot, save_model_stats,
    save_params, seconds_to_hms, save_model, add_times,
    monitor_gpu_usage
)
from preprocessing import oversample_classes, undersample_classes
from gridsearch import gridsearch, randomgridsearch
from plotting_metrics import (
    get_f1_score, plot_confusion_matrix_display,
    plot_roc_curve, get_accuracy, get_roc_score,
    get_precision_recall, get_feature_importance, get_conf_indices,
    get_kappa_score
)
from transformer import *

# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# Suppress UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def load_parameters():
    parameter_filepath = os.getcwd() + "\\parameters.yaml"
    with open(parameter_filepath, 'r') as file:
        params = yaml.safe_load(file)
    return params

class Statistics():
    def __init__(self):
        self.epochs = []
        self.legend = ['Train Accuracy',
                       'Train F1', 
                       'Train Kappa',
                       'Train Precision', 
                       'Train Recall', 
                       'Training Loss', 
                       'Test Accuracy', 
                       'Test F1', 
                       'Test Kappa',
                       'Test Precision', 
                       'Test Recall', 
                       'Validation Loss']
    def update(self, 
               train_accuracy, 
               train_f1, 
               train_kappa,
               train_precision, 
               train_recall, 
               training_loss, 
               test_accuracy, 
               test_f1, 
               test_kappa,
               test_precision, 
               test_recall, 
               validation_loss):
        # This should be a dictionary containing name:value pairs
        self.epochs.append([train_accuracy, 
                            train_f1, 
                            train_kappa,
                            train_precision, 
                            train_recall, 
                            training_loss, 
                            test_accuracy, 
                            test_f1, 
                            test_kappa,
                            test_precision, 
                            test_recall, 
                            validation_loss])

    def save_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.legend)
            for i in range(len(self.epochs)):
                writer.writerow(self.epochs[i])
    
    def generate_graphs(self, save_path=None):
        # Transpose the data to have statistics in columns and epochs in rows
        stats_per_epoch = list(map(list, zip(*self.epochs)))

        # Create subplots
        fig, axs = plt.subplots(len(stats_per_epoch), 1, figsize=(10, 2 * len(stats_per_epoch)))

        # Plot each statistic
        for i, stat in enumerate(stats_per_epoch):
            axs[i].plot(list(range(len(self.epochs))), stat, label=f'Stat {i+1}')
            axs[i].set_title(f'{self.legend[i]} per Epoch')
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel(f'{self.legend[i]}')

        # Adjust layout
        plt.tight_layout()

        # Save the plots to the specified destination
        if save_path:
            plt.savefig(save_path)
        else:
            # Show the plots if no save_path is provided
            plt.show()
        

class OptunaTuning():
    def __init__(self):

        print("Init Optuna Object")

    def start_study(self, trials, filename):
        self.study = optuna.create_study(direction='maximize')
        self.data_path = os.getcwd() + "\\" + filename

        # Read parameters from file
        self.params = load_parameters()

        self.ml_obj = MLAnalysis()
        self.ml_obj.prepare_data_nn(self.data_path)

        self.study.optimize(self.objective, n_trials=trials)
        print('Best hyperparameters: ', self.study.best_params)

    
    def objective(self, trial):
        K = trial.suggest_int('K', 50, 100)
        self.params['K'] = K

        fc_hidden_size = trial.suggest_int('fc_hidden_size', 128, 1024)
        self.params['fc_hidden_size'] = fc_hidden_size

        batch_size = trial.suggest_int('batch_size', 300, 700)
        self.params['batch_size'] = batch_size

        epochs = trial.suggest_int('epochs', 25, 50)
        self.params['epochs'] = epochs

        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
        self.params['learning_rate'] = learning_rate

        results_dict, graphs_dict = self.ml_obj.run_nn_model(self.params)
        f1 = results_dict['f1']

        return f1
        

class EEGTransformer():
    def __init__(self, parameters=None):

        torch.cuda.empty_cache()

        # Check if a GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            print("Using CUDA")
        else:
            print("Using CPU")
        self.name = "transformer1"
        self.model_identifier = None
        self.params = parameters
        self.batch_size = self.params['tr_batch_size']
        self.n_epochs = self.params['tr_epochs']
        self.c_dim = self.params['tr_c_dim']
        self.lr = self.params['tr_learning_rate']
        self.b1 = self.params['tr_b1']
        self.b2 = self.params['tr_b2']
        self.num_heads = self.params['tr_num_heads']
        self.drop_p = self.params['tr_drop_p']
        self.forward_expansion = self.params['tr_forward_expansion']
        self.forward_drop_p = self.params['tr_forward_drop_p']
        self.embedding_size = self.params['tr_emb_size']
        self.depth = self.params['tr_depth']
        self.n_classes = self.params['tr_n_classes']
        self.dimension = (190, 50)

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        # self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_cls = torch.nn.BCELoss().cuda()

        self.model = Conformer(emb_size=self.embedding_size,
                               depth=self.depth,
                               n_classes=self.n_classes,
                               num_heads=self.num_heads,
                               drop_p=self.drop_p,
                               forward_expansion=self.forward_expansion,
                               forward_drop_p=self.forward_drop_p).cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()

        self.stats = Statistics()

        # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label-1).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def load_and_split_data(self, eeg_data, labels, psd_data, split_ratio=0.8):

        self.train_dataset, self.test_dataset, self.train_loader, self.test_loader = load_and_split_data(self,
                                                                                                         eeg_data,
                                                                                                         labels, 
                                                                                                         psd_data, 
                                                                                                         self.batch_size, 
                                                                                                         split_ratio)

    def train_model(self, save=False):

        self.stats = Statistics()
        self.load_and_split_data(self.X, self.y, self.psd, self.params['data_split'])

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        train_start_time = time.time()
        for e in range(self.n_epochs):
            epoch_start_time = time.time()
            self.model.train()
            all_predictions = []
            all_labels = []
            running_loss = 0
            for i, data in enumerate(self.train_loader, 0):
                inputs = data['eeg_data']
                labels = data['label']

                # inputs = inputs.unsqueeze(3)

                # data augmentation M.P. This needs to be worked on
                # aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                # img = torch.cat((img, aug_data))
                # label = torch.cat((label, aug_label))

                tok, outputs = self.model(inputs)
                loss = self.criterion_cls(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                local_outputs = torch.round(outputs)
                local_outputs = local_outputs.cpu().detach().numpy()
                local_labels = labels.cpu().detach().numpy()
                train_f1 = get_f1_score(local_labels, local_outputs)
                train_acc = get_accuracy(local_labels, local_outputs)
                train_precision, train_recall = get_precision_recall(local_labels, local_outputs)
                running_loss += loss.item()
                hours, minutes, seconds = seconds_to_hms(time.time() - train_start_time)
                print(f"Epoch {e} - Batch {i} :",
                      f" F1 {train_f1:.6f},",
                      f" Accuracy {train_acc:.6f},",
                      f" Precision {train_precision:.6f},",
                      f" Recall {train_recall:.6f},",
                      f" Positives {np.sum(local_outputs == 1)}, ",
                      f" Time {hours:.1f}h {minutes:.1f}m {seconds:.1f}s")

                all_predictions.extend(local_outputs)
                all_labels.extend(local_labels)

            # Calculate train stats
            train_f1 = get_f1_score(all_labels, all_predictions)
            train_acc = get_accuracy(all_labels, all_predictions)
            train_precision, train_recall = get_precision_recall(all_labels, all_predictions)
            train_kappa = get_kappa_score(all_labels, all_predictions)
                
            # out_epoch = time.time()

            avg_loss, test_loss, f1, test_acc, precision, recall, kappa = self.evaluate_model_training(self.test_loader,
                                                                                                       self.criterion_cls)

            epoch_time = time.time() - epoch_start_time
            epoch_hours, epoch_minutes, epoch_seconds = seconds_to_hms(epoch_time)
            rem_hours, rem_minutes, rem_seconds = seconds_to_hms(epoch_time * (self.n_epochs - (e + 1)))
            hours, minutes, seconds = seconds_to_hms(time.time() - train_start_time)
            final_time = add_times((hours, minutes, seconds), (rem_hours, rem_minutes, rem_seconds))
            print("==================================================================")
            print(f'Epoch: {e}\n',
                    f'  Train loss: {running_loss / len(self.train_loader):.6f} \n',
                    f'  Train accuracy {train_acc:.6f} \n',
                    f'  Train f1 {train_f1:.6f} \n',
                    f'  Train Kappa {train_kappa:.6f} \n',
                    f'  Train precision {train_precision:.6f} \n',
                    f'  Train recall {train_recall:.6f} \n',
                    f'  Test loss: {avg_loss:.6f} \n',
                    f'  Test accuracy is {test_acc:.6f} \n',
                    f'  Test F1 score is {f1:.6f} \n',
                    f'  Test Kappa score is {kappa:.6f} \n',
                    f'  Test precision is {precision:.6f} \n',
                    f'  Test recall is {recall:.6f} \n\n',
                    f'  Time to train epoch #{e}: {epoch_hours}h {epoch_minutes}m {epoch_seconds}s \n',
                    f'  Time left to train model {rem_hours}h {rem_minutes}m {rem_seconds}s \n',
                    f'  Model final time {final_time[0]}h {final_time[1]}m {final_time[2]}s \n')
            monitor_gpu_usage()
            print("==================================================================")

            if e %self.params['tr_save_model_every_n_epoch'] == 0 and save==True:
                save_model(self.model, self.model_identifier, f"models/epoch_{e}")

            self.stats.update(train_acc, 
                              train_f1, 
                              train_kappa,
                              train_precision, 
                              train_recall, 
                              running_loss / len(self.train_loader), 
                              test_acc, 
                              f1, 
                              kappa,
                              precision, 
                              recall, 
                              avg_loss)

    
    def evaluate_model_training(self, dataloader, criterion):
        self.model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for data in dataloader:
                inputs = data['eeg_data']
                labels = data['label']
                # Forward pass through the model

                tok, outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # Convert outputs to binary predictions
                predictions = torch.round(outputs)
                all_predictions.extend(predictions.cpu().detach().numpy())
                all_labels.extend(labels.cpu().detach().numpy())

        # Calculate F1 score
        f1 = get_f1_score(all_labels, all_predictions)
        acc = get_accuracy(all_labels, all_predictions)
        precision, recall = get_precision_recall(all_labels, all_predictions)
        kappa = get_kappa_score(all_labels, all_predictions)

        return running_loss / len(dataloader), loss, f1, acc, precision, recall, kappa


    def evaluate_model(self):
        # Set the model to evaluation mode
        self.model.eval()

        # Initialize variables to keep track of predictions and labels
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_inputs = []

        threshold = 0.6

        # Iterate over the test loader
        for data in self.test_loader:
            inputs = data['eeg_data']
            labels = data['label'].flatten()
            
            probabilities, outputs = self.model(inputs)

            # Convert the outputs to predictions by taking the argmax
            binary_predictions = (outputs > threshold).int()

            # Append the predictions and labels to the respective lists
            all_inputs.extend(inputs)
            all_predictions.extend(binary_predictions.flatten().tolist())
            all_probabilities.extend(probabilities.tolist())
            all_labels.extend(labels.tolist())

        self.write_to_csv(all_probabilities, all_labels, all_predictions, "testing.csv")
        f1 = get_f1_score(all_labels, all_predictions)
        print(f"[{self.name}]: F1 Macro Score: {f1}")

        acc = get_accuracy(all_labels, all_predictions)
        print(f"[{self.name}]: Accuracy: {acc}")

        roc_auc = get_roc_score(all_labels, all_probabilities)
        print(f"[{self.name}]: Area under ROC curve: {roc_auc}")

        precision, recall = get_precision_recall(all_labels, all_predictions)
        print(f"[{self.name}]: Precision: {precision}")
        print(f"[{self.name}]: Recall: {recall}")

        kappa = get_kappa_score(all_labels, all_predictions)
        print(f"[{self.name}]: Kappa: {kappa}")

        conf_mat = plot_confusion_matrix_display(all_labels, all_predictions)
        roc_curve = plot_roc_curve(all_labels, all_probabilities)
        
        graphs_dict = {'conf_mat': conf_mat, 'roc_curve': roc_curve}
        results_dict = {'f1': f1,
                        'accuracy': acc,
                        'roc_auc': roc_auc,
                        'precision': precision,
                        'recall': recall,
                        'kappa': kappa}

        return results_dict, graphs_dict


    def write_to_csv(self, probabilities, labels, predictions, csv_filename):
        # Check if the lengths of input lists are the same
        if len(probabilities) != len(labels) or len(labels) != len(predictions):
            raise ValueError("Input lists must have the same length")

        # Combine the lists into a list of tuples
        data = list(zip(probabilities, labels, predictions))

        # Open the CSV file for writing
        with open(csv_filename, 'w', newline='') as csvfile:
            # Create a CSV writer
            csv_writer = csv.writer(csvfile)

            # Write header
            csv_writer.writerow(['Probability', 'Label', 'Prediction'])

            # Write data
            csv_writer.writerows(data)

# Define the neural network model
class EEGClassifier(nn.Module):

    def __init__(self, parameters=None):
        super(EEGClassifier, self).__init__()
        # Check if a GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            print("Using CUDA")
        else:
            print("Using CPU")
        
        if parameters==None:
            self.params = load_parameters()
        else:
            self.params = parameters

        self.channels = None
        self.freq = None

        self.data_sample_size = 376
        self.K = self.params['K'] # Size of output layer - Play with this value
        self.in_channels = 62
        self.out_channels = 1
        self.kernel_size_1 = 3
        self.kernel_size_2, self.kernel_stride_2 = self.compute_kernel_stride(self.data_sample_size, self.K)
        self.fc_hidden_size = self.params['fc_hidden_size']
        self.linear_input = (self.data_sample_size - self.kernel_size_2)//self.kernel_stride_2 + 1
        self.psd_size = 74

        self.to(self.device) # move model to device

        
        self.model_name = self.params['model_name']
        self.batch_size = self.params['batch_size']
        self.epochs = self.params['epochs']
        self.learning_rate = self.params['learning_rate']

        self.conv1d_1 = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size_1, stride=1, padding=1).to(self.device)
        self.conv1d_2 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size_2, stride=self.kernel_stride_2).to(self.device)
        self.conv1d_psd = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size_1, stride=1).to(self.device)

        self.batchnorm = nn.BatchNorm1d(self.out_channels).to(self.device)
        self.dropout = nn.Dropout(p=self.params['dropout_prob'])  # Specify dropout probability


        if self.params['psd_usage']==1:
            self.fc1 = nn.Linear(self.linear_input + self.psd_size, self.fc_hidden_size).to(self.device)
        else:
            self.fc1 = nn.Linear(self.linear_input, self.fc_hidden_size).to(self.device)
        self.fc2 = nn.Linear(self.fc_hidden_size, 1).to(self.device)


        # Initialize Weights
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        # init.uniform_(self.fc1.weight, a=0, b=1)  # You can adjust 'a' and 'b' based on your requirements
        # init.uniform_(self.fc2.weight, a=0, b=1)  # You can adjust 'a' and 'b' based on your requirements

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

    def forward(self, eeg, psd=None):
        eeg = self.conv1d_1(eeg)
        eeg = self.conv1d_2(eeg)
        if self.params['batch_norm']==1:
            eeg = self.batchnorm(eeg)
        eeg = eeg.view(eeg.size(0), -1)

        if self.params['psd_usage']==1:
            psd = self.conv1d_psd(psd)
            psd = psd.view(psd.size(0), -1)
            x = torch.cat((eeg, psd), dim=1)
        else:
            x = eeg

        x = self.dropout(x)

        # Fully connected layers
        if self.params['activation']=='relu':
            x = F.relu(self.fc1(x))
        elif self.params['activation']=='elu':
            x = F.elu(self.fc1(x))
        elif self.params['activation']=='leaky_relu':
            x = F.leaky_relu(self.fc1(x))
        elif self.params['activation']=='selu':
            x = F.selu(self.fc1(x))

        x = torch.sigmoid(self.fc2(x))  # binary classification (1 output node)
        # x = x.squeeze()
        return x

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
        return 3, 1 

    def load_and_split_data(self, eeg_data, labels, psd_data, split_ratio=0.8):

        self.train_dataset, self.test_dataset, self.train_loader, self.test_loader = load_and_split_data(self,
                                                                                                         eeg_data,
                                                                                                         labels, 
                                                                                                         psd_data, 
                                                                                                         self.batch_size, 
                                                                                                         split_ratio)


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
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.params['optimizer'] == 'Adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=self.learning_rate)

        # Train the model
        for epoch in range(self.epochs):
            running_loss = 0.0
            all_predictions = []
            all_labels = []
            for i, data in enumerate(self.train_loader, 0):
                inputs = data['eeg_data']
                labels = data['label']
                optimizer.zero_grad()
                # Forward pass through the model
                if self.params['psd_usage'] == 1:
                    psd = data['psd_data']
                    outputs = self(inputs, psd)
                else:
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
            train_f1 = get_f1_score(all_labels, all_predictions)

            # Validation phase (evaluate on the test set)
            test_loss, test_f1 = self.evaluate_model_training(self.test_loader, criterion)

            print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(self.train_loader)}, Training F1 Score: {train_f1}, Test Loss: {test_loss}, Test F1 Score: {test_f1}")

    def evaluate_model_training(self, dataloader, criterion):
        self.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for data in dataloader:
                inputs = data['eeg_data']
                labels = data['label']
                # Forward pass through the model
                if self.params['psd_usage'] == 1:
                    psd = data['psd_data']
                    outputs = self(inputs, psd)
                else:
                    outputs = self(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # Convert outputs to binary predictions
                predictions = torch.round(outputs)
                all_predictions.extend(predictions.cpu().detach().numpy())
                all_labels.extend(labels.cpu().detach().numpy())

        # Calculate F1 score
        f1 = get_f1_score(all_labels, all_predictions)

        return running_loss / len(dataloader), f1

    def write_to_csv(self, probabilities, labels, predictions, csv_filename):
        # Check if the lengths of input lists are the same
        if len(probabilities) != len(labels) or len(labels) != len(predictions):
            raise ValueError("Input lists must have the same length")

        # Combine the lists into a list of tuples
        data = list(zip(probabilities, labels, predictions))

        # Open the CSV file for writing
        with open(csv_filename, 'w', newline='') as csvfile:
            # Create a CSV writer
            csv_writer = csv.writer(csvfile)

            # Write header
            csv_writer.writerow(['Probability', 'Label', 'Prediction'])

            # Write data
            csv_writer.writerows(data)

    # Evaluate the model on the test set
    def evaluate_model(self):
        # Set the model to evaluation mode
        self.eval()

        # Initialize variables to keep track of predictions and labels
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_inputs = []

        threshold = 0.6

        info = mne.create_info(ch_names=self.channels, sfreq=256, ch_types='eeg')

        # Iterate over the test loader
        for data in self.test_loader:
            inputs = data['eeg_data']
            labels = data['label'].flatten()
            
            # Forward pass through the model
            if self.params['psd_usage'] == 1:
                psd = data['psd_data']
                outputs = self(inputs, psd)
            else:
                outputs = self(inputs)

            # Convert the outputs to probabilities
            probabilities = torch.sigmoid(outputs)

            # Convert the outputs to predictions by taking the argmax
            binary_predictions = (outputs > threshold).int()

            # Append the predictions and labels to the respective lists
            all_inputs.extend(inputs)
            all_predictions.extend(binary_predictions.flatten().tolist())
            all_probabilities.extend(probabilities.tolist())
            all_labels.extend(labels.tolist())

        self.write_to_csv(all_probabilities, all_labels, all_predictions, "testing.csv")
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
        
        # fp, fn, tp, tn = get_conf_indices(all_labels, all_predictions)

        # # Get average amplitude of each section        
        # fp_array = [all_inputs[i] for i in fp]
        # fn_array = [all_inputs[i] for i in fn]
        # tp_array = [all_inputs[i] for i in tp]
        # tn_array = [all_inputs[i] for i in tn]

        # fp_array_stacked = torch.stack(fp_array, dim=0)
        # fp_array_mean = torch.mean(fp_array_stacked, dim=2)
        # fn_array_stacked = torch.stack(fn_array, dim=0)
        # fn_array_mean = torch.mean(fn_array_stacked, dim=2)
        # tp_array_stacked = torch.stack(tp_array, dim=0)
        # tp_array_mean = torch.mean(tp_array_stacked, dim=2)
        # tn_array_stacked = torch.stack(tn_array, dim=0)
        # tn_array_mean = torch.mean(tn_array_stacked, dim=2)

        # fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        # im_fp = plot_heatmap(axs[0, 0], fp_array_mean.cpu().numpy(), 'False Positive Array Mean')
        # im_fn = plot_heatmap(axs[0, 1], fn_array_mean.cpu().numpy(), 'False Negative Array Mean')
        # im_tp = plot_heatmap(axs[1, 0], tp_array_mean.cpu().numpy(), 'True Positive Array Mean')
        # im_tn = plot_heatmap(axs[1, 1], tn_array_mean.cpu().numpy(), 'True Negative Array Mean')

        # # Add colorbars
        # cbar_fp = fig.colorbar(im_fp, ax=axs[0, 0])
        # cbar_fn = fig.colorbar(im_fn, ax=axs[0, 1])
        # cbar_tp = fig.colorbar(im_tp, ax=axs[1, 0])
        # cbar_tn = fig.colorbar(im_tn, ax=axs[1, 1])
        # plt.tight_layout()

        # fp_array_mean_2 = torch.mean(fp_array_stacked, dim=1)
        # fn_array_mean_2 = torch.mean(fn_array_stacked, dim=1)
        # tp_array_mean_2 = torch.mean(tp_array_stacked, dim=1)
        # tn_array_mean_2 = torch.mean(tn_array_stacked, dim=1)

        # fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        # im_fp_2 = plot_heatmap(axs[0, 0], fp_array_mean_2.cpu().numpy(), 'False Positive Array Mean')
        # im_fn_2 = plot_heatmap(axs[0, 1], fn_array_mean_2.cpu().numpy(), 'False Negative Array Mean')
        # im_tp_2 = plot_heatmap(axs[1, 0], tp_array_mean_2.cpu().numpy(), 'True Positive Array Mean')
        # im_tn_2 = plot_heatmap(axs[1, 1], tn_array_mean_2.cpu().numpy(), 'True Negative Array Mean')

        # # Add colorbars
        # cbar_fp_2 = fig.colorbar(im_fp, ax=axs[0, 0])
        # cbar_fn_2 = fig.colorbar(im_fn, ax=axs[0, 1])
        # cbar_tp_2 = fig.colorbar(im_tp, ax=axs[1, 0])
        # cbar_tn_2 = fig.colorbar(im_tn, ax=axs[1, 1])
        # plt.tight_layout()
        # plt.show()

        return results_dict, graphs_dict


class MLAnalysis:
    def __init__(self, paramater_filepath=None):
        self.r_seed = 10
        if paramater_filepath == None:
            self.parameter_filepath = os.getcwd() + "\\parameters.yaml"
        self.params = load_parameters()
        self.__set_seed()

        # self.models = {'svm': SVC(probability=True),
        #           'random_forest': RandomForestClassifier()}

        self.models = {'random_forest': RandomForestClassifier()}
        # self.model_gs_keys = {'svm': ['svm', 'svm_poly'],
        #                  'random_forest': ['random_forest']}
        self.model_gs_keys = {'svm': ['svm'],
                         'random_forest': ['random_forest']}
        
    def __set_seed(self):
        random.seed(self.r_seed)
        np.random.seed(self.r_seed)
        torch.manual_seed(self.r_seed)


    def start_transformer_optuna_study(self, trials, params):
        now = dt.now()
        now_string = now.strftime("%d_%m_%y_%H_%M_%S")
        logging.basicConfig(filename=f'optuna_log_{now_string}.txt', level=logging.INFO)
        self.study = optuna.create_study(direction='maximize')

        self.study.optimize(lambda trial: self.transformer_objective(trial, params), n_trials=trials)
        print('Best hyperparameters: ', self.study.best_params)

    def transformer_objective(self, trial, params):

        # Modify params
        self.__init__(self.params)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
        params['tr_learning_rate']= learning_rate

        b1 = trial.suggest_float('b1', 0.7, 0.999)
        params['tr_b1'] = b1

        b2 = trial.suggest_float('b2', 0.7, 0.9999)
        params['tr_b2'] = b2
        
        epochs = trial.suggest_int('epochs', 50, 200)
        params['tr_epochs'] = epochs

        batch_size = trial.suggest_int('batch_size', 32, 50)
        params['tr_batch_size'] = batch_size

        # Initialize the transformer model
        transformer = EEGTransformer(params)

        # Add the data to the transformer
        transformer.X = self.X
        transformer.y = self.y
        transformer.psd = self.psd

        logging.info(f"Optuna parameters for trial {trial.number}: learning_rate={learning_rate}, b1={b1}, b2={b2}, epochs={epochs}, batch_size={batch_size}")
        print(f"Optuna parameters for trial {trial.number}: learning_rate={learning_rate}, b1={b1}, b2={b2}, epochs={epochs}, batch_size={batch_size}")
        # print(f"Optuna parameters: learning_rate={learning_rate}, b1={b1}, b2={b2}, epochs={epochs}, batch_size={batch_size}")

        # Run model
        transformer.train_model()
        results_dict, graph_dict = transformer.evaluate_model()
        logging.info(f"Final results: {results_dict}")
        print(f"Final results: {results_dict}")

        f1 = results_dict['f1']
        torch.cuda.empty_cache()

        return f1
    
    def run_transformer_model(self, params=None, save=False):
        params = load_parameters()
        if params != None:
            transformer = EEGTransformer(params)
        else:
            transformer = EEGTransformer()

        now = dt.now()
        now_string = now.strftime("%d_%m_%y_%H_%M_%S")
        model_identifier = f"{transformer.name}_{params['tr_model_name']}_{now_string}"
        transformer.model_identifier = model_identifier
        
        # print("Building data loaders...", end="")
        # transformer.load_and_split_data(self.X, self.y, self.psd, self.params['data_split'])
        # print("Done", end="\n")

        if params['tr_optuna']==1:
            print(f"Starting Optuna model training with {params['tr_optuna_trials']} trials...", end="\n")
            self.start_transformer_optuna_study(trials=params['tr_optuna_trials'], params=params)
            # transformer.start_optuna_study(trials=params['tr_optuna_trials'])
            print("Optuna model training Done", end="\n")
        else:
            transformer.X = self.X
            transformer.y = self.y
            transformer.psd = self.psd

            print("Starting model training...", end="\n")
            transformer.train_model(save=True)
            print("Model training Done", end="\n")

            print("Evaluating model...", end="\n")
            results_dict, graph_dict = transformer.evaluate_model()
            print("Evaluation done", end="\n")

            if save:
                save_model_stats(model_identifier,
                                self.params, results_dict,
                                graph_dict['conf_mat'],
                                graph_dict['roc_curve'],
                                transformer.model,
                                transformer.stats)
        
        return results_dict, graph_dict



    def run_nn_model(self, params=None, save=False):
        if params != None:
            nn_model = EEGClassifier(params)
        else:
            nn_model = EEGClassifier()

        nn_model.channels = self.channels
        print("Building data loaders...", end="")
        nn_model.load_and_split_data(self.X, self.y, self.psd, 0.8)

        print("Done", end="\n")
        # self.nn_model.load_data(self.X_train, self.y_train, self.X_test, self.y_test)

        print("Starting model training...", end="")
        nn_model.train_model()
        print("Done", end="\n")

        print("Evaluating model...", end="")
        results_dict, graph_dict = nn_model.evaluate_model()
        print("Done", end="\n")

        if save:
            save_model_stats(f"{nn_model.model_name}_{results_dict['f1']}",
                                self.params, results_dict,
                                graph_dict['conf_mat'],
                                graph_dict['roc_curve'],
                                nn_model)
        
        return results_dict, graph_dict


    def prepare_data_nn(self, data_path):
        filepath = data_path + "\\" + self.params['database_name']
        flankerdata = load_flanker_data_from_pickle(filepath)
        
        self.channels = flankerdata.participants[1].channels
        self.channels = [s for s in self.channels if s not in ('Cz', 'ECG1')]
        self.X, self.y, self.psd = flankerdata.concatenate_data()
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





