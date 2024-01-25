
from collections import Counter
from datetime import datetime
from paths import PARAMS_DIR, CONFIG_PATH, PLOTS_PATH, RESULTS_PATH
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import GPUtil
from preprocessing import oversample_classes, undersample_classes

def monitor_gpu_usage():
    """
    Monitor GPU usage and print the information.

    This function retrieves GPU information using the GPUtil library and prints the usage and memory information for each GPU. 

    Parameters:
        None

    Returns:
        None

    Raises:
        GPUtil.GPUException: If there is an error retrieving GPU information.

    Example usage:
        >>> monitor_gpu_usage()
        GPU Usage:
        GPU 1: GeForce GTX 1080 Ti
          - GPU Usage: 50.00%
          - GPU Memory Usage: 30.00%
          
        GPU 2: GeForce GTX 2080
          - GPU Usage: 70.00%
          - GPU Memory Usage: 40.00%
    """
    try:
        # Get GPU information
        gpus = GPUtil.getGPUs()

        print("GPU Usage:")
        for i, gpu in enumerate(gpus):
            print(f"GPU {i + 1}: {gpu.name}")
            print(f"  - GPU Usage: {gpu.load * 100:.2f}%")
            print(f"  - GPU Memory Usage: {gpu.memoryUtil * 100:.2f}%")
            print()

    except GPUtil.GPUException as e:
        print(f"Error: {e}")

def add_times(time1, time2):
    """
    Add two times (hours, minutes, seconds).

    Parameters:
    - time1: Tuple of (hours, minutes, seconds) for the first time
    - time2: Tuple of (hours, minutes, seconds) for the second time

    Returns:
    - Tuple representing the sum of the two times
    """
    total_seconds = (time1[0] + time2[0]) * 3600 + (time1[1] + time2[1]) * 60 + (time1[2] + time2[2])
    
    # Calculate new hours, minutes, seconds
    new_hours, remainder = divmod(total_seconds, 3600)
    new_minutes, new_seconds = divmod(remainder, 60)

    return new_hours, new_minutes, new_seconds

def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return hours, minutes, seconds
class CustomDataset(Dataset):
    def __init__(self, eeg_data, labels, device):
        self.eeg_data = eeg_data
        self.labels = labels
        self.device = device

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg_sample = torch.Tensor(self.eeg_data[idx]).to(self.device)
        label = torch.Tensor([self.labels[idx]]).to(self.device)

        return {'eeg_data': eeg_sample, 'label': label}
    
def load_and_split_data(self, eeg_data, labels, batch_size, params):
    check_class_distribution(labels)

    train_size = int(params['data_split'] * len(labels))
    test_size = len(labels) - train_size

    # Generate random indices for the random split
    indices = list(range(len(labels)))
    train_indices, test_indices = random_split(indices, [train_size, test_size])

    # Use the indices to split the eeg_data and labels
    train_eeg_data = eeg_data[train_indices]
    train_labels = labels[train_indices]

    test_eeg_data = eeg_data[test_indices]
    test_labels = labels[test_indices]

    if params['undersampling'] == 1:
        print("Performing under sampling:")
        # Apply sampling strategy
        train_eeg_data, train_labels = undersample_classes(train_eeg_data, train_labels)
        check_class_distribution(train_labels)
    elif params['oversampling'] == 1:
        
        # Reshape X first
        X_shape = train_eeg_data.shape
        X_reshaped = train_eeg_data.reshape(train_eeg_data.shape[0], -1)
        if params['os_strategy'] == 'SMOTE':
            print("Performing oversampling with SMOTE oversampler")
            X_reshaped, train_labels = oversample_classes(X_reshaped, train_labels, strategy="SMOTE",
                                                            ratio=params['smote_ratio'])
        elif params['os_strategy'] == 'random':
            print("Performing oversampling with random oversampler")
            X_reshaped, train_labels = oversample_classes(X_reshaped, train_labels, strategy="random",
                                                            ratio=params['smote_ratio'])
        train_eeg_data = X_reshaped.reshape(-1, *X_shape[1:])
        check_class_distribution(train_labels)

    
    # # Convert the resampled data to PyTorch tensors
    # train_eeg_data = torch.Tensor(train_eeg_data).to(self.device)
    # train_labels = torch.Tensor(train_labels).to(self.device)

    # # Convert test data to PyTorch tensors
    # test_eeg_data = torch.Tensor(test_eeg_data).to(self.device)
    # test_labels = torch.Tensor(test_labels).to(self.device)
        
    # Create CustomDataset instances for training and testing data
    train_dataset = CustomDataset(train_eeg_data, train_labels, self.device)
    test_dataset = CustomDataset(test_eeg_data, test_labels, self.device)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # # Create list of dictionaries for each sample
    # train_data = [{'eeg_data': eeg, 'label': label} for eeg, label in zip(train_eeg_data, train_labels)]
    # test_data = [{'eeg_data': eeg, 'label': label} for eeg, label in zip(test_eeg_data, test_labels)]

    # # Create DataLoader instances
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def check_class_distribution(y):
    class_distribution = Counter(y)
    # Print the class distribution
    for label, count in class_distribution.items():
        print(f"Class {label}: {count} samples")


def plot_heatmap(ax, data, title):
    im = ax.imshow(data, cmap='viridis', aspect='auto', interpolation='none')
    ax.set_title(title)
    ax.set_xlabel('Channels')
    ax.set_ylabel('Epochs')
    return im

def save_model(model, model_name, additional_save_path=None):
    # Save model to a pickle file
    if additional_save_path != None:
        path = RESULTS_PATH / model_name / additional_save_path
    else:
        path = RESULTS_PATH / model_name
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), f'{path}/model.pth')

def save_model_stats(model_name, params, results, conf_mat, roc_curve, model=None, stats=None, additional_params=None):
    # Save model parameters and statistics to a text file
    root_path = RESULTS_PATH / model_name
    if not root_path.is_dir():
        root_path.mkdir(parents=True, exist_ok=False)

    save_path = RESULTS_PATH / model_name / "results.txt"
    with open(save_path, 'w') as file:
        file.write("Model Parameters:\n")
        for key, value in params.items():
            file.write(f"{key}: {value}\n")

        if additional_params != None:
            file.write("Additional Parameters:\n")
            for key, value in additional_params.items():
                file.write(f"{key}: {value}\n")

        file.write(" ======================= Model Statistics: ========================\n\n")
        file.write(f"Accuracy: {results['accuracy']:.4f}\n\n")
        file.write(f"F1 Score: {results['f1']:.4f}\n\n")
        file.write(f"ROC AUC Score: {results['roc_auc']:.4f}\n\n")
        file.write(f"Precision Score: {results['precision']:.4f}\n\n")
        file.write(f"Recall Score: {results['recall']:.4f}\n\n")
        if results['kappa'] is not None:
            file.write(f"Kappa Score: {results['kappa']:.4f}\n\n") 
        file.write("====================================================================\n\n")

        save_plot(conf_mat, f"{model_name}_conf_mat.png", root_path)
        save_plot(roc_curve, f"{model_name}_roc_curve.png", root_path)
        if model!=None:
            save_model(model, model_name)

        if stats!=None:
            stats.save_to_csv(RESULTS_PATH / model_name / "stats.csv")
            stats.generate_graphs(RESULTS_PATH / model_name / 'graph.png')

def save_params(params, model_name=None):
    """
    Save a model's parameter json to file with a filename specified in 'filename'.
    Filename should not be a path. The file's path will be the root / params directory.
    """

    if type(params) is not dict:
        raise TypeError("The params to be saved are not in a dict. They are: {}".format(type(params)))
    else:

        if model_name:
            now = datetime.now()
            now_string = now.strftime("%d_%m_%y_%H_%M_%S")
            filename = (f"{model_name}_params_{now_string}.json")

        file_path = PARAMS_DIR / filename
        if not file_path.is_dir():
            file_path.mkdir(parents=True, exist_ok=False)
        with open(file_path, "w") as fout:
            json.dump(params, fout)

        print("Parameters were saved to {}".format(file_path))


def load_params(filename):
    """
    Handles finding the parameters file json, and loading it into a python dictionary to be used with
    scikit models.
    """
    filepath = PARAMS_DIR / filename
    if not filepath.is_file():
        raise FileNotFoundError("Parameter file {} does not exist.".format(filepath.resolve()))

    with open(filepath, "r") as fin:
        params = json.load(fin)

    return params


def check_pos_int_arg(val):
    """
    Used to validate and convert command line user input to int.
    """
    i = int(val)
    if i < 0:
        argparse.ArgumentTypeError("{} is not a positive integer value.".format(val))
    else:
        return i


def check_pos_float_arg(val):
    """
    Used to validate and convert command line user input to float.
    """
    i = float(val)
    if i < 0:
        argparse.ArgumentTypeError("{} is not a positive float value.".format(val))
    else:
        return i


def save_plot(p, filename, path=None):
    """
    Helper function used to save a plot as a png to the plot save directory.
    """

    if path==None:
        p_path = Path(PLOTS_PATH)
        if not p_path.is_dir(): # Make sure plot dir exists.
            p_path.mkdir(parents=False, exist_ok=False)
    else:
        p_path = path

    # Save plot
    filepath = p_path / filename

    try:
        # Case where p is a matplotlib figure
        p.savefig(filepath)
    except AttributeError:
        pass
    try:
        # Case where p is a Display object from scikit
        p.figure_.savefig(filepath)
    except Exception as err:
        raise