
from collections import Counter
from datetime import datetime
from paths import PARAMS_DIR, CONFIG_PATH, PLOTS_PATH, RESULTS_PATH
import json
from pathlib import Path


def check_class_distribution(y):
    class_distribution = Counter(y)
    # Print the class distribution
    for label, count in class_distribution.items():
        print(f"Class {label}: {count} samples")

def save_model_stats(model_name, params, results, conf_mat, roc_curve):
    # Save model parameters and statistics to a text file
    root_path = RESULTS_PATH / model_name
    if not root_path.is_dir():  # Make sure plot dir exists.
        root_path.mkdir(parents=False, exist_ok=False)

    save_path = RESULTS_PATH / model_name / "results.txt"
    with open(save_path, 'w') as file:
        file.write("Model Parameters:\n")
        for key, value in params.items():
            file.write(f"{key}: {value}\n")

        file.write("Model Statistics:\n")
        file.write(f"Accuracy: {results['accuracy']:.4f}\n\n")
        file.write(f"F1 Score: {results['f1']:.4f}\n\n")
        file.write(f"ROC AUC Score: {results['roc_auc']:.4f}\n\n")
        file.write(f"Precision Score: {results['precision']:.4f}\n\n")
        file.write(f"Recall Score: {results['recall']:.4f}\n\n")

        now = datetime.now()
        now_string = now.strftime("%d_%m_%y_%H_%M_%S")
        save_plot(conf_mat, f"{model_name}_conf_mat_{now_string}.png", root_path)
        save_plot(roc_curve, f"{model_name}_roc_curve_{now_string}.png", root_path)

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