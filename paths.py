from pathlib import Path

# Paths to all the subdirectories of the project

ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
PARAMS_DIR = ROOT_DIR / "params"
CONFIG_PATH = ROOT_DIR / "config.json"
PLOTS_PATH = ROOT_DIR / "saved_plots"
RESULTS_PATH = ROOT_DIR / "results"