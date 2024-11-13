from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT_PATH / "data"
PATIENTS_FILE_PATH = ROOT_PATH / "data/patients"
SURROGATE_FILE_PATH = ROOT_PATH / "data/surrogate_windows"
CONFIG_FILE_PATH = ROOT_PATH / "config"
RESULT_PATH = ROOT_PATH / "results"
MOVIE24_LABEL_PATH = f"{DATA_PATH}/8concepts_merged.npy"
TWILIGHT_LABEL_PATH = f"{DATA_PATH}/twilight_concepts.npy"
TWILIGHT_MERGE_LABEL_PATH = f"{DATA_PATH}/twilight_concepts_merged.npy"
MOVIE_LABEL_TWILIGHT_VS_24 = f"{DATA_PATH}/twilight_vs_24.npy"
