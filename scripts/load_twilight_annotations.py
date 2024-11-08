import os

import numpy as np
import pandas as pd

from brain_decoding.config.file_path import DATA_PATH
from brain_decoding.param.param_data import TWILIGHT_LABELS, TWILIGHT_LABELS_MERGE

annotation_file = (
    "/Users/XinNiuAdmin/Library/CloudStorage/Box-Box/Vwani_Movie/movie_info/Twilight_Characters_frame-by-frame_Tinn.csv"
)
output_filename = os.path.join(DATA_PATH, "twilight_concepts.npy")

annotations = pd.read_csv(annotation_file, header=0, index_col=None)
annotations = annotations[annotations["ms"] <= 45 * 60 * 1000]
annotations["time_bin"] = annotations["ms"] // 250
annotations = annotations.groupby("time_bin").max()

np.save(output_filename, annotations[TWILIGHT_LABELS].to_numpy().transpose())
print(annotations.shape)

# merge concepts:
other_characters = [c for c in TWILIGHT_LABELS if c not in TWILIGHT_LABELS_MERGE]
annotations["Others"] = annotations[other_characters].max(axis=1) - annotations[TWILIGHT_LABELS_MERGE[:-1]].max(axis=1)
annotations["Others"] = annotations["Others"].clip(lower=0)
np.save(output_filename.replace(".npy", "_merged.npy"), annotations[TWILIGHT_LABELS_MERGE].to_numpy().transpose())
