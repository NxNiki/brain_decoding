import os

import numpy as np
import pandas as pd

from brain_decoding.config.file_path import TWILIGHT_LABEL_PATH, TWILIGHT_MERGE_LABEL_PATH
from brain_decoding.param.param_data import TWILIGHT_ANNOTATION_FS, TWILIGHT_LABELS, TWILIGHT_LABELS_MERGE

annotation_file = (
    "/Users/XinNiuAdmin/Library/CloudStorage/Box-Box/Vwani_Movie/movie_info/Twilight_Characters_frame-by-frame_Tinn.csv"
)

annotations = pd.read_csv(annotation_file, header=0, index_col=None)
annotations = annotations[annotations["ms"] <= 45 * 60 * 1000]
annotations["time_bin"] = annotations["ms"] // (1000 / TWILIGHT_ANNOTATION_FS)
annotations = annotations.groupby("time_bin").max()

np.save(TWILIGHT_LABEL_PATH, annotations[TWILIGHT_LABELS].to_numpy().transpose())
print(annotations.shape)

# merge concepts:
other_characters = [c for c in TWILIGHT_LABELS if c not in TWILIGHT_LABELS_MERGE]
annotations["Others"] = 0
annotations["No.Characters"] = 0
annotations.loc[
    annotations[[x for x in TWILIGHT_LABELS if x not in TWILIGHT_LABELS_MERGE]].sum(axis=1) > 0, "Others"
] = 1

# recalculate No.Characters to avoid overlap at edges:
annotations.loc[annotations[[x for x in TWILIGHT_LABELS if x != "No.Characters"]].sum(axis=1) == 0, "No.Characters"] = 1

annotations.loc[annotations[TWILIGHT_LABELS].sum(axis=1) > 1, "Bella.Swan"] = 0
annotations.loc[annotations[TWILIGHT_LABELS].sum(axis=1) > 1, "Edward.Cullen"] = 0
np.save(TWILIGHT_MERGE_LABEL_PATH, annotations[TWILIGHT_LABELS_MERGE].to_numpy().transpose())
