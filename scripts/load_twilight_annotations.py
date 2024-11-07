import os

import numpy as np
import pandas as pd

from brain_decoding.config.file_path import DATA_PATH

annotation_file = (
    "/Users/XinNiuAdmin/Library/CloudStorage/Box-Box/Vwani_Movie/movie_info/Twilight_Characters_frame-by-frame_Tinn.csv"
)
output_filename = os.path.join(DATA_PATH, "twilight_concepts.npy")

characters = [
    "Alice.Cullen",
    "Angela.Weber",
    "Bella.Swan",
    "Billy.Black",
    "Carlisle.Cullen",
    "Charlie.Swan",
    "Edward.Cullen",
    "Emmett.Cullen",
    "Eric.Yorkie",
    "Jacob.Black",
    "Jasper.Hale",
    "Jessica.Stanley",
    "Mike.Newton",
    "No.Characters",
    "Rosalie.Hale",
    "Side.Character",
    "Renee.Swan",
    "Tyler.Crowley",
]

annotations = pd.read_csv(annotation_file, header=0, index_col=None)
annotations = annotations[annotations["ms"] <= 45 * 60 * 1000]
annotations["time_bin"] = annotations["ms"] // 250
annotations = annotations.groupby("time_bin").max()

np.save(output_filename, annotations[characters].to_numpy().transpose())
print(annotations.shape)
