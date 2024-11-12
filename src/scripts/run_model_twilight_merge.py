import datetime
import os
import random
import string
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

import wandb
from brain_decoding.config.config import PipelineConfig
from brain_decoding.config.file_path import CONFIG_FILE_PATH, DATA_PATH
from brain_decoding.config.save_config import config
from brain_decoding.main import pipeline, set_config
from brain_decoding.param.base_param import device
from brain_decoding.trainer import Trainer
from brain_decoding.utils.analysis import concept_frequency
from brain_decoding.utils.initializer import initialize_dataloaders, initialize_evaluator, initialize_model

# torch.autograd.set_detect_anomaly(True)
# torch.backends.cuda.matmul.allow_tf32=True
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
# torch.use_deterministic_algorithms(True)

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    patient = 570
    phase_train = "twilight_1"
    phase_test = "sleep_1"
    CONFIG_FILE = CONFIG_FILE_PATH / "config_sleep-None-None_2024-10-16-19:17:43.yaml"

    config = set_config(
        # CONFIG_FILE,
        config,
        patient,
        phase_train,
        phase_test,
    )
    config.experiment.name = "twilight_merged"

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = f"{patient}_{config.data.data_type}_{config.model.architecture}_test_optimalX_CARX_{current_time}"
    output_path = os.path.join(config.data.result_path, config.experiment.name, output_folder)
    config.data.train_save_path = os.path.join(output_path, "train")
    config.data.valid_save_path = os.path.join(output_path, "valid")
    config.data.test_save_path = os.path.join(output_path, "test")
    config.data.memory_save_path = os.path.join(output_path, "memory")

    config.data.movie_label_path = str(DATA_PATH / "twilight_concepts_merged.npy")
    config.data.movie_label_sr = 4  # 1 for 24, 4 for twilight

    config.model.num_labels = 4  # 8 for 24, 18 for twilight, 4 for twilight_merged
    config.experiment.train_phases = ["twilight_1"]
    config.model.labels = ["Bella.Swan", "Edward.Cullen", "No.Characters", "Others"]

    # _, concept_weights = concept_frequency(config.data.movie_label_path, config.model.labels)
    # concept_weights = torch.from_numpy(concept_weights.astype(np.float32))
    # concept_weights = concept_weights.to(device)
    # config.model.train_loss = torch.nn.BCEWithLogitsLoss(reduction="none", weight=concept_weights)

    print("start: ", patient)

    os.environ["WANDB_MODE"] = "offline"
    # os.environ['WANDB_API_KEY'] = '5a6051ed615a193c44eb9f655b81703925460851'
    wandb.login()
    run_name = f"LFP Concept level {config.experiment['patient']} MultiEncoder"
    wandb.init(project="twilight_merge", name=run_name, reinit=True, entity="24")

    trainer = pipeline(config)

    print("Start training")
    trainer.train(config.model["epochs"], 1)
    print("done: ", patient)
    print()
