import copy
import glob
import os
import pickle
import random
import re
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from scipy.stats import zscore
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler, WeightedRandomSampler
from torchvision.transforms import transforms

from brain_decoding.config.config import PipelineConfig
from brain_decoding.dataloader.clusterless_clean import sort_file_name
from brain_decoding.dataloader.save_clusterless import PREDICTION_FS
from brain_decoding.param.param_data import SF


class NeuronDataset:
    def __init__(self, config: PipelineConfig):
        self.patient = config.experiment["patient"]
        self.use_spontaneous = config.experiment["use_spontaneous"]
        self.use_spike = config.experiment["use_spike"]
        self.use_lfp = config.experiment["use_lfp"]
        self.use_overlap = config.experiment["use_overlap"]
        self.use_combined = config.experiment["use_combined"]
        self.lfp_data_mode = config.data["lfp_data_mode"]
        self.spike_data_mode = config.data["spike_data_mode"]
        self.spike_data_sd = config.data["spike_data_sd"]

        # assume in label/sec
        self.movie_sampling_rate = config.data["movie_sampling_rate"]
        self.movie_label_path = config.data["movie_label_path"]

        self.resolution = PREDICTION_FS
        self.lfp_sf = SF  # Hz
        self.ml_label = np.load(self.movie_label_path)
        # self.ml_label = np.append(self.ml_label, np.zeros((1, self.ml_label.shape[1])), axis=0)
        self.ml_label = np.repeat(self.ml_label, int(self.resolution / config.data.movie_label_sr), axis=1)

        self.smoothed_ml_label = np.copy(self.ml_label)  # self.smooth_label()
        self.data = defaultdict()
        self.label = []
        self.smoothed_label = []
        self.lfp_channel_by_region = {}

        # create spike data
        if self.use_spike:
            self.data["clusterless"] = self.load_data(config.data["spike_path"], config.experiment.train_phases)

        # create lfp data
        if self.use_lfp:
            self.data["lfp"] = self.load_data(config.data["lfp_path"], config.experiment.train_phases)

        for c, category in enumerate(self.spike_data_sd):
            # size = sample_size[c]
            self.label.append(self.ml_label.transpose().astype(np.float32))
            self.smoothed_label.append(self.smoothed_ml_label.transpose().astype(np.float32))

        self.label = np.concatenate(self.label, axis=0)
        self.smoothed_label = np.concatenate(self.smoothed_label, axis=0)

        if self.use_overlap:
            self.label = self.label[1:-1]
            self.smoothed_label = self.smoothed_label[1:-1]
        # filter low occurrence samples
        class_value, class_count = np.unique(self.label[:, 0:8], axis=0, return_counts=True)
        occurrence_threshold = 200 * len(self.spike_data_sd)
        good_indices = np.where(class_count >= occurrence_threshold)[0]
        indices_of_good_samples = []
        for index in good_indices:
            label = class_value[index]
            label_indices = np.where((self.label[:, 0:8] == label[None, :]).all(axis=1))[0]
            indices_of_good_samples.extend(label_indices)
        indices_of_good_samples = sorted(indices_of_good_samples)

        self.label = self.label[indices_of_good_samples]
        self.smoothed_label = self.smoothed_label[indices_of_good_samples]

        print("Neuron Data Loaded")
        self.preprocess_data()
        if config.experiment["use_augment"]:
            self.time_backword()
        if config.experiment["use_shuffle_diagnostic"]:
            # self.brute_shuffle()
            self.circular_shift()

    def load_data(self, data_path: str, categories: List[str]) -> np.ndarray[Any]:
        sample_size = []
        spike_data = []
        for c, category in enumerate(categories):
            version = self.spike_data_mode
            for sd in self.spike_data_sd:
                spike_path = os.path.join(
                    data_path,
                    str(self.patient),
                    version,
                    "time_{}".format(category.lower()),
                )
                spike_files = glob.glob(os.path.join(spike_path, "*.npz"))
                spike_files = sorted(spike_files, key=sort_file_name)
                spike_data.append(self.load_clustless(spike_files, sd))
                sample_size.append(spike_data[-1].shape[0])

        return np.concatenate(spike_data, axis=0)

    def smooth_label(self):
        sigma = 1
        kernel = np.exp(-(np.arange(-1, 2) ** 2) / (2 * sigma**2))
        kernel /= np.sum(kernel)
        # kernel = np.tile(kernel, (12, 1))

        smoothed_label = convolve1d(self.ml_label, kernel, axis=1)
        max_val = np.max(smoothed_label, axis=1)
        # smoothed_label = smoothed_label / max_val[:, np.newaxis]
        return np.round(smoothed_label, 2)

    @staticmethod
    def channel_max(data: np.ndarray[float]) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        """

        :param data:
        :return:
        """
        b, c, h, w = data.shape
        normalized_data = data.transpose(2, 0, 1, 3).reshape(h, -1)
        vmax = np.max(normalized_data, axis=1)
        vmin = np.min(normalized_data, axis=1)
        epsilon = 1e-10
        vmax = np.where(vmax == 0, epsilon, vmax)
        return vmax, vmin

    def load_clustless(self, files, sds):
        if not files:
            raise ValueError("input files are empty!")

        spike = []
        for file in files:
            data = np.load(file)["data"]
            spike.append(data[:, :, None])

        spike = np.concatenate(spike, axis=2)
        spike[spike < sds] = 0
        # spike[spike > 500] = 0
        vmax = np.max(spike)
        normalized_spike = spike / vmax
        return normalized_spike

    def load_lfp(self, files):
        lfp = []
        for file in files:
            data = np.load(file)["data"]

            """
            filter out noisy channel
            """
            noisy_channel = {"FUS": [0, 3], "PARS": [6], "PHC": [1]}
            soz = ["LTC", "HPC"]
            region = file.split("marco_lfp_spectrum_")[-1].split(".npz")[0]
            if region in soz:
                continue
            if region in noisy_channel:
                mask = np.ones(data.shape[1], dtype=bool)
                num = len(mask) // 8
                ignore_list = []
                for i in noisy_channel[region]:
                    tmp = list(np.arange(i, len(mask), num))
                    ignore_list += tmp
                mask[ignore_list] = False
                data = data[:, mask, :]
            """
            filter out noisy channel
            """

            self.lfp_channel_by_region[file.split("marco_lfp_spectrum_")[-1].split(".npz")[0]] = data.shape[1]
            if len(data.shape) == 2:
                lfp.append(data[:, None, :])
            else:
                lfp.append(data)

        lfp = np.concatenate(lfp, axis=1)
        vmin = np.min(lfp)
        vmax = np.max(lfp)
        normalized_lfp = (lfp - vmin) / (vmax - vmin)
        return normalized_lfp[:, None]

    def preprocess_data(self):
        pass

    def time_backword(self):
        fliped = np.flip(self.data, axis=-1)
        self.data = np.concatenate((self.data, fliped), axis=0)
        self.label = np.repeat(self.label, 2, axis=0)
        self.smoothed_label = np.repeat(self.smoothed_label, 2, axis=0)

    def brute_shuffle(self):
        b, c, h, w = self.data.shape
        data = self.data.transpose(2, 0, 1, 3).reshape(h, -1)
        shuffled_data = np.apply_along_axis(np.random.permutation, axis=1, arr=data)
        self.data = shuffled_data.reshape(h, b, c, w).transpose(1, 2, 0, 3)

    def circular_shift(self):
        for key in self.data.keys():
            data = self.data[key]
            b, c, h, w = data.shape
            shift_amount = np.random.randint(100, b - 100)
            self.data[key] = np.roll(data, shift=shift_amount, axis=0)

    def __len__(self):
        return len(self.label)


class MyDataset(Dataset):
    def __init__(self, lfp_data, spike_data, label, indices, transform=None, pos_weight=None):
        self.lfp_data = None
        self.spike_data = None
        if lfp_data is not None:
            self.lfp_data = lfp_data
        if spike_data is not None:
            self.spike_data = spike_data
        self.label = label
        self.transform = transform
        self.pos_weight = pos_weight
        self.indices = indices

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        label = self.label[index]
        idx = self.indices[index]
        if self.lfp_data is not None and self.spike_data is None:
            # lfp = {key: value[index] for key, value in self.lfp_data.items()}
            lfp = self.lfp_data[index]
            return lfp, label, idx
        elif self.lfp_data is None and self.spike_data is not None:
            spike = self.spike_data[index]
            return spike, label, idx

        lfp = self.lfp_data[index]
        spike = self.spike_data[index]
        # if self.transform is not None:
        #     neuron_feature = self.transform(neuron_feature)
        # if self.transform:
        #     random_number = random.random()
        #     if random_number < 0.5:
        #         neuron_feature = random_shift(neuron_feature, 2)
        return (lfp, spike), label, idx


def create_weighted_loaders(
    dataset: NeuronDataset,
    config: PipelineConfig,
    batch_size: int = 128,
    seed: int = 42,
    p_val: float = 0.1,
    batch_sample_num: int = 2048,
    shuffle: bool = True,
    transform: bool = None,
):
    # assert 0 < p_val < 1.0, 'p_val must be greater than 0 and smaller than 1'
    if p_val > 0:
        assert 0 < p_val < 1.0, "p_val must be greater than 0 and smaller than 1"
        dataset_size = len(dataset)
        all_indices = list(range(dataset_size))

        class_value, class_count = np.unique(dataset.label, axis=0, return_counts=True)
        class_weight_dict = {key.tobytes(): dataset_size / value for key, value in zip(class_value, class_count)}
        data_weights = np.array([class_weight_dict[label.tobytes()] for label in dataset.label])
        tag_combinations = np.apply_along_axis(lambda x: "".join(map(str, x)), 1, dataset.label)
        unique_combinations, indices = np.unique(tag_combinations, return_inverse=True)
        grouped_indices = {tag: np.where(indices == i)[0] for i, tag in enumerate(unique_combinations)}

        def find_continuous_chunks(index_array):
            chunks = np.split(index_array, np.where(np.diff(index_array) != 1)[0] + 1)
            return chunks

        train_indices = []
        val_indices = []

        for tag, idx_group in grouped_indices.items():
            continuous_chunks = find_continuous_chunks(idx_group)
            for chunk in continuous_chunks:
                chunk_length = len(chunk)
                if chunk_length > 1:
                    val_size_start = max(1, int(np.floor(chunk_length * 0.1)))
                    val_size_end = max(1, int(np.ceil(chunk_length * 0.1)))

                    val_indices_start = chunk[:val_size_start]
                    val_indices_end = chunk[-val_size_end:]
                    train_indices_chunk = chunk[val_size_start:-val_size_end] if chunk_length > 2 else []

                    val_indices.extend(val_indices_start)
                    val_indices.extend(val_indices_end)
                    train_indices.extend(train_indices_chunk)
                else:
                    val_indices.extend(chunk)

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)

        train_label_save_path = os.path.join(config.data["test_save_path"], "train_label")
        np.save(train_label_save_path, dataset.label[train_indices])

        val_label_save_path = os.path.join(config.data["test_save_path"], "val_label")
        np.save(val_label_save_path, dataset.label[val_indices])

        if config.experiment["use_lfp"] and not config.experiment["use_combined"]:
            val_save_path = os.path.join(config.data["test_save_path"], "val_lfp")
            # np.save(val_save_path, {key: value[val_indices] for key, value in dataset.lfp_data.items()})
            np.save(val_save_path, dataset.data[val_indices])
        elif config.experiment["use_spike"] and not config.experiment["use_combined"]:
            val_save_path = os.path.join(config.data["test_save_path"], "val_clusterless")
            np.save(val_save_path, dataset.data[val_indices])
        elif config.experiment["use_combiend"]:
            val_save_path = os.path.join(config.data["test_save_path"], "val_lfp")
            np.save(val_save_path, dataset.data[val_indices])
            val_save_path = os.path.join(config.data["test_save_path"], "val_clusterless")
            np.save(val_save_path, dataset.data[val_indices])

        assert len(set(val_indices)) + len(set(train_indices)) == len(all_indices)
        if shuffle:
            # np.random.seed(seed)
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)

        if config.experiment["use_combined"]:
            spike_train = dataset.data["clusterless"][train_indices]
            spike_val = dataset.data["clusterless"][val_indices]
            lfp_train = dataset.data["lfp"][train_indices]
            lfp_val = dataset.data["lfp"][val_indices]
        else:
            spike_train = dataset.data[train_indices] if config.experiment["use_spike"] else None
            spike_val = dataset.data[val_indices] if config.experiment["use_spike"] else None
            lfp_train = dataset.data[train_indices] if config.experiment["use_lfp"] else None
            lfp_val = dataset.data[val_indices] if config.experiment["use_lfp"] else None

        label_train = dataset.smoothed_label[train_indices]
        label_val = dataset.smoothed_label[val_indices]
        # label_train = dataset.label[train_indices]
        # label_val = dataset.label[val_indices]
        train_pos = label_train.sum(axis=0)
        train_neg = label_train.shape[0] - train_pos
        train_pos_weights = train_neg / train_pos
        val_pos = label_val.sum(axis=0)
        val_neg = label_val.shape[0] - val_pos
        val_pos_weights = val_neg / val_pos

        train_dataset = MyDataset(
            lfp_train,
            spike_train,
            label_train,
            train_indices,
            transform=transform,
            pos_weight=train_pos_weights,
        )
        val_dataset = MyDataset(lfp_val, spike_val, label_val, val_indices, pos_weight=val_pos_weights)
        test_dataset = None

        num_workers = 1
        pin_memory = True

        sampler_train = WeightedRandomSampler(
            weights=data_weights[train_indices],
            num_samples=batch_sample_num,
            replacement=True,
        )

        # sampler_val = WeightedRandomSampler(weights=data_weights[val_indices], num_samples=len(val_dataset), replacement=True)
        sampler_val = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=sampler_val,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )

        test_loader = None
    elif p_val == 0:
        dataset_size = len(dataset)
        all_indices = list(range(dataset_size))

        class_value, class_count = np.unique(dataset.label[:, 0:8], axis=0, return_counts=True)

        class_weight_dict = {key.tobytes(): dataset_size / value for key, value in zip(class_value, class_count)}

        data_weights = np.array([class_weight_dict[label.tobytes()] for label in dataset.label[:, 0:8][all_indices]])
        train_indices = np.array(all_indices)

        os.makedirs(config.data["test_save_path"], exist_ok=True)
        label_save_path = os.path.join(config.data["test_save_path"], "train_label.npy")
        np.save(label_save_path, dataset.label[train_indices])

        if shuffle:
            # np.random.seed(seed)
            np.random.shuffle(train_indices)

        spike_train = dataset.data["clusterless"][train_indices] if config.experiment["use_spike"] else None
        lfp_train = dataset.data["lfp"][train_indices] if config.experiment["use_lfp"] else None

        label_train = dataset.smoothed_label[train_indices]
        # label_train = dataset.label[train_indices]
        # label_val = dataset.label[val_indices]
        train_pos = label_train.sum(axis=0)
        train_neg = label_train.shape[0] - train_pos
        # train_pos_weights = np.sqrt(train_neg / train_pos)
        train_pos_weights = train_neg / train_pos

        train_dataset = MyDataset(
            lfp_train,
            spike_train,
            label_train,
            train_indices,
            transform=transform,
            pos_weight=train_pos_weights,
        )

        sampler_train = WeightedRandomSampler(
            weights=data_weights[train_indices],
            num_samples=batch_sample_num,
            replacement=True,
        )

        # sampler_val = WeightedRandomSampler(weights=data_weights[val_indices], num_samples=len(val_dataset), replacement=True)
        sampler_val = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler_train,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )

        test_loader = None
        val_loader = None
    # Return the training, validation, test DataLoader objects
    return train_loader, val_loader, test_loader


def create_inference_loaders(
    dataset,
    batch_size=128,
    seed=42,
    batch_sample_num=2048,
    shuffle=False,
    extras={},
):
    num_workers = 1
    pin_memory = False
    inference_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )

    return inference_loader
