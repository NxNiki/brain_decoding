import glob
import json

# import librosa
# import librosa.display
# import torch
import math
import re
import time
import warnings
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as LA
from clusterless_clean import *
from general import *
from kneed import KneeLocator
from lfp_helper import *
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks, hilbert, iirnotch, lfilter, sosfiltfilt

# from spike_localization import patient_localization_mapping
from scipy.stats import zscore

OFFSET = {
    "555_1": 4.58,
    "562_1": 134.194,
    "563_1": 25.59,
    "564_1": [(792.79, 1945), (3732.44, 5091)],
    "564_2": [(792.79, 1945), (3732.44, 5091)],
    "565_1": [(917.796, 2490.63), (2692.16, 3599.14 - 43)],
    "565_2": [(3744.36, 4278.64), (4441.95, 6387.16 - 44)],
    "566_1": 1691273171.09898 - 1691272807.193047,
    "567_1": 1692748176.35772 - 1692747794.1907692,
    "568_1": 400.426,
    "570_1": 1706308502.12459 - 1706304396.2999392,
    "572_1": 1711143392.69897 - 1711142763.677046,
    "573_1": 1714775660.18883 - 1714775636.2359192,
    "i717_1": 1699291373.40026 - 1699290962.3770342,
    "i728_1": 1702317343.74561 - 1702316927.5185993,
}

# '566_1': 364.07, #380.5814,
# '567_1': 382.376,
# '568_1': 400.426,
# '570_1': 1706308502.12459 - 1706304396.2999392,
# '572_1': 1711143392.69897 - 1711142763.677046,
# '573_1': 1714775660.18883 - 1714775636.2359192,
# 'i717_1': 1699291373.40026 - 1699290962.3770342,
# 'i728_1': 416.374,
FREE_RECALL_TIME = {
    "555_1": (42 * 60 + 10, 44 * 60 + 19),
    "562_FR1": (43 * 60 + 58, 49 * 60 + 55),
    "562_FR2": (0 * 60 + 20, 5 * 60 + 50),
    "562_3": (1 * 3600 + 1 * 60 + 32, 1 * 3600 + 16 * 60 + 52),  # memory test
    "563_FR1": (42 * 60 + 59, 48 * 60 + 14),
    "563_FR2": (0 * 60 + 49, 5 * 60 + 53),
    "564_1": (1 * 3600 + 30 * 60 + 59, 1 * 3600 + 35 * 60 + 35),
    "564_2": (2 * 60 + 45, 7 * 60 + 48),
    "565_1": (1 * 3600 + 48 * 60 + 55, 1 * 3600 + 53 * 60 + 23),
    "565_2": (8 * 60 + 30, 14 * 60 + 1),
    "565_3": (4 * 60 + 30, 9 * 60 + 35),
    "566_FR1": (48 * 60 + 52, 53 * 60 + 47),
    "566_CR1": (1 * 3600 + 4 * 60 + 30, 1 * 3600 + 13 * 60 + 38),
    "566_FR2": (4 * 60 + 38, 18 * 60 + 25),
    "566_CR2": (29 * 60 + 7, 35 * 60 + 12),
    "566_5": (0, 1 * 3600),
    "566_6": (2 * 60 + 1, 25 * 60 + 20),  # anime
    "566_7": (25 * 60 + 44, 47 * 60 + 5),  # lion
    "567_FR1": (48 * 60 + 25, 52 * 60 + 38),
    "567_CR1": (1 * 3600 + 4 * 60 + 44, 1 * 3600 + 10 * 60 + 14),
    "567_FR2": (2 * 60 + 31, 8 * 60 + 0),
    "567_CR2": (19 * 60 + 27, 24 * 60 + 15),
    "567_Ctrl1": (0 * 60 + 20, 10 * 60 + 20),  # yoda
    "567_Ctrl2": (1 * 60 + 0, 41 * 60 + 0),  # house
    "567_Ctrl1R2": (28 * 60 + 22, 30 * 60 + 20),  # yoda post sleep recall
    "567_Ctrl2R1": (44 * 60 + 2, 47 * 60 + 37),  # house pre sleep recall
    "567_Ctrl2R2": (24 * 60 + 38, 28 * 60 + 2),  # house post sleep recall
    "568_FR1": (49 * 60 + 58, 52 * 60 + 37),
    "568_CR1": (1 * 3600 + 25 * 60 + 0, 1 * 3600 + 31 * 60 + 7),
    "568_FR2": (0 * 60 + 37, 5 * 60 + 44),
    "568_CR2": (19 * 60 + 23, 25 * 60 + 58),
    "568_Ctrl1": (1 * 60 + 0, 44 * 60 + 0),  # nature
    "568_Ctrl1R2": (0 * 60 + 14, 4 * 60 + 29),  # nature post sleep recall
    "570_FR1": (51 * 60 + 33, 58 * 60 + 5),
    "570_CR1": (1 * 3600 + 9 * 60 + 52, 1 * 3600 + 15 * 60 + 45),
    "570_FR2": (1 * 60 + 18, 6 * 60 + 33),
    "570_CR2": (17 * 60 + 45, 23 * 60 + 35),
    "572_FR1": (0 * 60 + 28, 4 * 60 + 30),
    "572_CR1": (16 * 60 + 48, 22 * 60 + 30),
    "572_FR2": (1 * 60 + 22, 5 * 60 + 56),
    "572_CR2": (17 * 60 + 53, 22 * 60 + 31),
    "573_FR1": (0 * 60 + 38, 5 * 60 + 55),
    "573_CR1": (25 * 60 + 44, 31 * 60 + 35),
    "573_FR2": (0 * 60 + 34, 8 * 60 + 30),
    "573_CR2": (20 * 60 + 30, 26 * 60 + 50),
    "i717_FR1": (1 * 60 + 20, 5 * 60 + 9),
    "i717_CR1": (40 * 60 + 36, 45 * 60 + 54),
    "i717_FR2": (2 * 60 + 42, 7 * 60 + 11),
    "i717_CR2": (39 * 60 + 48, 45 * 60 + 20),
    "i728_FR1a": (48 * 60 + 36, 55 * 60 + 45),
    "i728_FR1b": (1 * 60 + 52, 10 * 60 + 0),
    "i728_CR1": (26 * 60 + 58, 34 * 60 + 58),
    "i728_FR2": (0 * 60 + 46, 11 * 60 + 21),
    "i728_CR2": (22 * 60 + 3, 27 * 60 + 33),
    "i728_Ctrl1": (1 * 60 + 0, 13 * 60 + 0),  # control1
    "i728_Ctrl2": (5 * 60 + 0, 27 * 60 + 0),  # control2
    "i728_Ctrl1R1": (13 * 60 + 43, 15 * 60 + 0),
    "i728_Ctrl2R1": (28 * 60 + 38, 36 * 60 + 15),
}
CONTROL = {
    "566": [(121, 1520), (1544, 2825)],
}


def construct_movie_wf(file, patient_number, category, phase):
    data = np.load(file)
    json_path = file.replace(".npz", ".json")
    with open(json_path, "r") as json_file:
        metadata = json.load(json_file)

    n_samples = metadata["params"]["n_samples"]
    win = int(metadata["params"]["waveform_win_length"])

    wf_1d = np.zeros(n_samples)
    wf = data["waveform"]
    indices = data["idx"]
    sf = int(metadata["params"]["SR_Hz"])

    for i, idx in enumerate(indices):
        i1 = idx - win // 2
        i2 = idx + win // 2
        if i1 < 0 or i2 > n_samples:
            print("skip some")
            continue
        if np.any(np.isnan(wf[i])):
            print("{} contains NaN values.".format(os.path.split(file)[-1]))
            continue
        wf_1d[i1:i2] = np.abs(np.min(wf[i]))

    movie_label_path = "/mnt/SSD2/yyding/Datasets/12concepts/12concepts_merged_more.npy"
    movie_label = np.load(movie_label_path)
    # movie_label = np.repeat(movie_label, resolution, axis=1)
    if category == "movie" and isinstance(OFFSET[patient_number + "_" + str(phase)], list):
        if patient_number == "565":
            movie_sample_range = []
            num_samples = 0
            for alignment_offset in OFFSET[patient_number + "_" + str(phase)]:
                sample_range = [alignment_offset[0] * sf, alignment_offset[1] * sf]
                movie_sample_range.append(sample_range)
                num_samples += int((sample_range[1] - sample_range[0]) // sf * 4)
        else:
            alignment_offset = OFFSET[patient_number + "_" + str(phase)][phase - 1]
            movie_sample_range = [alignment_offset[0] * sf, alignment_offset[1] * sf]
            num_samples = int((movie_sample_range[1] - movie_sample_range[0]) / sf * 4)
    else:
        if category == "movie":
            alignment_offset = OFFSET[patient_number + "_" + str(phase)]  # seconds
            movie_sample_range = [alignment_offset * sf, (alignment_offset + movie_label.shape[-1]) * sf]
            num_samples = int(movie_label.shape[-1] * 4)
        elif category == "control":
            alignment_offset = 0
            # control_length = min(z.shape[-1], movie_label.shape[-1] * sf)
            # movie_sample_range = [alignment_offset * sf, alignment_offset * sf + control_length]
            control_length = 250 * sf
            movie_sample_range = [150 * sf, 400 * sf]
            num_samples = control_length / sf * 4
        elif category == "recall":
            alignment_offset = 0
            recall_start = FREE_RECALL_TIME[patient_number + "_" + str(phase)][0]
            recall_end = FREE_RECALL_TIME[patient_number + "_" + str(phase)][1]
            movie_sample_range = [(alignment_offset + recall_start) * sf, (alignment_offset + recall_end) * sf]
            num_samples = int((movie_sample_range[1] - movie_sample_range[0]) / sf * 4)
    if patient_number == "565" and category == "movie":
        movie_wf = []
        for i, (s, e) in enumerate(movie_sample_range):
            movie_wf.append(wf_1d[int(s) : int(e)])
        movie_wf = np.concatenate(movie_wf, axis=0)
    else:
        movie_wf = wf_1d[int(movie_sample_range[0]) : int(movie_sample_range[1])]

    return movie_wf, num_samples, sf


def get_sleep(patient_number, desired_samplerate, mode):
    def sort_filename(filename):
        """Extract the numeric part of the filename and use it as the sort key"""
        return [int(x) if x.isdigit() else x for x in re.findall(r"\d+|\D+", filename)]

    # folder contains the clustless data, I saved the folder downloaded from the drive as '562/clustless_raw'
    spike_path = "E://projects//Datasets//neuron//spike_data//{}//raw_{}//".format(patient_number, mode)
    spike_files = glob.glob(os.path.join(spike_path, "*.npz"))
    spike_files = sorted(spike_files, key=sort_filename)
    # spike_files = ['E://projects//Datasets//neuron//spike_data//566//raw_sleep//clustless_CSC62.npz']
    """
    {0: exp5, 1: exp6, 2: exp7}.
    since we agree to maintain each experiment individually, no longer need this 'phase' parameter
    """
    for _, file in enumerate(spike_files):
        save_folder = "E://projects//Datasets//neuron//spike_data//{}//time_{}//".format(patient_number, mode)
        save_path = os.path.join(save_folder, os.path.split(file)[-1])
        os.makedirs(save_folder, exist_ok=True)
        data = np.load(file)
        json_path = file.replace(".npz", ".json")
        with open(json_path, "r") as json_file:
            metadata = json.load(json_file)

        n_samples = metadata["params"]["n_samples"]
        win = int(metadata["params"]["waveform_win_length"])

        wf_1d = np.zeros(n_samples)
        sf = int(metadata["params"]["SR_Hz"])
        wf = data["waveform"]
        indices = data["idx"]
        for i, idx in enumerate(indices):
            i1 = idx - win // 2
            i2 = idx + win // 2
            if i1 < 0 or i2 > n_samples:
                print("skip some")
                continue
            if np.any(np.isnan(wf[i])):
                print("{} contains NaN values.".format(os.path.split(file)[-1]))
                continue
            wf_1d[i1:i2] = wf[i]

        num_hours = len(wf_1d) // sf // 3600
        resolution = 4
        final_spike_data = []
        for h in range(num_hours):
            movie_wf = wf_1d[h * sf * 3600 : (h + 1) * sf * 3600]
            num_samples = 3600 * resolution
            for second in range(num_samples):
                window_left = second / resolution * sf
                window_right = (second + 1) / resolution * sf
                if window_left < 0 or window_right > movie_wf.shape[-1]:
                    continue
                features = movie_wf[int(window_left) : int(window_right)]
                # features = get_short(features)
                # window_size = 160
                # binned_features = []
                # for i in range(0, len(features), window_size):
                #     window = features[i:i + window_size]
                #     binned_features.append(np.ptp(window))
                # features = np.array(binned_features)
                features = features.reshape(features.shape[0] // 4, 4)
                features = np.mean(features, axis=1)
                # import matplotlib.pyplot as plt
                # plt.stem(features)
                if np.any(np.isnan(features)):
                    print("{} contains NaN values. Fatal!!".format(os.path.split(file)[-1]))
                final_spike_data.append(features)

        final_spike_data = np.array(final_spike_data, dtype=np.float32)
        np.savez(save_path, data=final_spike_data)
        print(os.path.split(file)[-1])


def get_ready(patient_number, desired_samplerate, mode, category="recall", phase=-1):
    def sort_filename(filename):
        """Extract the numeric part of the filename and use it as the sort key"""
        return [int(x) if x.isdigit() else x for x in re.findall(r"\d+|\D+", filename)]

    # folder contains the clustless data, I saved the folder downloaded from the drive as '562/clustless_raw'
    spike_path = "E://projects//Datasets//neuron//spike_data//{}//raw_{}//".format(patient_number, mode)
    spike_files = glob.glob(os.path.join(spike_path, "*.npz"))
    spike_files = sorted(spike_files, key=sort_filename)

    for _, file in enumerate(spike_files):
        save_folder = "E://projects//Datasets//neuron//spike_data//{}//time_{}{}_{}//".format(
            patient_number, category, phase, mode
        )
        save_path = os.path.join(save_folder, os.path.split(file)[-1])
        os.makedirs(save_folder, exist_ok=True)
        data = np.load(file)
        json_path = file.replace(".npz", ".json")
        with open(json_path, "r") as json_file:
            metadata = json.load(json_file)

        n_samples = metadata["params"]["n_samples"]
        win = int(metadata["params"]["waveform_win_length"])

        wf_1d = np.zeros(n_samples)
        wf = data["waveform"]
        indices = data["idx"]
        sf = int(metadata["params"]["SR_Hz"])

        high_sos = signal.butter(4, [300, 3000], "bandpass", output="sos", fs=32000)
        wf = signal.sosfiltfilt(high_sos, wf)
        notch_freq = np.arange(300, 3001, 60)
        notch = [signal.butter(4, (f - 2, f + 2), "bandstop", output="sos", fs=32000) for f in notch_freq]
        for sos in notch:
            wf = signal.sosfiltfilt(sos, wf)

        movie_label_path = "E://projects//Datasets//12concepts//12concepts_merged_more.npy"
        movie_label = np.load(movie_label_path)
        resolution = 4
        # movie_label = np.repeat(movie_label, resolution, axis=1)
        if category == "recall":
            alignment_offset = 0
            movie_sample_range = [
                (alignment_offset + FREE_RECALL_TIME["566_4"][0]) * sf,
                (alignment_offset + FREE_RECALL_TIME["566_4"][1]) * sf,
            ]
            num_samples = int(FREE_RECALL_TIME["566_4"][1] - FREE_RECALL_TIME["566_4"][0]) * resolution
        elif category == "anime":
            alignment_offset = 0
            movie_sample_range = [
                (alignment_offset + CONTROL[patient_number][0][0]) * sf,
                (alignment_offset + CONTROL[patient_number][0][1]) * sf,
            ]
            num_samples = int(CONTROL[patient_number][0][1] - CONTROL[patient_number][0][0]) * resolution
        elif category == "lion":
            alignment_offset = 0
            movie_sample_range = [
                (alignment_offset + CONTROL[patient_number][1][0]) * sf,
                (alignment_offset + CONTROL[patient_number][1][1]) * sf,
            ]
            num_samples = int(CONTROL[patient_number][1][1] - CONTROL[patient_number][1][0]) * resolution
        elif category == "control":
            alignment_offset = 1 * 60
            movie_sample_range = [alignment_offset * sf, (alignment_offset + movie_label.shape[-1]) * sf]
            num_samples = movie_label.shape[-1] * 4
        else:
            alignment_offset = OFFSET[patient_number]  # seconds
            movie_sample_range = [alignment_offset * sf, (alignment_offset + movie_label.shape[-1]) * sf]
            num_samples = movie_label.shape[-1] * 4
        wf = wf[int(movie_sample_range[0]) : int(movie_sample_range[1])]
        final_spike_data = np.zeros_like(wf)

        # negative_idx = flt_data[channel] < 0
        # negative = flt_data[channel][negative_idx]
        negative = wf
        envelope = hilbert(negative)
        envelope = np.abs(envelope)
        threshold = 3 * np.median(np.abs(envelope)) / 0.6745
        peaks, _ = find_peaks(-wf, distance=80, height=threshold)
        for peak in peaks:
            # start = max(0, peak - 30)
            # end = min(len(wf), peak + 30)
            start = peak - 30
            end = peak + 30

            if start < 0 or end > n_samples:
                print("skip some")
                continue
            if np.any(np.isnan(wf[start:end])):
                print("{} contains NaN values.".format(os.path.split(file)[-1]))
                continue
            final_spike_data[start:end] = wf[start:end]

        final_spike_data = final_spike_data.reshape(-1, 160)
        final_spike_data = -np.min(final_spike_data, axis=-1)

        if np.any(np.isnan(final_spike_data)):
            print("{} contains NaN values.".format(os.path.split(file)[-1]))

        final_spike_data = np.array(final_spike_data, dtype=np.float32)
        np.savez(save_path, data=final_spike_data)
        print(os.path.split(file)[-1])


def get_oneshot_blur(patient_number, desired_samplerate, mode, category="recall", phase=-1):
    def sort_filename(filename):
        """Extract the numeric part of the filename and use it as the sort key"""
        return [int(x) if x.isdigit() else x for x in re.findall(r"\d+|\D+", filename)]

    # folder contains the clustless data, I saved the folder downloaded from the drive as '562/clustless_raw'
    spike_path = "E://projects//Datasets//neuron//spike_data//{}//raw_{}//".format(patient_number, mode)
    spike_files = glob.glob(os.path.join(spike_path, "*.npz"))
    spike_files = sorted(spike_files, key=sort_filename)

    for _, file in enumerate(spike_files):
        save_folder = "E://projects//Datasets//neuron//spike_data//{}//time_{}{}_{}//".format(
            patient_number, category, phase, mode
        )
        save_path = os.path.join(save_folder, os.path.split(file)[-1])
        os.makedirs(save_folder, exist_ok=True)
        data = np.load(file)
        json_path = file.replace(".npz", ".json")
        with open(json_path, "r") as json_file:
            metadata = json.load(json_file)

        n_samples = metadata["params"]["n_samples"]
        win = int(metadata["params"]["waveform_win_length"])

        wf_1d = np.zeros(n_samples)
        wf = data["waveform"]
        indices = data["idx"]
        sf = int(metadata["params"]["SR_Hz"])

        for i, idx in enumerate(indices):
            i1 = idx - win // 2
            i2 = idx + win // 2
            if i1 < 0 or i2 > n_samples:
                print("skip some")
                continue
            if np.any(np.isnan(wf[i])):
                print("{} contains NaN values.".format(os.path.split(file)[-1]))
                continue
            wf_1d[i1:i2] = wf[i]

        movie_label_path = "E://projects//Datasets//12concepts//12concepts_merged_more.npy"
        movie_label = np.load(movie_label_path)
        resolution = 4
        # movie_label = np.repeat(movie_label, resolution, axis=1)
        if category == "movie" and isinstance(OFFSET[patient_number + "_" + str(phase)], list):
            if patient_number == "565":
                movie_sample_range = []
                num_samples = 0
                for alignment_offset in OFFSET[patient_number + "_" + str(phase)]:
                    sample_range = [alignment_offset[0] * sf, alignment_offset[1] * sf]
                    movie_sample_range.append(sample_range)
                    num_samples += int((sample_range[1] - sample_range[0]) // sf * 4)
            else:
                alignment_offset = OFFSET[patient_number + "_" + str(phase)][phase - 1]
                movie_sample_range = [alignment_offset[0] * sf, alignment_offset[1] * sf]
                num_samples = int((movie_sample_range[1] - movie_sample_range[0]) / sf * 4)
        else:
            if category == "movie":
                alignment_offset = OFFSET[patient_number + "_" + str(phase)]  # seconds
                movie_sample_range = [alignment_offset * sf, (alignment_offset + movie_label.shape[-1]) * sf]
                num_samples = int(movie_label.shape[-1] * 4)
            elif category == "control":
                alignment_offset = 0
                # control_length = min(z.shape[-1], movie_label.shape[-1] * sf)
                # movie_sample_range = [alignment_offset * sf, alignment_offset * sf + control_length]
                control_length = 250 * sf
                movie_sample_range = [150 * sf, 400 * sf]
                num_samples = control_length / sf * 4
            elif category == "recall":
                alignment_offset = 0
                recall_start = FREE_RECALL_TIME[patient_number + "_" + str(phase)][0]
                recall_end = FREE_RECALL_TIME[patient_number + "_" + str(phase)][1]
                movie_sample_range = [(alignment_offset + recall_start) * sf, (alignment_offset + recall_end) * sf]
                num_samples = int((movie_sample_range[1] - movie_sample_range[0]) / sf * 4)
        if patient_number == "565" and category == "movie":
            movie_wf = []
            for i, (s, e) in enumerate(movie_sample_range):
                movie_wf.append(wf_1d[int(s) : int(e)])
            movie_wf = np.concatenate(movie_wf, axis=0)
        else:
            movie_wf = wf_1d[int(movie_sample_range[0]) : int(movie_sample_range[1])]

        final_spike_data = []
        for second in range(num_samples):
            window_left = second / resolution * sf
            window_right = (second + 1) / resolution * sf
            if window_left < 0 or window_right > movie_wf.shape[-1]:
                continue
            features = movie_wf[int(window_left) : int(window_right)]
            # features = get_short(features)
            window_size = 160
            binned_features = []
            for i in range(0, len(features), window_size):
                window = features[i : i + window_size]
                # binned_features.append(np.ptp(window))
                binned_features.append(np.abs(np.min(window)))
            features = np.array(binned_features)
            # features = features.reshape(features.shape[0] // 4, 4)
            # features = np.mean(features, axis=1)
            # import matplotlib.pyplot as plt
            # plt.stem(features)
            if np.any(np.isnan(features)):
                print("{} contains NaN values.".format(os.path.split(file)[-1]))
            final_spike_data.append(features)

        final_spike_data = np.array(final_spike_data, dtype=np.float32)
        np.savez(save_path, data=final_spike_data)
        print(os.path.split(file)[-1], np.max(final_spike_data))


def get_oneshot_clean(patient_number, desired_samplerate, mode, category="recall", phase=None, version="notch"):
    def sort_filename(filename):
        """Extract the numeric part of the filename and use it as the sort key"""
        return [int(x) if x.isdigit() else x for x in re.findall(r"\d+|\D+", filename)]

    # folder contains the clustless data, I saved the folder downloaded from the drive as '562/clustless_raw'
    spike_path = f"/mnt/SSD2/yyding/Datasets/neuron/spike_data/{patient_number}/raw_{mode}/"
    spike_files = glob.glob(os.path.join(spike_path, "*.csv"))
    spike_files = sorted(spike_files, key=sort_filename)

    for bundle in range(0, len(spike_files), 8):
        bundle_csv = spike_files[bundle : bundle + 8]
        df = load_data_from_bundle(bundle_csv)
        df_clean = cross_chan_event_detection(df, 2, 4)
        # df_clean = cross_chan_binned_clean(df, 3, 4)
        grouped = df_clean.groupby("channel")
        channel_dataframes = {channel: group for channel, group in grouped}
        for channel, data in channel_dataframes.items():
            save_folder = (
                f"/mnt/SSD2/yyding/Datasets/neuron/spike_data/{patient_number}/{version}/time_{category}_{phase}/"
            )
            name = re.sub(r"\d+", "", os.path.split(bundle_csv[0])[-1].split(".csv")[0])
            name_check = re.sub(r"\d+", "", os.path.split(bundle_csv[-1])[-1].split(".csv")[0])
            assert name == name_check, "wrong bundle name"
            file = os.path.join(spike_path, f"{name}{channel}.csv")
            save_path = os.path.join(save_folder, f"{name}{channel}.npz")
            os.makedirs(save_folder, exist_ok=True)

            # for _, file in enumerate(spike_files):
            #     save_folder = f'/mnt/SSD2/yyding/Datasets/neuron/spike_data/{patient_number}/{version}/time_{category}_{phase}/'
            #     save_path = os.path.join(save_folder, os.path.split(file)[-1])
            #     save_path = save_path.replace('.csv', '.npz')
            #     os.makedirs(save_folder, exist_ok=True)
            #     data = pd.read_csv(file)
            json_path = file.replace(".csv", ".json")
            with open(json_path, "r") as json_file:
                metadata = json.load(json_file)

            n_samples = metadata["params"]["n_samples"]
            sf = int(metadata["params"]["fs_estimate_Hz"])
            wf_2d = np.zeros((1, n_samples))

            # filter based on threshold
            # threshold = metadata['params']['across_exp_SD'] * 3
            # data = data[data['amplitude'].abs() > threshold]

            # amplitudes = list(data['amplitude'])
            # indices = list(data['index'])

            # win = 60
            # for i, (idx, amp) in enumerate(zip(indices, amplitudes)):
            #     i1 = idx-win//2
            #     i2 = idx+win//2
            #     if i1 < 0 or i2 > n_samples:
            #         print('skip some')
            #         continue
            #     if amp < 0:
            #         wf_2d[0][i1:i2] = amp
            #     else:
            #         wf_2d[1][i1:i2] = amp
            # sd = metadata['params']['threshold'] / 3
            # data['amplitude'] = -data['amplitude']
            # data1 = data[(data['amplitude'] > sd * 4) & (data['amplitude'] <= sd * 6)]
            # data2 = data[(data['amplitude'] > sd * 6) & (data['amplitude'] <= sd * 10)]
            # data3 = data[(data['amplitude'] > sd * 10) & (data['amplitude'] <= sd * 20)]
            # amplitudes1 = list(data1['amplitude'])
            # indices1 = list(data1['index'])
            # amplitudes2 = list(data2['amplitude'])
            # indices2 = list(data2['index'])
            # amplitudes3 = list(data3['amplitude'])
            # indices3 = list(data3['index'])
            # for i, (idx, amp) in enumerate(zip(indices1, amplitudes1)):
            #     wf_2d[0][idx] = amp
            # for i, (idx, amp) in enumerate(zip(indices2, amplitudes2)):
            #     wf_2d[1][idx] = amp
            # for i, (idx, amp) in enumerate(zip(indices3, amplitudes3)):
            #     # if amp > 500:
            #     #     continue
            #     wf_2d[2][idx] = amp

            # sd = metadata['params']['threshold'] / 3
            # data['amplitude'] = -data['amplitude']
            # data1 = data[(data['amplitude'] >= sd * 5) & (data['amplitude'] <= sd * 30)]
            # amplitudes1 = list(data1['amplitude'])
            # indices1 = list(data1['index'])
            # for i, (idx, amp) in enumerate(zip(indices1, amplitudes1)):
            #     wf_2d[0][idx] = amp

            data["amplitude"] = -data["amplitude"]
            sd = metadata["params"]["threshold"] / 3
            ratio_list = np.arange(3, 5.1, 0.1)
            ratio_list = np.round(ratio_list, 1)
            peak_num_list = []
            for ratio in ratio_list:
                dd = data[(data["amplitude"] >= sd * ratio) & (data["amplitude"] <= sd * 30)]
                amp = list(dd["amplitude"])
                peak_num_list.append(len(amp))

            # # Create an index array (x-axis)
            xx = np.arange(len(peak_num_list))

            # Use KneeLocator to find the elbow (knee point)
            knee_locator = KneeLocator(xx, peak_num_list, curve="convex", direction="decreasing")

            # Get the index of the elbow
            elbow_index = knee_locator.elbow

            first_order_diff = np.diff(peak_num_list)
            peak_thres = -10000
            flatten_index = np.argmax(first_order_diff >= peak_thres)
            # cutoff_rr = ratio_list[flatten_index-1]
            # cutoff_rr = ratio_list[flatten_index]
            cutoff_rr = ratio_list[elbow_index - 1]
            # if cutoff_rr != 5.0:
            #     continue
            # # Plot the data and mark the elbow
            # plt.figure(figsize=(10, 6))
            # plt.plot(xx, peak_num_list, label='Decreasing Data')
            # plt.axvline(elbow_index, color='red', linestyle='--', label=f'Elbow at sd {ratio_list[elbow_index]}')
            # plt.xticks(xx, ratio_list)
            # plt.title('Elbow Detection')
            # plt.xlabel('Index')
            # plt.ylabel('Values')
            # plt.legend()
            # plt.savefig('elbow.png')
            # plt.close()

            cutoff_rr = 3.5
            print(cutoff_rr)
            step = 0.5
            for rr in np.arange(cutoff_rr, 30 + step, step):
                data1 = data[(data["amplitude"] >= sd * rr) & (data["amplitude"] < sd * (rr + step))]
                amplitudes1 = list(data1["amplitude"])
                indices1 = list(data1["index"])
                for i, (idx, amp) in enumerate(zip(indices1, amplitudes1)):
                    wf_2d[0, idx] = np.max([wf_2d[0, idx], rr])

            rr = 30 + step
            data1 = data[data["amplitude"] >= sd * rr]
            amplitudes1 = list(data1["amplitude"])
            indices1 = list(data1["index"])
            for i, (idx, amp) in enumerate(zip(indices1, amplitudes1)):
                wf_2d[0, idx] = np.max([wf_2d[0, idx], 32])
            # sd = metadata['params']['threshold'] / 3
            # data['amplitude'] = -data['amplitude'] # only once
            # step = 0.5
            # for rr in np.arange(3, 30.5, step):
            #     data1 = data[(data['amplitude'] >= sd * rr) & (data['amplitude'] <= sd * (rr+step))]
            #     amplitudes1 = list(data1['amplitude'])
            #     indices1 = list(data1['index'])
            #     for i, (idx, amp) in enumerate(zip(indices1, amplitudes1)):
            #         wf_2d[0, idx] = np.max([wf_2d[0, idx], rr])

            # rr = 30.5
            # data1 = data[data['amplitude'] >= sd * rr]
            # amplitudes1 = list(data1['amplitude'])
            # indices1 = list(data1['index'])
            # for i, (idx, amp) in enumerate(zip(indices1, amplitudes1)):
            #     wf_2d[0, idx] = np.max([wf_2d[0, idx], 33])

            movie_label_path = "/mnt/SSD2/yyding/Datasets/12concepts/12concepts_merged_more.npy"
            movie_label = np.load(movie_label_path)
            resolution = 4
            # movie_label = np.repeat(movie_label, resolution, axis=1)
            if category == "movie" and isinstance(OFFSET[patient_number + "_" + str(phase)], list):
                if patient_number == "565":
                    movie_sample_range = []
                    num_samples = 0
                    for alignment_offset in OFFSET[patient_number + "_" + str(phase)]:
                        sample_range = [alignment_offset[0] * sf, alignment_offset[1] * sf]
                        movie_sample_range.append(sample_range)
                        num_samples += int((sample_range[1] - sample_range[0]) // sf * 4)
                else:
                    alignment_offset = OFFSET[patient_number + "_" + str(phase)][phase - 1]
                    movie_sample_range = [alignment_offset[0] * sf, alignment_offset[1] * sf]
                    num_samples = int((movie_sample_range[1] - movie_sample_range[0]) / sf * 4)
            else:
                if category == "movie":
                    alignment_offset = OFFSET[patient_number + "_" + str(phase)]  # seconds
                    movie_sample_range = [alignment_offset * sf, (alignment_offset + movie_label.shape[-1]) * sf]
                    num_samples = int(movie_label.shape[-1] * 4)
                elif category == "control":
                    alignment_offset = 0
                    # control_length = min(z.shape[-1], movie_label.shape[-1] * sf)
                    # movie_sample_range = [alignment_offset * sf, alignment_offset * sf + control_length]
                    control_length = 250 * sf
                    movie_sample_range = [150 * sf, 400 * sf]
                    num_samples = control_length / sf * 4
                elif category == "recall":
                    alignment_offset = 0
                    recall_start = FREE_RECALL_TIME[patient_number + "_" + str(phase)][0]
                    recall_end = FREE_RECALL_TIME[patient_number + "_" + str(phase)][1]
                    movie_sample_range = [(alignment_offset + recall_start) * sf, (alignment_offset + recall_end) * sf]
                    num_samples = int((movie_sample_range[1] - movie_sample_range[0]) / sf * 4)
            if patient_number == "565" and category == "movie":
                movie_wf = []
                for i, (s, e) in enumerate(movie_sample_range):
                    movie_wf.append(wf_2d[:, int(s) : int(e)])
                movie_wf = np.concatenate(movie_wf, axis=0)
            else:
                movie_wf = wf_2d[:, int(movie_sample_range[0]) : int(movie_sample_range[1])]

            final_spike_data = []
            for second in range(num_samples):
                window_left = second / resolution * sf
                window_right = (second + 1) / resolution * sf

                """if overlap"""
                # window_left = window_left - sf / resolution / 2
                # window_right = window_right + sf / resolution / 2

                # rr = second % 4
                # mm = second // 4
                # partition = [(0, 12800), (6400, 19200), (12800, 25600), (19200, 32000)]
                # window_left = mm * sf + partition[rr][0]
                # window_right = mm * sf + partition[rr][1]

                if window_left < 0 or window_right > movie_wf.shape[-1]:
                    continue
                features = movie_wf[:, int(window_left) : int(window_right)]
                # features = get_short(features)
                window_size = 160
                binned_features = []
                for i in range(0, features.shape[-1], window_size):
                    window = features[:, i : i + window_size]
                    # masked_window = np.ma.masked_equal(window, 0)
                    # res = masked_window.mean(axis=1).filled(0)
                    # non_zero_mask = window != 0
                    # magnitude = np.sum(window * non_zero_mask, axis=1)
                    # intensity = np.sum(non_zero_mask, axis=1) / window_size
                    # res = magnitude * intensity

                    # res = np.sum(window, axis=1)
                    # if res[0] > 0:
                    #     res[0] = 1
                    # if res[1] > 0:
                    #     res[1] = 2
                    # if res[2] > 0:
                    #     res[2] = 3

                    non_zero_mask = window != 0
                    non_zero_sum = np.sum(window * non_zero_mask, axis=1)
                    non_zero_count = np.count_nonzero(non_zero_mask, axis=1)
                    res = np.divide(
                        non_zero_sum, non_zero_count, out=np.zeros_like(non_zero_sum), where=non_zero_count != 0
                    )
                    # res = [np.max(window)]
                    binned_features.append(res)
                features = np.column_stack(binned_features)
                # features = features.reshape(features.shape[0] // 4, 4)
                # features = np.mean(features, axis=1)
                # import matplotlib.pyplot as plt
                # plt.stem(features)
                if np.any(np.isnan(features)):
                    print("{} contains NaN values.".format(os.path.split(file)[-1]))
                final_spike_data.append(features)

            final_spike_data = np.array(final_spike_data, dtype=np.float32)
            num_samples_with_non_negative = np.sum(np.any(final_spike_data != 0, axis=(1, 2)))
            np.savez(save_path, data=final_spike_data)
            print(
                os.path.split(file)[-1],
                np.max(final_spike_data),
                np.min(final_spike_data),
                num_samples_with_non_negative,
            )


def get_oneshot_by_region(patient_number, desired_samplerate, mode, category="recall", phase=None, version="notch"):
    def sort_filename(filename):
        """Extract the numeric part of the filename and use it as the sort key"""
        return [int(x) if x.isdigit() else x for x in re.findall(r"\d+|\D+", filename)]

    # folder contains the clustless data, I saved the folder downloaded from the drive as '562/clustless_raw'
    spike_path = f"/mnt/SSD2/yyding/Datasets/neuron/spike_data/{patient_number}/raw_{mode}/"
    spike_files = glob.glob(os.path.join(spike_path, "*.npz"))
    spike_files = sorted(spike_files, key=sort_filename)
    # region_map = patient_localization_mapping[patient_number]
    region_map = {}  # patient_localization_mapping not available

    # Create a reverse mapping from index to region
    index_to_region = {}
    for region, indices in region_map.items():
        for index in indices:
            index_to_region[index] = region
    grouped_paths = {region: [] for region in region_map}

    resolution = 4
    poyo_dict = {}
    for file in spike_files:
        max_magnitude = -np.inf
        index = int(re.search(r"clustless_CSC(\d+)\.npz", file).group(1))
        assert index is not None
        region = index_to_region[index]
        grouped_paths[region].append(file)

        movie_wf, num_samples, sf = construct_movie_wf(file, patient_number, category, phase)
        for second in range(num_samples):
            window_left = second / resolution * sf
            window_right = (second + 1) / resolution * sf
            if window_left < 0 or window_right > movie_wf.shape[-1]:
                continue
            features = movie_wf[int(window_left) : int(window_right)]
            magnitudes, times = np.unique(features, return_index=True)
            magnitudes = np.round(magnitudes)
            times = times / len(features)
            max_magnitude = max(max_magnitude, np.max(magnitudes))

            non_zero_mask = magnitudes != 0
            magnitudes_non_zero = magnitudes[non_zero_mask].tolist() if np.any(non_zero_mask) else []
            times_non_zero = times[non_zero_mask].tolist() if np.any(non_zero_mask) else []

            sp = poyo_dict.setdefault(second, {})
            ch = sp.setdefault(index, {})
            tm = ch.setdefault("timestamps", times_non_zero)
            mg = ch.setdefault("magnitudes", magnitudes_non_zero)
            rg = ch.setdefault("regions", [region] * len(magnitudes_non_zero))

            # tm.append(times_non_zero)
            # mg.append(magnitudes_non_zero)
            # rg.append(region)
        print("max mag: ", max_magnitude)
    save_folder = f"/mnt/SSD2/yyding/Datasets/neuron/spike_data/{patient_number}/{version}/time_{category}_{phase}/"
    os.makedirs(save_folder, exist_ok=True)

    with open(os.path.join(save_folder, "poyo_data.json"), "w") as json_file:
        json.dump(poyo_dict, json_file, indent=4)


if __name__ == "__main__":
    version = "notch CAR-clean-cutoff-ch4"
    get_oneshot_clean("563", 1000, "presleep", category="movie", phase=1, version=version)
    get_oneshot_clean("563", 1000, "presleep", category="recall", phase="FR1", version=version)
    get_oneshot_clean("563", 1000, "postsleep", category="recall", phase="FR2", version=version)
    print()
    get_oneshot_clean("562", 1000, "presleep", category="movie", phase=1, version=version)
    get_oneshot_clean("562", 1000, "presleep", category="recall", phase="FR1", version=version)
    get_oneshot_clean("562", 1000, "postsleep", category="recall", phase="FR2", version=version)
    print()

    get_oneshot_clean("i728", 1000, "presleep1", category="movie", phase=1, version=version)
    get_oneshot_clean("i728", 1000, "presleep1", category="recall", phase="FR1a", version=version)
    get_oneshot_clean("i728", 1000, "presleep2", category="recall", phase="FR1b", version=version)
    get_oneshot_clean("i728", 1000, "presleep2", category="recall", phase="CR1", version=version)
    get_oneshot_clean("i728", 1000, "postsleep", category="recall", phase="FR2", version=version)
    get_oneshot_clean("i728", 1000, "postsleep", category="recall", phase="CR2", version=version)
    print()
    get_oneshot_clean("572", 1000, "presleep1", category="movie", phase=1, version=version)
    get_oneshot_clean("572", 1000, "presleep2", category="recall", phase="FR1", version=version)
    get_oneshot_clean("572", 1000, "presleep2", category="recall", phase="CR1", version=version)
    get_oneshot_clean("572", 1000, "postsleep", category="recall", phase="FR2", version=version)
    get_oneshot_clean("572", 1000, "postsleep", category="recall", phase="CR2", version=version)
    print()
    get_oneshot_clean("567", 1000, "presleep", category="movie", phase=1, version=version)
    get_oneshot_clean("567", 1000, "presleep", category="recall", phase="FR1", version=version)
    get_oneshot_clean("567", 1000, "presleep", category="recall", phase="CR1", version=version)
    get_oneshot_clean("567", 1000, "postsleep", category="recall", phase="FR2", version=version)
    get_oneshot_clean("567", 1000, "postsleep", category="recall", phase="CR2", version=version)
    print()
    get_oneshot_clean("566", 1000, "presleep", category="movie", phase=1, version=version)
    get_oneshot_clean("566", 1000, "presleep", category="recall", phase="FR1", version=version)
    get_oneshot_clean("566", 1000, "presleep", category="recall", phase="CR1", version=version)
    get_oneshot_clean("566", 1000, "postsleep", category="recall", phase="FR2", version=version)
    get_oneshot_clean("566", 1000, "postsleep", category="recall", phase="CR2", version=version)
    print()
    get_oneshot_clean("570", 1000, "presleep", category="movie", phase=1, version=version)
    get_oneshot_clean("570", 1000, "presleep", category="recall", phase="FR1", version=version)
    get_oneshot_clean("570", 1000, "presleep", category="recall", phase="CR1", version=version)
    get_oneshot_clean("570", 1000, "postsleep", category="recall", phase="FR2", version=version)
    get_oneshot_clean("570", 1000, "postsleep", category="recall", phase="CR2", version=version)

    print()
    get_oneshot_clean("573", 1000, "movie", category="movie", phase=1, version=version)
    get_oneshot_clean("573", 1000, "presleep", category="recall", phase="FR1", version=version)
    get_oneshot_clean("573", 1000, "presleep", category="recall", phase="CR1", version=version)
    get_oneshot_clean("573", 1000, "postsleep", category="recall", phase="FR2", version=version)
    get_oneshot_clean("573", 1000, "postsleep", category="recall", phase="CR2", version=version)

    print()
    get_oneshot_clean("i717", 1000, "movie", category="movie", phase=1, version=version)
    get_oneshot_clean("i717", 1000, "presleep", category="recall", phase="FR1", version=version)
    get_oneshot_clean("i717", 1000, "presleep", category="recall", phase="CR1", version=version)
    get_oneshot_clean("i717", 1000, "postsleep", category="recall", phase="FR2", version=version)
    get_oneshot_clean("i717", 1000, "postsleep", category="recall", phase="CR2", version=version)

    print()
    get_oneshot_clean("568", 1000, "presleep", category="movie", phase=1, version=version)
    get_oneshot_clean("568", 1000, "presleep", category="recall", phase="FR1", version=version)
    get_oneshot_clean("568", 1000, "presleep", category="recall", phase="CR1", version=version)
    get_oneshot_clean("568", 1000, "postsleep", category="recall", phase="FR2", version=version)
    get_oneshot_clean("568", 1000, "postsleep", category="recall", phase="CR2", version=version)
