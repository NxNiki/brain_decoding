import numpy as np
import pandas as pd
import os
import pickle
from scipy.io import loadmat
from scipy.stats import zscore
import matplotlib.pyplot as plt
# from params import sorted_channels, taylor_channel_path, tonmoy_channel_path, path_shift_index, bin_len
from scipy.interpolate import interp1d
import torch
import re
import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torchvision.transforms import transforms

class VwaniDataset(Dataset):
    def __init__(self, config):
        self.patient = config['patient']
        self.use_spike = config['use_spike']
        self.use_lfp = config['use_lfp']
        self.use_combined = config['use_combined']
        self.use_sleep = config['use_sleep']
        self.phase = config['free_recall_phase']
        self.lfp_data_mode = config['lfp_data_mode']
        self.spike_data_mode = config['spike_data_mode']
        self.lfp_channel_by_region = {}

        self.lfp_data = None
        self.spike_data = None
        self.data = []

        self.data_path = os.path.join('/mnt/SSD2/yyding/24', 'results', config['model_name'], 'test')

        def sort_filename(filename):
                """Extract the numeric part of the filename and use it as the sort key"""
                return [int(x) if x.isdigit() else x for x in re.findall(r'\d+|\D+', filename)]

        if self.use_spike:
            spike_file = os.path.join(self.data_path, 'val_clusterless.npy')
            self.spike_data = np.load(spike_file)
            self.data.append(self.spike_data)

        # if LFP
        if self.use_lfp:
            lfp_file = os.path.join(self.data_path, 'val_lfp.npy')
            self.lfp_data = np.load(lfp_file)
            self.data.append(self.lfp_data)

        if self.use_combined:
             self.data = {'clusterless': self.data[0], 'lfp': self.data[1]}
        else:
            self.data = self.data[0]
        del self.lfp_data
        del self.spike_data
        self.preprocess_data()
        print("Done")

    def preprocess_data(self):
        if self.use_combined:
            assert self.data['clusterless'].shape[0] == self.data['lfp'].shape[0]
            length = self.data['clusterless'].shape[0]
        else:
            length = self.data.shape[0]
        self.data_length = length

    def __len__(self):
        return self.data_length

class InferenceDataset(Dataset):
    def __init__(self, config):
        self.patient = config['patient']
        self.use_spike = config['use_spike']
        self.use_lfp = config['use_lfp']
        self.use_combined = config['use_combined']
        self.use_sleep = config['use_sleep']
        self.phase = config['free_recall_phase']
        self.lfp_data_mode = config['lfp_data_mode']
        self.spike_data_mode = config['spike_data_mode_inference']
        self.spike_data_sd = config['spike_data_sd_inference']
        self.lfp_channel_by_region = {}

        self.lfp_data = None
        self.spike_data = None
        self.data = []

        def sort_filename(filename):
                """Extract the numeric part of the filename and use it as the sort key"""
                return [int(x) if x.isdigit() else x for x in re.findall(r'\d+|\D+', filename)]
        
        if self.use_spike:
            if self.use_sleep:
                spike_path = os.path.join(config['spike_path'], self.patient, 'time_sleep')
                spike_files = glob.glob(os.path.join(spike_path, '*.npz'))
                spike_files = sorted(spike_files, key=sort_filename)
                self.spike_data = self.load_clustless(spike_files)
            else:
                if isinstance(self.phase, str) and 'all' in self.phase:
                    if self.patient == 'i728':
                        phases = ['FR1a', 'FR1b']
                    else:
                        phases = ['FR1', 'CR1']
                    for phase in phases:
                        version = self.spike_data_mode  
                        spike_path = os.path.join(config['spike_path'], self.patient, version, 'time_recall_{}'.format(phase))
                        spike_files = glob.glob(os.path.join(spike_path, '*.npz'))
                        spike_files = sorted(spike_files, key=sort_filename)
                        self.spike_data = self.load_clustless(spike_files)
                elif isinstance(self.phase, str) and 'control' in self.phase:
                    version = self.spike_data_mode  
                    spike_path = os.path.join(config['spike_path'], self.patient, version, 'time_{}'.format(self.phase))
                    spike_files = glob.glob(os.path.join(spike_path, '*.npz'))
                    spike_files = sorted(spike_files, key=sort_filename)
                    self.spike_data = self.load_clustless(spike_files)
                elif isinstance(self.phase, str) and 'movie' in self.phase:
                    version = self.spike_data_mode  
                    spike_path = os.path.join(config['spike_path'], self.patient, version, 'time_{}'.format(self.phase))
                    spike_files = glob.glob(os.path.join(spike_path, '*.npz'))
                    spike_files = sorted(spike_files, key=sort_filename)
                    self.spike_data = self.load_clustless(spike_files)
                else:
                    version = self.spike_data_mode  
                    spike_path = os.path.join(config['spike_path'], self.patient, version, 'time_recall_{}'.format(self.phase))
                    spike_files = glob.glob(os.path.join(spike_path, '*.npz'))
                    spike_files = sorted(spike_files, key=sort_filename)
                    self.spike_data = self.load_clustless(spike_files)
            self.data.append(self.spike_data)

        # if LFP
        if self.use_lfp:
            if self.use_sleep:
                version = ''
                lfp_path = os.path.join(config['lfp_path'], self.patient, version, 'spectrogram_sleep')
                lfp_files = glob.glob(os.path.join(lfp_path, '*.npz'))
                lfp_files = sorted(lfp_files, key=sort_filename)
                self.lfp_data = self.load_lfp(lfp_files)
            else:
                if isinstance(self.phase, str) and 'all' in self.phase:
                    if self.patient == 'i728':
                        phases = [1, 3]
                    else:
                        phases = [1, 2]
                    for phase in phases:
                        version = self.lfp_data_mode
                        # value = self.lfp_data.setdefault(version, [])
                        lfp_path = os.path.join(config['lfp_path'], self.patient, version, 'spectrogram_recall_{}'.format(phase))
                        lfp_files = glob.glob(os.path.join(lfp_path, '*.npz'))
                        lfp_files = sorted(lfp_files, key=sort_filename)
                        lfp_data = self.load_lfp(lfp_files)
                        self.lfp_data = lfp_data
                elif isinstance(self.phase, str) and 'control' in self.phase:
                    version = self.lfp_data_mode
                    # value = self.lfp_data.setdefault(version, [])
                    lfp_path = os.path.join(config['lfp_path'], self.patient, version, 'spectrogram_{}'.format(self.phase))
                    lfp_files = glob.glob(os.path.join(lfp_path, '*.npz'))
                    lfp_files = sorted(lfp_files, key=sort_filename)
                    lfp_data = self.load_lfp(lfp_files)
                    self.lfp_data = lfp_data
                else:
                    version = self.lfp_data_mode
                    # value = self.lfp_data.setdefault(version, [])
                    lfp_path = os.path.join(config['lfp_path'], self.patient, version, 'spectrogram_recall_{}'.format(self.phase))
                    lfp_files = glob.glob(os.path.join(lfp_path, '*.npz'))
                    lfp_files = sorted(lfp_files, key=sort_filename)
                    lfp_data = self.load_lfp(lfp_files)
                    self.lfp_data = lfp_data

            # self.lfp_data = {key: np.concatenate(value_list, axis=0) for key, value_list in self.lfp_data.items()}
            self.data.append(self.lfp_data)

        if self.use_combined:
             self.data = {'clusterless': self.data[0], 'lfp': self.data[1]}
        else:
            self.data = self.data[0]
        del self.lfp_data
        del self.spike_data
        self.preprocess_data()
        print("Done")

    @staticmethod
    def channel_max(data):
        b, c, h, w = data.shape
        normalized_data = data.transpose(2, 0, 1, 3).reshape(h, -1)
        vmax = np.max(normalized_data, axis=1)
        vmin = np.min(normalized_data, axis=1)
        epsilon = 1e-10
        vmax = np.where(vmax == 0, epsilon, vmax)
        return vmax, vmin
    
    def load_clustless(self, files):
        spike = []
        for file in files:
            data = np.load(file)['data']
            spike.append(data[:, :, None])

        spike = np.concatenate(spike, axis=2)
        # sd1 = spike[:, 0:1, :, :]
        # sd2 = spike[:, 1:2, :, :]
        # sd3 = spike[:, 2:3, :, :]
        # spike = np.maximum(np.maximum(sd1, sd2), sd3)
        
        # data_max = np.max(spike)
        # for i in range(5):
        #     spike[(spike > i/5 * data_max) & (spike <= (i+1)/5 * data_max)] = i+1
        # spike[spike < self.spike_data_sd] = 0
        # vmax, vmin = self.channel_max(spike)
        # normalized_spike = 2 * (spike - vmin[None, None, :, None]) / (vmax[None, None, :, None] - vmin[None, None, :, None]) - 1
        spike[spike < self.spike_data_sd] = 0
        # spike[spike > 500] = 0
        vmax = np.max(spike)
        normalized_spike = spike / vmax
        return normalized_spike
        # outlier = 500
        # spike[np.abs(spike) > outlier] = 0
        # # p_n = positive - negative
        # # non_zero_count = (np.abs(negative) != 0).astype(int) + (positive != 0).astype(int)
        # # p_n = np.divide(np.abs(negative) + positive, non_zero_count, out=np.zeros_like(positive, dtype=np.float32), where=non_zero_count!=0)
        # b, c, h, w = spike.shape 
        # normalized_spike = spike.transpose(2, 0, 1, 3).reshape(h, -1)
        # vmax = np.max(normalized_spike, axis=1)
        # epsilon = 1e-10
        # vmax = np.where(vmax == 0, epsilon, vmax)
        # normalized_spike = spike / vmax[None, None, :, None]

        # negative = normalized_spike[:, 0]
        # positive = normalized_spike[:, 1]
        # return np.abs(negative)[:, None]
    # def load_clustless(files):
    #     spike = []
    #     for file in files:
    #         data = np.load(file)['data']
    #         spike.append(data[:, None])

    #     spike = np.concatenate(spike, axis=1)

    #     # outlier = spike.flatten()[spike.flatten() != 0]
    #     # outlier = np.percentile(outlier, 99.99)
    #     outlier = 500
    #     spike[np.abs(spike) > outlier] = 0

    #     b, h, w = spike.shape 
    #     normalized_spike = spike.transpose(1, 0, 2).reshape(h, -1)
    #     # normalized_spike = zscore(normalized_spike, axis=1)
    #     normalized_spike = normalized_spike.reshape(h, b, w).transpose(1, 0, 2)

    #     vmin = np.min(normalized_spike)
    #     vmax = np.max(normalized_spike)
    #     normalized_spike = (normalized_spike - vmin) / (vmax - vmin)
    #     return normalized_spike[:, None]

    def load_lfp(self, files):
        lfp = []
        for file in files:
            data = np.load(file)['data']

            """
            filter out noisy channel
            """
            noisy_channel = {'FUS': [0, 3], 'PARS': [6], 'PHC': [1]}
            soz = ['LTC', 'HPC']
            region = file.split('marco_lfp_spectrum_')[-1].split('.npz')[0]
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

            self.lfp_channel_by_region[file.split('marco_lfp_spectrum_')[-1].split('.npz')[0]] = data.shape[1]
            if len(data.shape) == 2:
                lfp.append(data[:, None, :])
            else:
                lfp.append(data)

        lfp = np.concatenate(lfp, axis=1)
        vmin = np.min(lfp)
        vmax = np.max(lfp)
        normalized_lfp = (lfp - vmin) / (vmax - vmin)
        return normalized_lfp[:, None]

    def interpolate_neural_data(self, data, original_timestamps):
        new_timestamps = np.arange(original_timestamps[0], original_timestamps[-1] + 1, step=1)
        f = interp1d(original_timestamps, data, axis=-1)
        new_data = f(new_timestamps)
        return new_data, new_timestamps

    def load_channels(self):
        spike_times = []
        channel_labels = []
        path = self.channel_path
        for channel in self.sorted_channels:
            try:
                spike_data = mat73.loadmat(os.path.join(path, f"times_CSC{channel}.mat"))
                print(channel, ' load with mat73')
            except:
                spike_data = loadmat(os.path.join(path, f"times_CSC{channel}.mat"))
                print(channel, ' load with scipy')
            cluster_class = spike_data["cluster_class"]
            n_count = np.max(cluster_class, axis=0)[0]
            # print(f"channel {channel} has {n_count} neurons")
            # print(f"n_count: {int(n_count)}")
            for neuron in range(1, int(n_count) + 1):
                # print(neuron)
                spike_times.append((cluster_class[np.where(cluster_class[:, 0] == neuron)])[:, 1])
                channel_labels.append(f"CSC{channel}_N{neuron}")
        return spike_times

    def load_npz(self, mode='multi'):
        def superVstack(a, b):
            # make it so you can vstack onto empty row
            if len(a) == 0:
                stack = b
            elif len(b) == 0:
                stack = a
            else:
                stack = np.vstack([a, b])
            return stack

        if mode == 'multi':
            lfp_mat = []
            lfp_files = glob.glob(os.path.join(self.lfp_data_path, 'marco_lfp_spectrum_*.npz'))
            for file in lfp_files:
                first_8_last_8 = np.load(file)['data']
                first_8_last_8 = np.concatenate((first_8_last_8[:8, :], first_8_last_8[-8:, :]), axis=0)
                # first_8_last_8 = first_8_last_8[:8, :]
                lfp_mat = superVstack(lfp_mat, first_8_last_8)
        else:
            fn = os.path.join(self.lfp_data_path, 'marco_lfp_john.npz')
            lfp_mat = np.load(fn)['data']
        return np.array(lfp_mat).astype(np.float32)

    def load_npz_by_chunk(self, hour=1):
        def superVstack(a, b):
            # make it so you can vstack onto empty row
            if len(a) == 0:
                stack = b
            elif len(b) == 0:
                stack = a
            else:
                stack = np.vstack([a, b])
            return stack

        lfp_mat = []
        lfp_files = glob.glob(os.path.join(self.lfp_data_path, 'marco_lfp_spectrum_*_hour_{}.npz'.format(hour)))
        for file in lfp_files:
            first_8_last_8 = np.load(file)['data']
            first_8_last_8 = np.concatenate((first_8_last_8[:8, :], first_8_last_8[-8:, :]), axis=0)
            # first_8_last_8 = first_8_last_8[:8, :]
            lfp_mat = superVstack(lfp_mat, first_8_last_8)
        return np.array(lfp_mat).astype(np.float32)

    def load_pickle(self, fn):
        if not os.path.exists(fn):
            return
        with open(fn, "rb") as f:
            lookup = pickle.load(f)
        return lookup

    def preprocess_data(self):
        if self.use_combined:
            assert self.data['clusterless'].shape[0] == self.data['lfp'].shape[0]
            length = self.data['clusterless'].shape[0]
        else:
            length = self.data.shape[0]
        self.data_length = length
        # self.label = np.array(self.ml_label).transpose()[:length, :].astype(np.float32)
        # self.smoothed_label = np.array(self.smoothed_ml_label).transpose()[:length, :].astype(np.float32)

    def visualization(self):
        combined_bins = np.vstack((self.data, self.labels))
        combined_bins = self.normalize_bins(combined_bins)
        figpath = "./bins.png"

        plt.figure()
        plt.imshow(combined_bins, aspect='auto', interpolation='nearest')
        # plt.plot(np.ones(bins.shape[1])*bins.shape[0]-1.5)
        plt.savefig(figpath)
        plt.show()

    def __len__(self):
        return self.data_length


class MyDataset(Dataset):
    def __init__(self, lfp_data, spike_data, label, indices, transform=None):
        self.lfp_data = None
        self.spike_data = None
        if lfp_data is not None:
            self.lfp_data = lfp_data
        if spike_data is not None:
            self.spike_data = spike_data
        # self.label = np.array(label).astype(np.float32)
        self.transform = transform
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # label = self.label[index]
        idx = self.indices[index]
        if self.lfp_data is not None and self.spike_data is None:
            lfp = self.lfp_data[index]
            return lfp, idx
        elif self.lfp_data is None and self.spike_data is not None:
            spike = self.spike_data[index]
            return spike, idx

        lfp = self.lfp_data[index]
        spike = self.spike_data[index]
        # if self.transform is not None:
        #     neuron_feature = self.transform(neuron_feature)
        # if self.transform:
        #     random_number = random.random()
        #     if random_number < 0.5:
        #         neuron_feature = random_shift(neuron_feature, 2)
        return (lfp, spike), idx


def create_inference_combined_loaders(
        dataset,
        config,
        batch_size=128,
        seed=42,
        batch_sample_num=2048,
        shuffle=False,
):
    num_workers = 1
    pin_memory = False

    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))

    if shuffle:
        # np.random.seed(seed)
        np.random.shuffle(all_indices)

    if config['use_combined']:
        spike_inference = dataset.data['clusterless'][all_indices]
        lfp_inference = dataset.data['lfp'][all_indices]
    else:
        spike_inference = dataset.data[all_indices] if config['use_spike'] else None
        lfp_inference = dataset.data[all_indices] if config['use_lfp'] else None

    # label_inference = dataset.smoothed_label[all_indices]
    label_inference = None

    inference_dataset = MyDataset(lfp_inference, spike_inference, label_inference, all_indices)

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
    return inference_loader