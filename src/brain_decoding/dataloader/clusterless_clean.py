import glob
import os
import re
from typing import List, Union

import numpy as np
import pandas as pd


class OpThresh:
    # usage: op_thresh = OpThresh('>', 5)
    operator = ""
    threshold = 0

    def __init__(self, operator: str, threshold: int):
        self.operator = operator
        self.threshold = threshold


def sort_file_name(filenames: str) -> List[Union[int, str]]:
    """Extract the numeric part of the filename and use it as the sort key"""
    return [int(x) if x.isdigit() else x for x in re.findall(r"\d+|\D+", filenames)]


def find_true_indices(mask, op_thresh: OpThresh = None):
    """
    Returns an nx3 matrix containing start, end, and length of all true samples in a 1D boolean mask.
    All inds are 0-indexed, and all values are given in samples.

    Parameters:
    mask: 1D boolean array.
    op_thresh: class containing string threshold operation ('>', '<', '<=', '>=', '==')
               and int threshold for length of true segment.
    Returns:
    out (numpy.ndarray): nx3 matrix with start, end, and length of true segments.
    """

    mask = mask.astype(int)  # convert to 0s and 1s for diff to work

    edges = np.diff(mask)

    edge_up = np.where(edges == 1)[0] + 1
    edge_down = np.where(edges == -1)[0]

    if mask[0] == 1:  # if first value is high, add to edge_up
        edge_up = np.concatenate(([0], edge_up))

    if mask[-1] == 1:  # if last value is high, add to edge_down
        edge_down = np.concatenate((edge_down, [len(mask) - 1]))

    out = np.zeros((len(edge_up), 3), dtype=int)
    out[:, :2] = np.column_stack((edge_up, edge_down))
    out[:, 2] = out[:, 1] - out[:, 0] + 1

    zero_length_inds = out[:, 2] == 0
    out[zero_length_inds, 2] = 1

    def filter_by_operation(indices, op, thresh):
        if op == ">":
            filtered_indices = indices[indices[:, 2] > thresh]
        elif op == "<":
            filtered_indices = indices[indices[:, 2] < thresh]
        elif op == "<=":
            filtered_indices = indices[indices[:, 2] <= thresh]
        elif op == ">=":
            filtered_indices = indices[indices[:, 2] >= thresh]
        elif op == "==":
            filtered_indices = indices[indices[:, 2] == thresh]
        else:
            raise ValueError("Invalid operation specified")
        return filtered_indices

    if op_thresh != None:
        op = op_thresh.operator
        thresh = op_thresh.threshold
        filtered_indices = filter_by_operation(out, op, thresh)
        return filtered_indices

    return out


def cross_chan_event_detection(clu_df, time_bin_ms, chan_number_cooccur):
    # sort by time
    clu_df = clu_df.sort_values(by="ts_rel_i_sec")
    clu_df = clu_df.reset_index(drop=True)

    # find the time difference between events
    event_dt = np.diff(clu_df["ts_rel_i_sec"])
    close_events = event_dt < time_bin_ms / 1000  # convert to seconds

    # find potential co-occurring events
    potential_cooccur = find_true_indices(close_events, OpThresh(">", chan_number_cooccur))

    # loop over each co-occurance and count how many unique channels are involved
    cooccur = []
    n_chans = []
    channel_data = clu_df["channel"].values

    for i in range(len(potential_cooccur)):
        # cooccur_chan = clu_df.loc[np.arange(potential_cooccur[i, 0], potential_cooccur[i, 1])]
        # unique_chan = len(np.unique(cooccur_chan["channel"]))
        start = potential_cooccur[i, 0]
        end = potential_cooccur[i, 1]
        cooccur_chan = channel_data[start:end]
        unique_chan = len(set(cooccur_chan))
        # if there are enough unique channels, add to the list
        if unique_chan > chan_number_cooccur:
            cooccur.append(potential_cooccur[i, 0:2])
            n_chans.append(unique_chan)

    # drop the rows that are co-occurring events
    if len(cooccur) == 0:
        clu_df_clean = clu_df
    else:
        rows_to_drop = []
        for co in cooccur:
            rows_to_drop.append(np.arange(co[0], co[1]))
        rows_to_drop = np.concatenate(rows_to_drop)

        clu_df_clean = clu_df.drop(rows_to_drop)
        clu_df_clean = clu_df_clean.reset_index(drop=True)

    return clu_df_clean


def cross_chan_binned_clean(clu_df, time_bin_ms, chan_number_cooccur):
    time_bin_s = time_bin_ms / 1000
    # sort by time
    clu_df = clu_df.sort_values(by="ts_rel_i_sec")
    clu_df = clu_df.reset_index(drop=True)
    # Optimize data types to save memory
    clu_df["channel"] = clu_df["channel"].astype("int8")
    # Assign each event to a bin
    clu_df["bin"] = (clu_df["ts_rel_i_sec"] // time_bin_s).astype(int)
    # Compute the number of unique channels per bin
    unique_channels_per_bin = clu_df.groupby("bin")["channel"].nunique()
    # Identify bins that meet the unique channel threshold
    valid_bins = unique_channels_per_bin[unique_channels_per_bin <= chan_number_cooccur].index
    # Filter the DataFrame to include only events in valid bins
    chan_data_clean = clu_df[clu_df["bin"].isin(valid_bins)].copy()
    # Drop the 'bin' column
    chan_data_clean.drop(columns=["bin"], inplace=True)
    # If you need to reset the index
    chan_data_clean.reset_index(drop=True, inplace=True)
    return chan_data_clean


def load_data_from_bundle(clu_bundle_filepaths):
    clu_fn_pattern = f"clusterless_[A-Za-z]+-?[A-Za-z]*(\d).csv"  # pattern to extract channel number from file name, included cases where there is a dash in the name

    clu_df = pd.DataFrame()
    for clu_fp in clu_bundle_filepaths:
        # get chan number from file name
        match = re.search(clu_fn_pattern, str(clu_fp))
        chan_num = int(match.group(1))
        # load the data
        csv_data = pd.read_csv(clu_fp)
        # take only negative amplitude events
        neg_amp_inds = csv_data["amplitude"] < 0
        csv_data = csv_data[neg_amp_inds].reset_index(drop=True)
        # add channel number
        csv_data["channel"] = chan_num
        # concatenate to the main dataframe
        clu_df = pd.concat([clu_df, csv_data], ignore_index=True)

    return clu_df


def get_oneshot_clean(patient_number, desired_samplerate, mode, category="recall", phase=None, version="notch"):
    # folder contains the clustless data, I saved the folder downloaded from the drive as '562/clustless_raw'
    spike_path = f"/mnt/SSD2/yyding/Datasets/neuron/spike_data/{patient_number}/raw_{mode}/"
    spike_files = glob.glob(os.path.join(spike_path, "*.csv"))
    spike_files = sorted(spike_files, key=sort_file_name)

    for bundle in range(0, len(spike_files), 8):
        df = load_data_from_bundle(spike_files[bundle : bundle + 8])
        df_clean = cross_chan_event_detection(df, 2, 4)
    print()


if __name__ == "__main__":
    version = "notch CAR-clean"

    print()
    get_oneshot_clean("573", 1000, "movie", category="movie", phase=1, version=version)
    get_oneshot_clean("573", 1000, "presleep", category="recall", phase="FR1", version=version)
    get_oneshot_clean("573", 1000, "presleep", category="recall", phase="CR1", version=version)
    get_oneshot_clean("573", 1000, "postsleep", category="recall", phase="FR2", version=version)
    get_oneshot_clean("573", 1000, "postsleep", category="recall", phase="CR2", version=version)
