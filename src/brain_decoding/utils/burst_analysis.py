# burst_analysis.py

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from numpy import dtype, floating, ndarray

from brain_decoding.utils.analysis import SLEEP_STAGE_THRESH, sleep_stage_iterator


class BurstAnalysis:
    """
    Base class for burst analysis parameters and methods.
    """

    def __init__(self, threshold: float = 0.5, min_burst_duration: float = 1):
        self.signal = None
        self.sampling_rate = None
        self.threshold = threshold
        self.min_burst_duration = min_burst_duration

    def set_signal(self, signal: np.ndarray, sampling_rate: int):
        """Set the signal and sampling rate for analysis."""
        self.signal = signal
        self.sampling_rate = sampling_rate

    def find_bursts(self) -> List[Tuple[int, int]]:
        if self.signal is None or self.sampling_rate is None:
            raise ValueError("Signal and sampling rate must be set before performing analysis.")

        min_burst_samples = int(self.min_burst_duration * self.sampling_rate)
        burst_samples = self.signal > self.threshold
        burst_onsets = np.where(np.diff(burst_samples.astype(int)) == 1)[0] + 1
        burst_offsets = np.where(np.diff(burst_samples.astype(int)) == -1)[0] + 1

        valid_bursts = [
            (onset, offset)
            for onset, offset in zip(burst_onsets, burst_offsets)
            if (offset - onset) >= min_burst_samples
        ]
        return valid_bursts

    def calculate_burst_rate(self) -> float:
        valid_bursts = self.find_bursts()
        duration = len(self.signal) / self.sampling_rate
        return len(valid_bursts) / duration if duration > 0 else 0.0

    def calculate_burst_durations(self) -> Tuple[float, List[float]]:
        valid_bursts = self.find_bursts()
        burst_durations = [(offset - onset) / self.sampling_rate for onset, offset in valid_bursts]
        avg_burst_duration = np.mean(burst_durations) if burst_durations else 0.0
        return avg_burst_duration, burst_durations

    def calculate_ibi(self) -> Union[float, floating[Any]]:
        valid_bursts = self.find_bursts()
        if len(valid_bursts) < 2:
            return np.nan
        ibis = [
            (valid_bursts[i + 1][0] - valid_bursts[i][1]) / self.sampling_rate for i in range(len(valid_bursts) - 1)
        ]
        return np.mean(ibis)


class ActivationBurstAnalysis(BurstAnalysis):
    """
    Class to manage and apply burst analysis on multiple labeled signals.
    Inherits from BurstAnalysis to reuse burst detection methods.
    """

    def __init__(self, signals: np.ndarray, labels: List[str], sampling_rate: int):
        super().__init__()
        if signals.shape[1] != len(labels):
            raise ValueError("Number of signals must match the number of labels.")

        self.signals = signals
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.window_size = None  # Will be set later as needed

    def sliding_window_analysis(self, method: Callable[[], float]) -> ndarray[Any, dtype[Any]]:
        """
        Apply sliding window analysis to each labeled signal.

        Parameters:
        - method (Callable): Burst analysis method to apply (e.g., calculate_burst_rate).
        - window_size (float): Window size in seconds.

        Returns:
        - np.ndarray: Array where each column corresponds to the sliding window analysis result for each signal.
        """
        if self.window_size is None:
            raise ValueError("window_size must be set before performing analysis.")

        window_samples = int(self.window_size * self.sampling_rate)
        num_windows = self.signals.shape[0] - window_samples + 1
        all_results = []

        # Perform sliding window analysis for each signal
        for i in range(self.signals.shape[1]):
            signal = self.signals[:, i]
            results = []

            for start in range(num_windows):
                end = start + window_samples
                window_signal = signal[start:end]
                # Set the signal for the inherited BurstAnalysis class
                self.set_signal(window_signal, self.sampling_rate)
                # Call the specified burst analysis method
                results.append(method())

            all_results.append(results)

        # Combine all results into a single np.ndarray with each result as a column
        return np.column_stack(all_results)

    def stage_analysis(
        self, sleep_score: pd.DataFrame, method: Callable[[], float]
    ) -> Tuple[ndarray[Any, dtype[Any]], List[str]]:
        """
        Apply staged analysis (segment-based) to each labeled signal.

        Parameters:
        - method (Callable): Burst analysis method to apply (e.g., calculate_burst_rate).
        - sleep_score (pd.DataFrame): Sleep score DataFrame with segments.

        Returns:
        - Dict[str, np.ndarray]: Dictionary with label as key and segment analysis result as value.
        """
        results = []
        stage_labels = []

        # Perform staged analysis for each signal
        for i, label in enumerate(self.labels):
            signal = self.signals[:, i]
            segment_results = []

            for j, (stage, start_index, end_index) in enumerate(
                sleep_stage_iterator(sleep_score, len(signal), SLEEP_STAGE_THRESH)
            ):
                segment_signal = signal[start_index:end_index]
                # Set the signal for the inherited BurstAnalysis class
                self.set_signal(segment_signal, self.sampling_rate)
                # Call the specified burst analysis method
                segment_results.append(method())
                if i == 0:
                    stage_length = (end_index - start_index) / self.sampling_rate
                    stage_label = f"{stage} ({j}-{stage_length:.1f} sec)"
                    stage_labels.append(stage_label)

            results.append(segment_results)

        return np.column_stack(results), stage_labels


def dot_plot(data: np.ndarray, x_tick_labels: list, column_labels: list, save_file_name: str):
    """
    Plots dot plots for the given data array using a color palette.

    Parameters:
    - data (np.ndarray): An n x m array, where n is the number of data points and m is the number of curves to plot.
    - x_tick_labels (list): List of labels for the x-axis ticks (must have n elements).
    - column_labels (list): List of labels for each series of dots (must have m elements).
    - save_file_name (str): File name to save the plot.

    Returns:
    - None: Displays the plot and saves the figure.
    """
    if data.shape[1] != len(column_labels):
        raise ValueError("Number of column labels must match the number of columns in the data array.")
    if data.shape[0] != len(x_tick_labels):
        raise ValueError("Number of x-tick labels must match the number of rows in the data array.")

    palette = sb.color_palette("husl", n_colors=data.shape[1])

    x_values = np.arange(data.shape[0])
    plt.figure(figsize=(10, 6))
    for i in range(data.shape[1]):
        plt.plot(x_values, data[:, i], label=column_labels[i], color=palette[i], alpha=0.6)

    plt.xticks(x_values, x_tick_labels, rotation=45, ha="right")
    plt.ylabel(os.path.basename(save_file_name).replace(".png", ""))
    plt.title("Burst Analysis")
    plt.legend()
    plt.grid(visible=True, linestyle="--", alpha=0.5)

    plt.savefig(save_file_name)
    plt.show()
