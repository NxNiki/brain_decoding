# burst_analysis.py

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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

    def calculate_ibi(self) -> Optional[floating]:
        valid_bursts = self.find_bursts()
        if len(valid_bursts) < 2:
            return None
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

    def stage_analysis(self, sleep_score: pd.DataFrame, method: Callable[[], float]) -> ndarray[Any, dtype[Any]]:
        """
        Apply staged analysis (segment-based) to each labeled signal.

        Parameters:
        - method (Callable): Burst analysis method to apply (e.g., calculate_burst_rate).
        - sleep_score (pd.DataFrame): Sleep score DataFrame with segments.

        Returns:
        - Dict[str, np.ndarray]: Dictionary with label as key and segment analysis result as value.
        """
        results = []

        # Perform staged analysis for each signal
        for i, label in enumerate(self.labels):
            signal = self.signals[:, i]
            segment_results = []

            for label, start_index, end_index in sleep_stage_iterator(sleep_score, len(signal), SLEEP_STAGE_THRESH):
                segment_signal = signal[start_index:end_index]
                # Set the signal for the inherited BurstAnalysis class
                self.set_signal(segment_signal, self.sampling_rate)
                # Call the specified burst analysis method
                segment_results.append(method())

            results[label] = np.array(segment_results)

        return np.column_stack(results)
