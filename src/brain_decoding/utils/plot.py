import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

from brain_decoding.dataloader.patients import Events


def prediction_curve(predictions: np.ndarray[float], labels: List[str], save_file_name: str) -> None:
    # Creating subplots
    palette = sb.color_palette("husl", n_colors=predictions.shape[1])

    y_min = np.min(predictions)
    y_max = np.max(predictions)

    fig, axes = plt.subplots(nrows=predictions.shape[1], ncols=1, figsize=(20, 12), sharex=True)
    for i in range(predictions.shape[1]):
        sb.lineplot(
            x=np.arange(predictions.shape[0]),
            y=predictions[:, i],
            ax=axes[i],
            color=palette[i],
        )
        axes[i].set_ylim([y_min, y_max])
        axes[i].set_title(labels[i])

    plt.ylabel("Activation")
    plt.xlabel("Time")
    plt.tight_layout()

    plt.savefig(save_file_name)


plt.show()


def prediction_heatmap(predictions: np.ndarray[float], events: Events, title: str, file_path: str):
    fig, ax = plt.subplots(figsize=(4, 8))
    heatmap = ax.imshow(predictions, cmap="viridis", aspect="auto", interpolation="none")

    for concept_i, concept_vocalizations in enumerate(events):
        if not len(concept_vocalizations) > 0:
            continue
        for concept_vocalization in concept_vocalizations:
            t = 4 * concept_vocalization / 1000
            ax.axhline(
                y=t,
                color="red",
                linestyle="-",
                alpha=0.6,
                xmin=concept_i / len(events),
                xmax=(concept_i + 1) / len(events),
            )

    cbar = plt.colorbar(heatmap)
    cbar.ax.tick_params(labelsize=10)
    tick_positions = np.arange(0, len(predictions), 15 * 4)  # 15 seconds * 100 samples per second
    tick_labels = [int(pos * 0.25) for pos in tick_positions]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(np.arange(0, predictions.shape[1], 1))
    ax.set_xticklabels(events.events_name, rotation=80)

    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Concept")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()
