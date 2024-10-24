import os
from time import sleep
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib.patches import Patch

from brain_decoding.dataloader.patients import Events

# from brain_decoding.dataloader.save_clusterless import SECONDS_PER_HOUR

PREDICTION_FS = 4
SLEEP_SCORE_FS = 1 / 30
SLEEP_SCORE_OFFSET = 0
SECONDS_PER_HOUR = 3600


def prediction_curve(
    predictions: np.ndarray, sleep_score: pd.DataFrame, labels: List[str], save_file_name: str
) -> None:
    """
    Plot prediction curves with background colors representing sleep stages and a legend.

    Parameters:
    - predictions (np.ndarray): n by m array of predictions.
    - sleep_score (pd.DataFrame): n by 2 DataFrame with sleep stage (column 0) and start index (column 1).
    - labels (List[str]): List of labels for each prediction curve.
    - save_file_name (str): The file path to save the plot.

    Returns:
    - None: The function saves the figure to the specified output file.
    """
    # Creating subplots
    palette = sb.color_palette("husl", n_colors=predictions.shape[1])

    y_min = np.min(predictions)
    y_max = np.max(predictions)

    # Assign a unique color for each unique sleep stage
    unique_stages = sleep_score["Score"].unique()
    stage_colors = sb.color_palette("Set2", len(unique_stages))
    stage_color_map = dict(zip(unique_stages, stage_colors))  # Map sleep stages to colors

    fig, axes = plt.subplots(nrows=predictions.shape[1], ncols=1, figsize=(20, 12), sharex=True)

    # Loop through each prediction curve
    for i in range(predictions.shape[1]):
        # Calculate time in hours
        time = np.arange(predictions.shape[0]) / PREDICTION_FS / SECONDS_PER_HOUR

        # Plot the prediction curve with time in hours
        sb.lineplot(
            x=time,
            y=predictions[:, i],
            ax=axes[i],
            color=palette[i],
            linewidth=1.5,
        )
        # Plot the mean curve with a dashed line
        sb.lineplot(
            x=time,
            y=np.mean(predictions[:, i]),
            ax=axes[i],
            color="#808080",
            linestyle="--",
        )

        # Add background color based on sleep_score start_index
        for j in range(len(sleep_score)):
            start = sleep_score.iloc[j]["start_index"] / PREDICTION_FS / SECONDS_PER_HOUR
            end = (
                sleep_score.iloc[j + 1]["start_index"] / PREDICTION_FS / SECONDS_PER_HOUR
                if j < len(sleep_score) - 1
                else predictions.shape[0] / PREDICTION_FS / SECONDS_PER_HOUR
            )

            if 0 <= start < predictions.shape[0] / PREDICTION_FS / SECONDS_PER_HOUR:
                color = stage_color_map[sleep_score.iloc[j]["Score"]]
                axes[i].axvspan(xmin=start, xmax=end, color=color, alpha=0.3)

        # Set y-axis limits and title
        axes[i].set_ylim([y_min, y_max])
        axes[i].set_title(labels[i], fontsize=14)

    # Create custom legend for the background colors
    legend_elements = [Patch(facecolor=stage_color_map[stage], label=stage, alpha=0.3) for stage in unique_stages]
    plt.legend(handles=legend_elements, loc="upper right", title="Sleep Stages")

    # Set a common y-label for the figure
    fig.supylabel("Activation", fontsize=14)
    plt.xlabel("Time (hours)", fontsize=14)
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_file_name)
    plt.show()


def stage_box_plot(predictions: np.ndarray, sleep_score: pd.DataFrame, labels: List[str], save_file_name: str) -> None:
    """
    Plot violin plots with swarms overlaid for each sleep stage, with a separate subplot for each label.
    Limit the number of swarm points per stage for performance improvement and add stage length to the label.

    Parameters:
    - predictions (np.ndarray): n by m array of predictions.
    - sleep_score (pd.DataFrame): n by 2 DataFrame with sleep stage (column 0) and start index (column 1).
    - labels (List[str]): List of labels for each prediction column.
    - save_file_name (str): The file path to save the plot.
    - sampling_rate (int): The sampling rate of the data (default is 4 Hz).

    Returns:
    - None: The function saves the figure with subplots to the specified output file.
    """
    n_samples, n_labels = predictions.shape

    # Create subplots for each label (column of predictions)
    fig, axes = plt.subplots(n_labels, 1, figsize=(12, 3 * n_labels), sharex=True)

    # If there's only one label, we need to convert axes to an iterable
    if n_labels == 1:
        axes = [axes]

    # Loop through each label (column of predictions)
    for i, label in enumerate(labels):
        # Overwrite the combined DataFrame for memory efficiency
        combined_df_list = []
        show_legend = True if i == 0 else False

        for j in range(len(sleep_score)):
            start = int(sleep_score.iloc[j]["start_index"])
            end = int(sleep_score.iloc[j + 1]["start_index"]) if j < len(sleep_score) - 1 else n_samples

            if 0 <= start < predictions.shape[0] and end - start > 600 * PREDICTION_FS:
                stage_data = predictions[start:end, i]
                stage_data = stage_data[stage_data > 0.5]  # Filter values greater than 0.5
                # Calculate stage length (duration in seconds)
                stage_length = (end - start) / PREDICTION_FS
                stage_label = f"Stage: {j} ({stage_length:.1f} sec)"

                # Overwrite combined_df each time to save memory
                combined_df_list.append(
                    pd.DataFrame(
                        {
                            "Stage": [stage_label] * len(stage_data),
                            "Value(>.5)": stage_data,
                            "Label": [label] * len(stage_data),
                            "Stage Label": [sleep_score.iloc[j]["Score"]] * len(stage_data),
                        }
                    )
                )

        combined_df = pd.concat(combined_df_list, axis=0)
        # Sample a maximum of n points per stage for the swarmplot
        combined_df_sample = (
            combined_df.groupby("Stage")
            .apply(lambda x: x.sample(n=min(len(x), 200), random_state=42))
            .reset_index(drop=True)
        )

        # Create a color palette for the stages
        unique_stages = combined_df["Stage Label"].unique()
        palette = sb.color_palette("Set2", len(unique_stages))
        stage_color_map = dict(zip(unique_stages, palette))

        # Plot the violin/box plot for this label on its respective axis
        ax = sb.boxplot(
            x="Stage",
            y="Value(>.5)",
            data=combined_df,
            hue="Stage Label",
            palette=stage_color_map,
            linewidth=1.5,
            color="none",
            width=0.7,
            notch=True,
            ax=axes[i],
            dodge=False,
            legend=False,
        )
        # Overlay the swarmplot with limited points
        ax = sb.swarmplot(
            x="Stage",
            y="Value(>.5)",
            data=combined_df_sample,
            hue="Stage Label",
            palette=stage_color_map,
            size=2,
            dodge=False,
            legend=show_legend,
            ax=axes[i],
        )

        if show_legend:
            c = ax.collections
            ax.legend(
                borderaxespad=0.0,
                loc="right",
                columnspacing=1.2,
                frameon=False,
                markerscale=5,
                handlelength=0.1,
                prop={"size": 10},
                title="",
                bbox_to_anchor=(1, 1.1),
                ncol=2,
            )

        # change boxplot edge color:
        for i, artist in enumerate(ax.patches):
            # Set the linecolor on the artist to the facecolor, and set the facecolor to None
            col = artist.get_facecolor()
            artist.set_edgecolor(col)
            artist.set_facecolor("None")

            # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
            # Loop over them here, and use the same colour as above
            for j in range(i * 6, i * 6 + 6):
                line = ax.lines[j]
                line.set_color(col)
                line.set_mfc(col)
                line.set_mec(col)

        # sb.violinplot(x='Stage', y='Value(>.5)', data=combined_df, hue='Stage Label', palette=stage_color_map,
        #            linewidth=1.5, facecolor="none", ax=axes[i], inner=None, dodge=False, legend=False)

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Set the title for each subplot
        ax.set_ylabel(label, fontsize=12)
        ax.tick_params(axis="x", rotation=45)

    # Add overall figure label
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_file_name)
    plt.show()


def correlation_heatmap(data: np.ndarray, column_labels: List[str], output_filename: str) -> None:
    """
    Calculate the correlation among the columns of the data array and plot a heatmap with the
    distribution of correlation values in a subplot.

    Parameters:
    - data (np.ndarray): n by m array where n is the number of samples and m is the number of columns.
    - column_labels (List[str]): A list of labels for each column.
    - output_filename (str): The file path to save the heatmap image.

    Returns:
    - None: The function saves the figure to the specified output file.
    """
    # Calculate the correlation matrix
    corr_matrix = np.corrcoef(data, rowvar=False)
    v_min, v_max = -1, 1

    # Flatten the correlation matrix and exclude the diagonal (correlation of a variable with itself)
    corr_values = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

    # Create a figure with 2 subplots: 1 for the heatmap, 1 for the histogram
    fig, (ax_heatmap, ax_hist) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [2.5, 1.5]})

    # Plot the heatmap on the first subplot
    sb.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        xticklabels=column_labels,
        center=0,
        vmin=v_min,
        vmax=v_max,
        cbar=False,
        annot_kws={"size": 12},
        yticklabels=column_labels,
        ax=ax_heatmap,
    )
    ax_heatmap.set_title("Correlation Heatmap")

    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, horizontalalignment="right", fontsize=12)
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0, fontsize=12)

    # Plot the distribution of correlation values on the second subplot
    ax_hist.hist(corr_values, bins=10, color="gray", edgecolor="black")
    ax_hist.set_title("Correlation Value Distribution")
    ax_hist.set_xlabel("Correlation")
    ax_hist.set_ylabel("Frequency")

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
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


def smooth_data(data: np.ndarray[float], window_size: int = 5) -> np.ndarray[float, Any]:
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def smooth_columns(data: np.ndarray[float], window_size: int = 5) -> np.ndarray[float, Any]:
    n_rows = data.shape[0]
    smoothed_data = np.zeros((n_rows - window_size + 1, data.shape[1]))  # Adjust size for smoothing

    # Smoothing each column
    for i in range(data.shape[1]):
        smoothed_data[:, i] = smooth_data(data[:, i], window_size=window_size)

    return smoothed_data


def combine_continuous_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine rows with continuous same values in the 'Score' column and keep the first value in the 'start_index' column.

    Parameters:
    - df (pd.DataFrame): A DataFrame with 'Score' and 'start_index' columns.

    Returns:
    - pd.DataFrame: A new DataFrame with combined rows, keeping the first 'start_index' value for each group.
    """

    # Create a mask to identify where the 'Score' changes
    df["group"] = (df["Score"] != df["Score"].shift()).cumsum()

    # Group by the 'group' column and aggregate 'Score' and 'start_index'
    combined_df = df.groupby("group").agg({"Score": "first", "start_index": "first"}).reset_index(drop=True)

    # Drop the temporary 'group' column if necessary
    combined_df = combined_df[["Score", "start_index"]]

    return combined_df


def read_sleep_score(filename: str) -> pd.DataFrame:
    sleep_score = pd.read_csv(filename, header=0)
    print(
        f"shape of sleep_score: {sleep_score.shape}, "
        f"duration: {sleep_score.shape[0] / SLEEP_SCORE_FS / SECONDS_PER_HOUR} hours"
    )
    sleep_score["start_index"] = [
        int(i * PREDICTION_FS / SLEEP_SCORE_FS + SLEEP_SCORE_OFFSET) for i in range(sleep_score.shape[0])
    ]
    sleep_score = combine_continuous_scores(sleep_score)

    print(f"shape of sleep_score after merge: {sleep_score.shape}")

    return sleep_score
