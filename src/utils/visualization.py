"""
Visualization Utilities
Generates plots for training curves, confusion matrices, and model comparisons.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

from src.config import FIGURES_DIR


def plot_training_history(
    history: dict,
    model_name: str = "Model",
    save: bool = True,
) -> None:
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name} - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(
            FIGURES_DIR / f"{model_name}_training_history.png",
            dpi=150, bbox_inches="tight",
        )
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    model_name: str = "Model",
    save: bool = True,
) -> None:
    """Plot a confusion matrix heatmap."""
    if class_names is None:
        class_names = ["Low", "High"]

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax1,
    )
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title(f"{model_name} - Confusion Matrix (Counts)")

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax2,
    )
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title(f"{model_name} - Confusion Matrix (Normalized)")

    plt.tight_layout()
    if save:
        plt.savefig(
            FIGURES_DIR / f"{model_name}_confusion_matrix.png",
            dpi=150, bbox_inches="tight",
        )
    plt.show()


def plot_model_comparison(
    results: dict[str, dict],
    metric: str = "accuracy",
    save: bool = True,
) -> None:
    """
    Bar chart comparing multiple models.

    Args:
        results: Dict of {model_name: {metric: value, ...}}
        metric: Which metric to compare
    """
    model_names = list(results.keys())
    values = [results[name].get(metric, 0) for name in model_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, values, color=sns.color_palette("viridis", len(model_names)))

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.2%}", ha="center", va="bottom", fontweight="bold",
        )

    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Model Comparison - {metric.capitalize()}")
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(
            FIGURES_DIR / f"model_comparison_{metric}.png",
            dpi=150, bbox_inches="tight",
        )
    plt.show()


def plot_eeg_sample(
    data: np.ndarray,
    channel_names: list[str] | None = None,
    sampling_rate: int = 128,
    title: str = "EEG Signal Sample",
    n_channels_to_show: int = 8,
    save: bool = True,
) -> None:
    """
    Plot a sample of EEG signals.

    Args:
        data: Single trial EEG data of shape (n_channels, n_samples)
        channel_names: Names of EEG channels
        sampling_rate: Sampling rate in Hz
        title: Plot title
        n_channels_to_show: Number of channels to display
    """
    n_channels = min(n_channels_to_show, data.shape[0])
    n_samples = data.shape[1]
    time = np.arange(n_samples) / sampling_rate

    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2 * n_channels), sharex=True)

    for i in range(n_channels):
        ax = axes[i] if n_channels > 1 else axes
        ax.plot(time, data[i, :], linewidth=0.5, color="steelblue")
        label = channel_names[i] if channel_names else f"Ch {i+1}"
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save:
        plt.savefig(
            FIGURES_DIR / "eeg_sample.png",
            dpi=150, bbox_inches="tight",
        )
    plt.show()
