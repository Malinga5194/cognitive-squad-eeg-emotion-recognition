"""
DEAP Dataset Loader
Loads and preprocesses the DEAP dataset for emotion classification.
"""
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.config import DATA_RAW, DEAP_CONFIG


def load_single_subject(subject_id: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load data for a single DEAP subject.

    Args:
        subject_id: Subject number (1-32)

    Returns:
        data: EEG data array of shape (n_trials, n_channels, n_samples)
        labels: Label array of shape (n_trials, 4) [valence, arousal, dominance, liking]
    """
    filename = DATA_RAW / f"s{subject_id:02d}.dat"
    if not filename.exists():
        raise FileNotFoundError(
            f"Subject file not found: {filename}\n"
            f"Please download the DEAP preprocessed Python data and place "
            f".dat files in: {DATA_RAW}"
        )

    with open(filename, "rb") as f:
        subject_data = pickle.load(f, encoding="latin1")

    # data shape: (40 trials, 40 channels, 8064 samples)
    # First 32 channels are EEG, last 8 are peripheral signals
    data = subject_data["data"][:, :32, :]  # Keep only EEG channels
    labels = subject_data["labels"]          # (40, 4)

    return data, labels


def remove_baseline(data: np.ndarray, baseline_samples: int = 384) -> np.ndarray:
    """
    Remove the 3-second baseline period from each trial.
    Baseline = 3 seconds * 128 Hz = 384 samples.

    Args:
        data: EEG data of shape (n_trials, n_channels, n_samples)
        baseline_samples: Number of samples in baseline period

    Returns:
        data: EEG data with baseline removed
    """
    return data[:, :, baseline_samples:]


def load_all_subjects(
    subject_ids: list[int] | None = None,
    remove_baseline_period: bool = True,
    binary_labels: bool = True,
    label_threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load data for multiple DEAP subjects.

    Args:
        subject_ids: List of subject IDs to load (default: all 32)
        remove_baseline_period: Whether to remove 3s baseline
        binary_labels: If True, convert labels to binary (high/low)
        label_threshold: Threshold for binary classification (default: 5.0)

    Returns:
        all_data: Combined EEG data
        all_labels: Combined labels
    """
    if subject_ids is None:
        subject_ids = list(range(1, DEAP_CONFIG["n_subjects"] + 1))

    all_data = []
    all_labels = []

    for sid in tqdm(subject_ids, desc="Loading subjects"):
        try:
            data, labels = load_single_subject(sid)

            if remove_baseline_period:
                data = remove_baseline(data)

            all_data.append(data)
            all_labels.append(labels)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

    if not all_data:
        raise RuntimeError("No subject data could be loaded.")

    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if binary_labels:
        # Convert to binary: 1 if >= threshold, 0 otherwise
        all_labels = (all_labels >= label_threshold).astype(np.int64)

    return all_data, all_labels
