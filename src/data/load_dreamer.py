# Copyright (c) 2026 D.S.M. Perera (MS26906294), Cognitive Squad.
# MSc in Information Technology, Sri Lanka Institute of Information Technology (SLIIT).
# Developed for the Artificial Intelligence module group project.
# All rights reserved.

"""
DREAMER Dataset Loader
Loads and preprocesses the DREAMER dataset for emotion classification.
Download from: https://zenodo.org/records/546113
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.config import DATA_RAW, DREAMER_CONFIG


def load_dreamer(
    binary_labels: bool = True,
    label_threshold: float = 3.0,
    target_length: int = 7680,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the DREAMER dataset from .mat file.

    The DREAMER dataset uses a 1-5 scale for labels (vs DEAP's 1-9),
    so the default threshold is 3.0 for binary classification.

    Args:
        binary_labels: If True, convert labels to binary (high/low)
        label_threshold: Threshold for binary classification
        target_length: Target number of samples per trial (for padding/truncating)

    Returns:
        all_data: EEG data array of shape (n_total_trials, 14, target_length)
        all_labels: Label array of shape (n_total_trials, 3)
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy is required to load DREAMER dataset")

    mat_file = DATA_RAW / "DREAMER.mat"
    if not mat_file.exists():
        raise FileNotFoundError(
            f"DREAMER.mat not found in {DATA_RAW}\n"
            f"Download from: https://zenodo.org/records/546113\n"
            f"Place DREAMER.mat in: {DATA_RAW}"
        )

    mat_data = loadmat(str(mat_file), simplify_cells=True)
    dreamer = mat_data["DREAMER"]["Data"]

    all_data = []
    all_labels = []

    n_subjects = len(dreamer)
    for s_idx in tqdm(range(n_subjects), desc="Loading DREAMER subjects"):
        subject = dreamer[s_idx]
        n_trials = len(subject["EEG"]["stimuli"])

        for t_idx in range(n_trials):
            eeg = subject["EEG"]["stimuli"][t_idx]

            # eeg shape: (n_samples, 14) - transpose to (14, n_samples)
            if eeg.ndim == 2:
                eeg = eeg.T
            else:
                continue

            # Pad or truncate to target_length
            n_samples = eeg.shape[1]
            if n_samples >= target_length:
                eeg = eeg[:, :target_length]
            else:
                pad_width = target_length - n_samples
                eeg = np.pad(eeg, ((0, 0), (0, pad_width)), mode="edge")

            # Extract labels
            valence = subject["ScoreValence"][t_idx]
            arousal = subject["ScoreArousal"][t_idx]
            dominance = subject["ScoreDominance"][t_idx]

            all_data.append(eeg)
            all_labels.append([valence, arousal, dominance])

    all_data = np.array(all_data, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.float32)

    if binary_labels:
        all_labels = (all_labels >= label_threshold).astype(np.int64)

    print(f"Loaded DREAMER: {all_data.shape[0]} trials, "
          f"{all_data.shape[1]} channels, {all_data.shape[2]} samples")

    return all_data, all_labels
