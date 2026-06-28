# Copyright (c) 2026 D.S.M. Perera (MS26906294), Cognitive Squad.
# MSc in Information Technology, Sri Lanka Institute of Information Technology (SLIIT).
# Developed for the Artificial Intelligence module group project.
# All rights reserved.

"""
EEG Feature Extraction
Extracts frequency-domain and time-domain features from EEG signals.
"""
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew

from src.config import EEG_BANDS


def compute_psd_features(
    data: np.ndarray,
    sampling_rate: int = 128,
    nperseg: int = 256,
) -> np.ndarray:
    """
    Compute Power Spectral Density (PSD) features for each EEG band.

    Args:
        data: EEG data of shape (n_trials, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        nperseg: Length of each segment for Welch's method

    Returns:
        features: Band power features of shape (n_trials, n_channels * n_bands)
    """
    n_trials, n_channels, _ = data.shape
    n_bands = len(EEG_BANDS)
    band_powers = np.zeros((n_trials, n_channels, n_bands))

    for trial_idx in range(n_trials):
        for ch_idx in range(n_channels):
            freqs, psd = signal.welch(
                data[trial_idx, ch_idx, :],
                fs=sampling_rate,
                nperseg=min(nperseg, data.shape[2]),
            )

            for band_idx, (band_name, (low, high)) in enumerate(
                EEG_BANDS.items()
            ):
                band_mask = (freqs >= low) & (freqs <= high)
                band_powers[trial_idx, ch_idx, band_idx] = np.mean(
                    psd[band_mask]
                )

    # Flatten: (n_trials, n_channels * n_bands)
    return band_powers.reshape(n_trials, -1)


def compute_statistical_features(data: np.ndarray) -> np.ndarray:
    """
    Compute statistical features from EEG signals.

    Features per channel: mean, std, skewness, kurtosis, max, min, median

    Args:
        data: EEG data of shape (n_trials, n_channels, n_samples)

    Returns:
        features: Statistical features of shape (n_trials, n_channels * 7)
    """
    n_trials, n_channels, _ = data.shape

    features = np.zeros((n_trials, n_channels, 7))
    features[:, :, 0] = np.mean(data, axis=2)
    features[:, :, 1] = np.std(data, axis=2)
    features[:, :, 2] = skew(data, axis=2)
    features[:, :, 3] = kurtosis(data, axis=2)
    features[:, :, 4] = np.max(data, axis=2)
    features[:, :, 5] = np.min(data, axis=2)
    features[:, :, 6] = np.median(data, axis=2)

    return features.reshape(n_trials, -1)


def compute_de_features(
    data: np.ndarray,
    sampling_rate: int = 128,
) -> np.ndarray:
    """
    Compute Differential Entropy (DE) features for each EEG band.
    DE is widely used in EEG emotion recognition research.

    DE = 0.5 * log(2 * pi * e * variance)

    Args:
        data: EEG data of shape (n_trials, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz

    Returns:
        features: DE features of shape (n_trials, n_channels * n_bands)
    """
    n_trials, n_channels, n_samples = data.shape
    n_bands = len(EEG_BANDS)
    de_features = np.zeros((n_trials, n_channels, n_bands))

    for band_idx, (band_name, (low, high)) in enumerate(EEG_BANDS.items()):
        # Design bandpass filter
        nyquist = sampling_rate / 2
        low_norm = low / nyquist
        high_norm = min(high / nyquist, 0.99)

        b, a = signal.butter(4, [low_norm, high_norm], btype="band")

        for trial_idx in range(n_trials):
            for ch_idx in range(n_channels):
                filtered = signal.filtfilt(
                    b, a, data[trial_idx, ch_idx, :]
                )
                variance = np.var(filtered)
                if variance > 0:
                    de_features[trial_idx, ch_idx, band_idx] = 0.5 * np.log(
                        2 * np.pi * np.e * variance
                    )

    return de_features.reshape(n_trials, -1)


def extract_all_features(
    data: np.ndarray,
    sampling_rate: int = 128,
) -> np.ndarray:
    """
    Extract all features (PSD + statistical + DE) and concatenate.

    Args:
        data: EEG data of shape (n_trials, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz

    Returns:
        features: Combined feature array
    """
    print("Extracting PSD features...")
    psd_feats = compute_psd_features(data, sampling_rate)

    print("Extracting statistical features...")
    stat_feats = compute_statistical_features(data)

    print("Extracting DE features...")
    de_feats = compute_de_features(data, sampling_rate)

    features = np.concatenate([psd_feats, stat_feats, de_feats], axis=1)
    print(f"Total features per trial: {features.shape[1]}")

    return features
