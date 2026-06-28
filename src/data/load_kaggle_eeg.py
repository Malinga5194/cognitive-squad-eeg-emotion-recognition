# Copyright (c) 2026 D.S.M. Perera (MS26906294), Cognitive Squad.
# MSc in Information Technology, Sri Lanka Institute of Information Technology (SLIIT).
# Developed for the Artificial Intelligence module group project.
# All rights reserved.

"""
Kaggle EEG Brainwave Dataset Loader
Dataset: "EEG Brainwave Dataset: Feeling Emotions" by Jordan J. Bird
Source: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions

This dataset contains pre-extracted statistical features from EEG signals
recorded using a Muse headband (4 channels: TP9, AF7, AF8, TP10).
3 emotion classes: POSITIVE, NEUTRAL, NEGATIVE
2132 samples, 2548 features per sample.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.config import DATA_RAW


def load_kaggle_eeg(
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> dict:
    """
    Load and prepare the Kaggle EEG Brainwave emotion dataset.

    Args:
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        scale: Whether to standardize features

    Returns:
        Dictionary with train/test splits and metadata
    """
    csv_path = DATA_RAW / "emotions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"emotions.csv not found in {DATA_RAW}\n"
            f"Download from: https://www.kaggle.com/datasets/"
            f"birdy654/eeg-brainwave-dataset-feeling-emotions"
        )

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]-1} features")

    # Separate features and labels
    X = df.drop("label", axis=1).values.astype(np.float32)
    y_text = df["label"].values

    # Encode labels: NEGATIVE=0, NEUTRAL=1, POSITIVE=2
    le = LabelEncoder()
    y = le.fit_transform(y_text).astype(np.int64)
    class_names = list(le.classes_)

    print(f"Classes: {class_names}")
    print(f"Class distribution: {dict(zip(class_names, np.bincount(y)))}")

    # Handle any NaN or infinite values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "class_names": class_names,
        "n_classes": len(class_names),
        "n_features": X_train.shape[1],
        "scaler": scaler,
        "label_encoder": le,
    }
