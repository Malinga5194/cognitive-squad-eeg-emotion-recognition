"""
Project Configuration
EEG-Based Cognitive State Classification
Cognitive Squad - MSc AI Project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create directories if they don't exist
for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET = "DEAP"  # Options: "DEAP", "DREAMER"

# DEAP dataset settings
DEAP_CONFIG = {
    "n_subjects": 32,
    "n_channels": 32,       # EEG channels only (out of 40 total)
    "n_trials": 40,
    "sampling_rate": 128,   # Hz (preprocessed)
    "trial_duration": 63,   # seconds (3s baseline + 60s trial)
    "baseline_duration": 3, # seconds
    "n_labels": 4,          # valence, arousal, dominance, liking
    "label_names": ["valence", "arousal", "dominance", "liking"],
}

# DREAMER dataset settings
DREAMER_CONFIG = {
    "n_subjects": 23,
    "n_channels": 14,
    "n_trials": 18,
    "sampling_rate": 128,
    "n_labels": 3,          # valence, arousal, dominance
    "label_names": ["valence", "arousal", "dominance"],
}

# Get active dataset config
DATASET_CONFIG = DEAP_CONFIG if DATASET == "DEAP" else DREAMER_CONFIG

# EEG preprocessing settings
EEG_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
