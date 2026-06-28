# EEG-Based Emotion Recognition

**Cognitive Squad** | MSc in Information Technology | SLIIT

Comparative analysis of 7 Machine Learning and Deep Learning models for classifying emotional states (Positive, Neutral, Negative) from EEG brainwave signals.

## Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Random Forest | **98.83%** | 0.9883 |
| Transformer | **98.83%** | 0.9883 |
| MLP | 98.36% | 0.9836 |
| CNN-1D | 98.36% | 0.9836 |
| LSTM | 98.36% | 0.9836 |
| SVM | 97.66% | 0.9765 |
| KNN | 93.44% | 0.9335 |

## Setup on a New PC

### Prerequisites
- Python 3.12+ ([Download](https://www.python.org/downloads/))
- NVIDIA GPU with CUDA support (optional, CPU works too)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/Malinga5194/cognitive-squad-eeg-emotion-recognition.git
cd cognitive-squad-eeg-emotion-recognition
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

**Windows:**
```bash
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

**With GPU (NVIDIA):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**Without GPU (CPU only):**
```bash
pip install torch torchvision
pip install -r requirements.txt
```

### Step 5: Download Dataset
1. Go to: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions
2. Download and extract `emotions.csv`
3. Place it in `data/raw/emotions.csv`

### Step 6: Run

**Web Dashboard (recommended for demo):**
```bash
streamlit run app.py
```
Opens in your browser at http://localhost:8501

**Live Demo (terminal):**
```bash
python demo.py
```

**Train All 7 Models:**
```bash
python train_kaggle.py
```

**Run Analysis + Generate Figures:**
```bash
python analyze_results.py
```

## Project Structure

```
├── src/
│   ├── config.py                 # Project settings
│   ├── data/
│   │   ├── load_kaggle_eeg.py    # Dataset loader
│   │   ├── dataset.py            # PyTorch datasets
│   │   └── features.py           # Feature extraction
│   ├── models/
│   │   ├── eegnet.py             # EEGNet CNN
│   │   ├── lstm_model.py         # LSTM model
│   │   ├── cnn_lstm.py           # Hybrid CNN-LSTM
│   │   └── transformer_model.py  # Transformer (Self-Attention)
│   └── utils/
│       ├── trainer.py            # Training engine
│       └── visualization.py      # Plot generation
├── data/raw/                     # Place emotions.csv here
├── results/figures/              # Generated charts
├── app.py                        # Web dashboard (Streamlit)
├── demo.py                       # Interactive terminal demo
├── train_kaggle.py               # Full training pipeline
├── analyze_results.py            # Cross-validation + analysis
└── requirements.txt              # Dependencies
```

## Team

| Name | Student ID | Role |
|------|-----------|------|
| D.S.M. Perera | MS26906294 | Team Lead / System Architect (Transformer, CNN-LSTM, LSTM, Web UI) |
| E.A.R. Fonseka | MS26904214 | Data Engineer (Pipeline, Features, PCA) |
| J.A.L. Manduli | MS26917016 | ML Engineer (SVM, RF, KNN) |
| K.M.H. Bandara | MS26917184 | Testing Engineer (MLP, Cross-Validation, Evaluation) |

## Dataset Citation

Bird, J.J., et al. (2019). "Mental Emotional Sentiment Classification with an EEG-based Brain-machine Interface." DISP'19, Springer.

## System Specifications (Development & Testing Environment)

This system was developed, trained, and tested on the following machine:

| Component | Specification |
|-----------|---------------|
| Operating System | Windows 11 Pro (64-bit, build 26200) |
| Processor (CPU) | AMD Ryzen 5 5600X (6 cores / 12 threads) |
| Memory (RAM) | 32 GB |
| Graphics (GPU) | NVIDIA GeForce RTX 3060, 12 GB VRAM |
| GPU Driver | 595.71 |
| Python | 3.12.10 |
| PyTorch | 2.11.0 (CUDA 12.6 build) |
| CUDA Toolkit | 12.6 |

> Note: A CUDA-capable GPU is optional. All models also run on CPU, though
> deep learning training (Transformer, CNN-1D, LSTM, MLP) is faster on the GPU.

## Copyright

Copyright (c) 2026 D.S.M. Perera (MS26906294), Cognitive Squad.
MSc in Information Technology, Sri Lanka Institute of Information Technology (SLIIT).

This project was developed for academic purposes as part of the Artificial
Intelligence module group project. All rights reserved by the author and team.

