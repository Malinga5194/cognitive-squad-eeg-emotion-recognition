"""Generate individual video-submission documents (PDF) for each team member.

Each document contains the project/member details, a short summary of the
project output, novelty, and the individual's contribution, plus a clearly
marked field to paste the Google Drive video link before submitting.
"""
from fpdf import FPDF
from pathlib import Path

PROJECT_TITLE = "EEG-Based Emotion Recognition Using Machine Learning and Deep Learning"
GROUP = "Cognitive Squad"
DEGREE = "MSc in Information Technology"
INSTITUTE = "Sri Lanka Institute of Information Technology (SLIIT)"
SUBJECT = "Artificial Intelligence"
REPO = "https://github.com/Malinga5194/cognitive-squad-eeg-emotion-recognition"

members = [
    {
        "name": "D.S.M. Perera",
        "id": "MS26906294",
        "role": "Team Lead / System Architect",
        "pdf": "paper/submission_Perera.pdf",
        "link": "https://mysliit-my.sharepoint.com/:v:/g/personal/ms26906294_my_sliit_lk/IQDobFgEMcFzQaOuuj6aOwTyAQYkcA133_cnh38444io-cw",
        "output": (
            "The project is an EEG-Based Emotion Recognition system that classifies three "
            "emotional states - Positive, Neutral, and Negative - directly from brainwave "
            "signals recorded by a consumer EEG headband. The system compares seven AI models "
            "through an interactive web dashboard and achieves 98.83% classification accuracy."
        ),
        "novelty": (
            "The novelty is the comprehensive comparison of seven models, ranging from "
            "traditional SVM and Random Forest to a state-of-the-art Transformer with "
            "self-attention. A key finding is that a traditional Random Forest matches the "
            "Transformer at 98.83%, showing that feature quality matters more than model complexity."
        ),
        "contribution": (
            "I designed the complete system architecture, deciding the project structure and "
            "frameworks and organising the code into modular components. I implemented three deep "
            "learning models in PyTorch - a Transformer with multi-head self-attention, a hybrid "
            "CNN-LSTM, and a bidirectional LSTM - and built the interactive Streamlit web dashboard "
            "and managed the team GitHub repository."
        ),
    },
    {
        "name": "E.A.R. Fonseka",
        "id": "MS26904214",
        "role": "Data Engineer",
        "pdf": "paper/submission_Fonseka.pdf",
        "output": (
            "The project is an EEG Emotion Recognition system that classifies three emotional "
            "states from brainwave signals with 98.83% accuracy. It is powered by a complete data "
            "pipeline that turns 2,132 raw EEG samples with 2,548 features each into clean, "
            "model-ready inputs for all seven models."
        ),
        "novelty": (
            "The novelty from my work is proving that carefully engineered statistical features "
            "from just four channels of a low-cost consumer EEG device are sufficient for "
            "near-perfect emotion classification, removing the need for expensive 32-channel "
            "clinical equipment."
        ),
        "contribution": (
            "I built the entire data pipeline: the data loading module (missing-value handling, "
            "label encoding, leakage-free z-score normalisation, and stratified splitting), the "
            "feature extraction module (Power Spectral Density, Differential Entropy, and "
            "statistical features), and the PyTorch Dataset classes. I also created the analysis "
            "visualisations - PCA and feature-importance plots - that explain why the models work."
        ),
    },
    {
        "name": "J.A.L. Manduli",
        "id": "MS26917016",
        "role": "ML Engineer",
        "pdf": "paper/submission_Manduli.pdf",
        "output": (
            "The project is an EEG Emotion Recognition system that recognises three emotional "
            "states from brainwave signals with 98.83% accuracy. Users can train the traditional "
            "machine learning models live in the web dashboard and view accuracy, confusion "
            "matrices, and feature importance instantly."
        ),
        "novelty": (
            "The novelty is that my Random Forest model - a traditional algorithm that trains in "
            "about one second - matches the Transformer neural network at 98.83%. This shows that "
            "with properly engineered features, expensive GPU hardware and complex deep learning "
            "are not required for state-of-the-art accuracy."
        ),
        "contribution": (
            "I implemented the three traditional machine learning models: Support Vector Machine "
            "(RBF kernel, tuned C=10), Random Forest (300 trees, depth 30, the best model at "
            "98.83%), and K-Nearest Neighbours (k=7). I also carried out the hyperparameter tuning "
            "to find the optimal configuration for each model."
        ),
    },
    {
        "name": "K.M.H. Bandara",
        "id": "MS26917184",
        "role": "Testing & Evaluation Engineer",
        "pdf": "paper/submission_Bandara.pdf",
        "output": (
            "The project is an EEG-Based Emotion Recognition system that classifies three "
            "emotional states from brainwave signals with 98.83% accuracy. Its results are backed "
            "by rigorous testing - cross-validation and per-class evaluation metrics - that prove "
            "the accuracy is consistent and reproducible."
        ),
        "novelty": (
            "The novelty is the rigorous scientific validation of the system. Rather than relying "
            "on a single accuracy number, the results are proven through 10-fold stratified "
            "cross-validation and per-class confusion-matrix analysis, giving the 98.83% result "
            "genuine scientific credibility."
        ),
        "contribution": (
            "I built the training and evaluation engine (training loop, loss computation, learning "
            "rate scheduling, and early stopping), implemented the MLP neural network model "
            "(98.36% accuracy), and developed the 10-fold stratified cross-validation pipeline that "
            "reports mean accuracy with standard deviation across all folds."
        ),
    },
]


def clean(text):
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text.encode("ascii", "replace").decode("ascii")


NAVY = (20, 30, 97)
BLUE = (21, 101, 192)
GRAY = (85, 85, 85)


def field(pdf, label, value, value_bold=False):
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(*GRAY)
    pdf.cell(42, 7, clean(label))
    pdf.set_font("Arial", "B" if value_bold else "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(138, 7, clean(value))


def section(pdf, heading, body):
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(180, 7, clean(heading))
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(180, 5.8, clean(body))
    pdf.ln(1)


for m in members:
    pdf = FPDF()
    pdf.add_font("Arial", "", "C:/Windows/Fonts/arial.ttf")
    pdf.add_font("Arial", "B", "C:/Windows/Fonts/arialbd.ttf")
    pdf.add_font("Arial", "I", "C:/Windows/Fonts/ariali.ttf")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    # Header band
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 26, "F")
    pdf.set_xy(15, 6)
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(180, 8, "Video Submission", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(15)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(180, 6, clean(GROUP + "  |  " + DEGREE + "  |  SLIIT"))
    pdf.ln(18)

    # Project + member details
    pdf.set_text_color(0, 0, 0)
    field(pdf, "Project Title:", PROJECT_TITLE, value_bold=True)
    field(pdf, "Subject:", SUBJECT)
    field(pdf, "Group:", GROUP)
    field(pdf, "Institute:", INSTITUTE)
    pdf.ln(2)
    field(pdf, "Member Name:", m["name"], value_bold=True)
    field(pdf, "Student ID:", m["id"])
    field(pdf, "Role:", m["role"])
    field(pdf, "Video Duration:", "3 minutes")

    pdf.ln(3)
    # Video link box
    link_url = m.get("link", "")
    pdf.set_draw_color(*BLUE)
    pdf.set_line_width(0.4)
    box_h = 20 if link_url else 16
    y = pdf.get_y()
    pdf.rect(15, y, 180, box_h)
    pdf.set_xy(18, y + 2)
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(*BLUE)
    pdf.cell(174, 6, "Video Link (OneDrive):", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(18)
    if link_url:
        pdf.set_font("Arial", "", 8)
        pdf.set_text_color(21, 101, 192)
        pdf.multi_cell(174, 5, clean(link_url), link=link_url)
    else:
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(174, 6, "[ Paste your shareable cloud-drive link here ]")
    pdf.set_y(y + box_h)
    pdf.ln(4)

    pdf.set_text_color(0, 0, 0)
    section(pdf, "1. Project Output", m["output"])
    section(pdf, "2. Novelty", m["novelty"])
    section(pdf, "3. Individual Contribution", m["contribution"])

    pdf.ln(2)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(*GRAY)
    pdf.multi_cell(180, 5, clean("Project repository: " + REPO))

    pdf.output(m["pdf"])
    print(f"Created: {m['pdf']}")

print("\nAll 4 submission documents generated!")
