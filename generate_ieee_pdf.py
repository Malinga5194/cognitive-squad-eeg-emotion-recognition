"""
Generate IEEE-formatted Research Paper PDF (Single Column).
EEG-Based Cognitive State Classification - Cognitive Squad
"""
from fpdf import FPDF
from pathlib import Path

FIGURES_DIR = Path("results/figures")
LEFT_MARGIN = 20
RIGHT_MARGIN = 20
TOP_MARGIN = 20
PAGE_WIDTH = 210 - LEFT_MARGIN - RIGHT_MARGIN  # 170mm usable


class IEEEPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("Arial", "", "C:/Windows/Fonts/arial.ttf")
        self.add_font("Arial", "B", "C:/Windows/Fonts/arialbd.ttf")
        self.add_font("Arial", "I", "C:/Windows/Fonts/ariali.ttf")
        self.add_font("Arial", "BI", "C:/Windows/Fonts/arialbi.ttf")
        self.set_margins(LEFT_MARGIN, TOP_MARGIN, RIGHT_MARGIN)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, str(self.page_no()), align="C")

    def ieee_title(self, title):
        self.set_font("Arial", "B", 20)
        self.multi_cell(0, 9, title, align="C")
        self.ln(5)

    def add_author_block(self, name, faculty, university, location, email):
        """Single author block in SLIIT style."""
        self.set_font("Arial", "", 11)
        self.cell(0, 5.5, name, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Arial", "I", 10)
        self.cell(0, 5, faculty, align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 5, university, align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 5, location, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Arial", "", 10)
        self.cell(0, 5, email, align="C", new_x="LMARGIN", new_y="NEXT")

    def add_author_row(self, authors, y_pos):
        """Place multiple author blocks side by side at given y position."""
        n = len(authors)
        col_w = PAGE_WIDTH / n
        for i, (name, email) in enumerate(authors):
            x = LEFT_MARGIN + i * col_w
            # Name
            self.set_xy(x, y_pos)
            self.set_font("Arial", "", 10)
            self.cell(col_w, 5, name, align="C")
            # Faculty
            self.set_xy(x, y_pos + 5)
            self.set_font("Arial", "I", 8.5)
            self.cell(col_w, 4.5, "Faculty of Graduate Studies", align="C")
            # University
            self.set_xy(x, y_pos + 9.5)
            self.cell(col_w, 4.5, "Sri Lanka Institute of Information", align="C")
            self.set_xy(x, y_pos + 14)
            self.cell(col_w, 4.5, "Technology", align="C")
            # Location
            self.set_xy(x, y_pos + 18.5)
            self.set_font("Arial", "", 8.5)
            self.cell(col_w, 4.5, "Malabe, Sri Lanka", align="C")
            # Email
            self.set_xy(x, y_pos + 23)
            self.cell(col_w, 4.5, email, align="C")

    def ieee_section(self, number, title):
        self.ln(5)
        heading = f"{number}. {title.upper()}" if number else title.upper()
        self.set_font("Arial", "B", 12)
        self.cell(0, 7, heading, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def ieee_subsection(self, label, title):
        self.ln(3)
        self.set_font("Arial", "I", 10)
        self.cell(0, 6, f"{label} {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 5, text, align="J")
        self.ln(1)

    def add_figure(self, path, caption, width=150):
        if not Path(path).exists():
            return
        self.ln(3)
        x = (210 - width) / 2
        self.image(path, x=x, w=width)
        self.ln(2)
        self.set_font("Arial", "I", 9)
        self.multi_cell(0, 4, caption, align="C")
        self.ln(3)

    def add_table(self, caption, headers, rows, col_widths=None):
        self.ln(2)
        self.set_font("Arial", "I", 9)
        self.multi_cell(0, 4, caption, align="C")
        self.ln(1)
        if col_widths is None:
            col_widths = [PAGE_WIDTH / len(headers)] * len(headers)
        x_start = (210 - sum(col_widths)) / 2
        self.set_font("Arial", "B", 9)
        self.set_fill_color(220, 230, 241)
        self.set_x(x_start)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Arial", "", 9)
        for row in rows:
            self.set_x(x_start)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 5.5, str(val), border=1, align="C")
            self.ln()
        self.ln(2)


def build_pdf():
    pdf = IEEEPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ==================== TITLE PAGE ====================
    pdf.add_page()
    pdf.ln(10)
    pdf.ieee_title("EEG-Based Emotion Recognition Using\nComparative Analysis of Machine Learning\nand Deep Learning Models")

    # Row 1: 2 authors
    y1 = pdf.get_y()
    pdf.add_author_row([
        ("D.S.M. Perera", "ms26906294@my.sliit.lk"),
        ("E.A.R. Fonseka", "ms26904214@my.sliit.lk"),
    ], y1)

    # Row 2: 2 authors (centered)
    y2 = y1 + 32
    offset = PAGE_WIDTH / 6
    authors_row2 = [
        ("J.A.L. Manduli", "ms26917016@my.sliit.lk"),
        ("K.M.H. Bandara", "ms26917184@my.sliit.lk"),
    ]
    col_w = PAGE_WIDTH / 3  # same width as row 1 columns
    for i, (name, email) in enumerate(authors_row2):
        x = LEFT_MARGIN + offset + i * col_w
        pdf.set_xy(x, y2)
        pdf.set_font("Arial", "", 10)
        pdf.cell(col_w, 5, name, align="C")
        pdf.set_xy(x, y2 + 5)
        pdf.set_font("Arial", "I", 8.5)
        pdf.cell(col_w, 4.5, "Faculty of Graduate Studies", align="C")
        pdf.set_xy(x, y2 + 9.5)
        pdf.cell(col_w, 4.5, "Sri Lanka Institute of Information", align="C")
        pdf.set_xy(x, y2 + 14)
        pdf.cell(col_w, 4.5, "Technology", align="C")
        pdf.set_xy(x, y2 + 18.5)
        pdf.set_font("Arial", "", 8.5)
        pdf.cell(col_w, 4.5, "Malabe, Sri Lanka", align="C")
        pdf.set_xy(x, y2 + 23)
        pdf.cell(col_w, 4.5, email, align="C")

    pdf.set_y(y2 + 32)

    # ==================== ABSTRACT ====================
    pdf.ln(5)
    pdf.set_font("Arial", "BI", 10)
    pdf.cell(18, 5, "Abstract")
    pdf.set_font("Arial", "B", 10)
    pdf.cell(3, 5, "-")
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5,
        "Emotion recognition from electroencephalogram (EEG) signals has emerged as a critical area of research "
        "in brain-computer interfaces (BCI) and affective computing. This study presents a comprehensive comparative "
        "analysis of six classification models - three traditional machine learning approaches (Support Vector Machine, "
        "Random Forest, and K-Nearest Neighbors) and three deep learning architectures (Multi-Layer Perceptron, "
        "one-dimensional Convolutional Neural Network, and Long Short-Term Memory network) - for classifying emotional "
        "states from EEG brainwave data. Using a publicly available dataset of EEG recordings captured via a consumer-grade "
        "Muse headband, we classify three emotional states: positive, neutral, and negative. Our results demonstrate that "
        "both the CNN-1D and Random Forest models achieve the highest classification accuracy of 98.83%, while all six "
        "models exceed 93% accuracy. Ten-fold stratified cross-validation confirms robustness, with Random Forest achieving "
        "98.69% (+/-0.81%) mean accuracy. Feature importance analysis reveals that minimum quantile features and mean "
        "signal values are the most discriminative EEG markers for emotion classification.",
        align="J"
    )
    pdf.ln(1)
    pdf.set_font("Arial", "BI", 10)
    pdf.cell(22, 5, "Keywords")
    pdf.set_font("Arial", "B", 10)
    pdf.cell(3, 5, "-")
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(0, 5, "EEG, Emotion Recognition, Brain-Computer Interface, Deep Learning, Machine Learning, Affective Computing")
    pdf.ln(2)

    # ==================== I. INTRODUCTION ====================
    pdf.ieee_section("I", "Introduction")
    pdf.body(
        "Human emotions play a fundamental role in cognitive processes, decision-making, social interactions, and overall "
        "mental well-being. The ability to accurately recognize and classify emotional states has significant implications "
        "across multiple domains, including mental health diagnostics, adaptive learning environments, human-computer "
        "interaction (HCI), and neurofeedback systems [1], [2]."
    )
    pdf.body(
        "Electroencephalography (EEG) has become one of the most widely adopted modalities for emotion recognition due to "
        "its non-invasive nature, high temporal resolution, and direct measurement of neural activity [3]. Unlike facial "
        "expression analysis or speech-based approaches, EEG signals capture the underlying neurophysiological processes "
        "associated with emotional experiences, making them less susceptible to voluntary suppression or social masking "
        "of emotions [4]."
    )
    pdf.body(
        "Recent advances in machine learning and deep learning have significantly improved the accuracy of EEG-based emotion "
        "classification systems. Traditional approaches rely on handcrafted feature extraction such as power spectral "
        "density (PSD), differential entropy (DE), and statistical measures, followed by classifiers like Support Vector "
        "Machines (SVM) and Random Forests [5]. Deep learning methods can learn hierarchical feature representations "
        "directly from raw or minimally processed signals [6]."
    )
    pdf.body(
        "However, a significant gap exists in the literature regarding systematic comparisons between traditional machine "
        "learning and deep learning approaches on the same dataset under identical experimental conditions. This study "
        "addresses this gap with contributions: (1) evaluation of six models under identical conditions, (2) rigorous "
        "10-fold cross-validation, (3) feature importance analysis, and (4) insights into model complexity trade-offs."
    )

    # ==================== II. LITERATURE REVIEW ====================
    pdf.ieee_section("II", "Literature Review")

    pdf.ieee_subsection("A.", "EEG and Emotion")
    pdf.body(
        "EEG signals are decomposed into frequency bands - theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), and "
        "gamma (30-45 Hz) - each associated with different cognitive and emotional processes [7]. Frontal alpha "
        "asymmetry is linked to approach-withdrawal motivation [8]. The present study adopts a discrete classification "
        "approach with three classes: positive, neutral, and negative emotional states."
    )

    pdf.ieee_subsection("B.", "Traditional Machine Learning for EEG")
    pdf.body(
        "Traditional approaches involve feature extraction followed by classification. Zheng and Lu [5] achieved "
        "83.99% accuracy for three-class emotion recognition on the SEED dataset using SVM with differential entropy "
        "features. Random Forests have shown strong performance due to robustness to noisy features [9]."
    )

    pdf.ieee_subsection("C.", "Deep Learning for EEG")
    pdf.body(
        "CNNs have been adapted through architectures like EEGNet [10]. LSTM networks capture temporal dependencies "
        "[11]. Hybrid CNN-LSTM architectures combine local feature extraction with temporal modeling [12]. Consumer-grade "
        "devices like the Muse headband have demonstrated viability for emotion recognition [13], [14]."
    )

    pdf.ieee_subsection("D.", "Summary of Related Work")
    pdf.body(
        "Table I summarizes key related studies in EEG-based emotion recognition, highlighting the datasets, "
        "methods, number of classes, and reported accuracies. Our study contributes a systematic comparison of "
        "six models across both traditional ML and deep learning paradigms on the same dataset."
    )
    pdf.add_table(
        "TABLE I: Summary of Related Work in EEG Emotion Recognition",
        ["Study", "Dataset", "Method", "Classes", "Accuracy"],
        [
            ["Zheng & Lu [5]", "SEED", "SVM + DE", "3", "83.99%"],
            ["Koelstra et al. [18]", "DEAP", "SVM + PSD", "2", "62.00%"],
            ["Li et al. [11]", "SEED", "BiDANN", "3", "92.38%"],
            ["Lawhern et al. [10]", "BCI Comp.", "EEGNet", "4", "82.10%"],
            ["Bird et al. [14]", "Muse EEG", "SVM + Stats", "3", "97.89%"],
            ["Schirrmeister [20]", "BCI Comp.", "Deep CNN", "4", "93.40%"],
            ["This Study", "Muse EEG", "RF / CNN-1D", "3", "98.83%"],
        ],
        col_widths=[35, 25, 30, 18, 22],
    )

    return pdf


def add_methodology(pdf):
    pdf.ieee_section("III", "Methodology")

    pdf.body(
        "This section describes the overall system architecture, dataset, preprocessing pipeline, model "
        "architectures, and evaluation methodology used in this study."
    )
    pdf.add_figure(str(FIGURES_DIR / "system_architecture.png"),
                   "Fig. 1. System architecture of the EEG-based emotion classification pipeline.", width=165)

    pdf.ieee_subsection("A.", "Dataset Description")
    pdf.body(
        "This study utilizes the EEG Brainwave Dataset for Feeling Emotions [14], collected using a Muse EEG headband "
        "with four dry electrodes (TP9, AF7, AF8, TP10). EEG recordings from two participants (one male, one female) "
        "were captured during exposure to emotionally evocative stimuli: film clips for positive/negative emotions and "
        "resting state for neutral. Each state was recorded for 3 minutes per participant. The authors applied a sliding "
        "window approach to extract 2,548 statistical features. The final dataset contains 2,132 samples: Negative "
        "(708, 33.2%), Neutral (716, 33.6%), and Positive (708, 33.2%)."
    )
    pdf.add_figure(str(FIGURES_DIR / "class_distribution.png"),
                   "Fig. 2. Dataset class distribution showing balanced representation.", width=120)

    pdf.ieee_subsection("B.", "Data Preprocessing")
    pdf.body(
        "Preprocessing: (1) NaN/infinite values replaced with zeros, (2) z-score normalization via StandardScaler "
        "fitted on training data only, (3) stratified 80/20 train-test split (1,705 training, 427 testing samples)."
    )

    pdf.ieee_subsection("C.", "Model Architectures")
    pdf.body(
        "Six models were evaluated. Traditional ML: (1) SVM with RBF kernel (C=10.0, gamma='scale'), (2) Random "
        "Forest (300 trees, max_depth=30), (3) KNN (k=7, Euclidean distance). Deep Learning: (4) MLP with four "
        "layers (2548-512-256-128-3, BatchNorm, Dropout p=0.4, ~1.47M parameters), (5) CNN-1D with three "
        "convolutional blocks (kernel sizes 7,5,3, ~44K parameters), (6) Bidirectional LSTM (2 layers, "
        "hidden_size=128, ~580K parameters)."
    )

    pdf.ieee_subsection("D.", "Training Configuration")
    pdf.body(
        "Deep learning models: Adam optimizer (lr=0.001, weight_decay=1e-4), Cross-Entropy loss, batch size 32, "
        "max 100 epochs, ReduceLROnPlateau scheduler, early stopping (patience=15). Hardware: NVIDIA GeForce "
        "RTX 3060 (12GB). Evaluation: accuracy, weighted F1, confusion matrices, 10-fold stratified CV."
    )
    return pdf


def add_results(pdf):
    pdf.ieee_section("IV", "Results and Discussion")

    pdf.ieee_subsection("A.", "Overall Classification Performance")
    pdf.body(
        "Table I presents results for all six models. CNN-1D and Random Forest achieved the highest accuracy "
        "of 98.83%, followed by MLP and LSTM at 98.36%."
    )
    pdf.add_table(
        "TABLE II: Classification Results for EEG Emotion Recognition",
        ["Model", "Accuracy (%)", "F1 Score", "Type"],
        [
            ["CNN-1D", "98.83", "0.9883", "Deep Learning"],
            ["Random Forest", "98.83", "0.9883", "Traditional ML"],
            ["MLP", "98.36", "0.9836", "Deep Learning"],
            ["LSTM", "98.36", "0.9836", "Deep Learning"],
            ["SVM (RBF)", "97.66", "0.9765", "Traditional ML"],
            ["KNN (k=7)", "93.44", "0.9335", "Traditional ML"],
        ],
        col_widths=[42, 35, 30, 42],
    )
    pdf.add_figure(str(FIGURES_DIR / "model_comparison_accuracy.png"),
                   "Fig. 4. Model comparison by classification accuracy.", width=145)

    pdf.ieee_subsection("B.", "Cross-Validation Results")
    pdf.body(
        "Ten-fold stratified cross-validation confirms robustness. Random Forest maintains the highest mean "
        "accuracy (98.69%) with the lowest variance (+/-0.81%)."
    )
    pdf.add_table(
        "TABLE III: 10-Fold Stratified Cross-Validation Results",
        ["Model", "Mean Acc (%)", "Std Dev (%)", "Mean F1"],
        [
            ["Random Forest", "98.69", "+/-0.81", "0.9869"],
            ["SVM (RBF)", "97.94", "+/-0.92", "0.9794"],
            ["KNN (k=7)", "94.18", "+/-1.87", "0.9409"],
        ],
        col_widths=[42, 35, 35, 37],
    )
    pdf.add_figure(str(FIGURES_DIR / "cv_boxplot.png"),
                   "Fig. 5. Cross-validation accuracy distribution across 10 folds.", width=145)

    pdf.ieee_subsection("C.", "Per-Class Analysis")
    pdf.body(
        "Confusion matrix analysis reveals that the Neutral class is consistently easiest to classify. The "
        "primary misclassification occurs between Positive and Negative classes, which share similar arousal "
        "levels and differ primarily in valence [15]. Table IV presents detailed per-class metrics."
    )
    pdf.add_table(
        "TABLE IV: Per-Class Classification Metrics (Best Models)",
        ["Model", "Class", "Precision", "Recall", "F1-Score"],
        [
            ["Random Forest", "NEGATIVE", "0.9790", "0.9859", "0.9825"],
            ["Random Forest", "NEUTRAL", "1.0000", "1.0000", "1.0000"],
            ["Random Forest", "POSITIVE", "0.9858", "0.9789", "0.9823"],
            ["SVM", "NEGATIVE", "0.9586", "0.9789", "0.9686"],
            ["SVM", "NEUTRAL", "0.9931", "1.0000", "0.9965"],
            ["SVM", "POSITIVE", "0.9783", "0.9507", "0.9643"],
        ],
        col_widths=[35, 30, 28, 28, 28],
    )
    pdf.add_figure(str(FIGURES_DIR / "all_confusion_matrices.png"),
                   "Fig. 6. Normalized confusion matrices for SVM, Random Forest, and KNN.", width=170)

    pdf.ieee_subsection("D.", "Feature Importance Analysis")
    pdf.body(
        "Random Forest feature importance reveals the most discriminative EEG features: (1) minimum quantile "
        "features, (2) mean signal values, (3) covariance matrix elements capturing inter-channel connectivity, "
        "(4) signal derivatives, and (5) higher-order statistical moments."
    )
    pdf.add_figure(str(FIGURES_DIR / "feature_importance_top30.png"),
                   "Fig. 7. Top 30 most important EEG features.", width=155)
    pdf.add_figure(str(FIGURES_DIR / "feature_category_importance.png"),
                   "Fig. 8. Feature category importance analysis.", width=135)

    pdf.ieee_subsection("E.", "Data Visualization")
    pdf.body(
        "PCA visualization reveals clear clustering of three emotional classes. The first two principal "
        "components explain 46.3% of total variance. The Neutral class forms the most distinct cluster."
    )
    pdf.add_figure(str(FIGURES_DIR / "pca_2d_visualization.png"),
                   "Fig. 9. PCA visualization of EEG emotion data.", width=135)

    pdf.ieee_subsection("F.", "Discussion")
    pdf.body(
        "The most striking finding is that Random Forest achieves performance equal to CNN-1D at 98.83%, "
        "suggesting that with high-quality features, feature engineering quality matters more than model "
        "choice. CNN-1D achieves top performance with only 44K parameters versus MLP's 1.47M. KNN's lower "
        "performance (93.44%) is attributed to the curse of dimensionality [16]."
    )
    pdf.body(
        "Comparison with Published Results: Our best accuracy of 98.83% compares favorably with prior work "
        "on the same dataset. Bird et al. [14] reported 97.89% using SVM, which our SVM result (97.66%) "
        "closely matches. Our Random Forest and CNN-1D models surpass this benchmark. Compared to other EEG "
        "emotion datasets, our results significantly exceed those on DEAP (62% with SVM [18]) and SEED "
        "(83.99% with SVM [5]), though direct comparison is limited by differences in dataset size, number "
        "of channels, and experimental protocols."
    )
    pdf.body(
        "Traditional ML vs. Deep Learning: On pre-extracted features, traditional ML models perform "
        "comparably to deep learning. This suggests that the feature extraction pipeline captures the "
        "essential discriminative information, leaving less room for deep learning to add value through "
        "automatic feature learning. Deep learning's advantage would likely be more pronounced with raw "
        "EEG signals where manual feature engineering is absent."
    )
    pdf.body(
        "Limitations: (1) Only two participants, limiting cross-subject generalizability. (2) Pre-extracted "
        "features prevent evaluation of end-to-end learning. (3) Controlled laboratory conditions may not "
        "reflect real-world emotional complexity. (4) Consumer-grade device with only 4 channels provides "
        "limited spatial resolution compared to clinical EEG systems."
    )
    return pdf


def add_conclusion_refs(pdf):
    pdf.ieee_section("V", "Ethical Considerations")
    pdf.body(
        "EEG signals contain sensitive cognitive and emotional information requiring strict privacy standards "
        "[17]. Emotion detection technology could be misused in surveillance or advertising. Limited demographic "
        "representation means models may not perform equitably across populations."
    )

    pdf.ieee_section("VI", "Conclusion and Future Work")
    pdf.body(
        "This study presented a comparative analysis of six models for EEG-based emotion recognition. Key "
        "findings: (1) CNN-1D and Random Forest achieved 98.83% accuracy. (2) Cross-validation confirmed "
        "robustness at 98.69% (+/-0.81%). (3) Minimum quantile and mean features are most discriminative. "
        "(4) Feature engineering quality is as important as model architecture selection."
    )
    pdf.body(
        "Future work: evaluation on larger datasets (DEAP, SEED), raw EEG signal processing, real-time "
        "classification, transformer architectures, cross-subject transfer learning, and multimodal fusion."
    )

    pdf.ieee_section("", "References")
    pdf.set_font("Arial", "", 8)
    refs = [
        "[1]  R. W. Picard, Affective Computing. MIT Press, 1997.",
        "[2]  R. A. Calvo and S. D'Mello, \"Affect detection: An interdisciplinary review,\" IEEE Trans. Affective Computing, vol. 1, no. 1, pp. 18-37, 2010.",
        "[3]  R. Jenke, A. Peer, and M. Buss, \"Feature extraction and selection for emotion recognition from EEG,\" IEEE Trans. Affective Computing, vol. 5, no. 3, pp. 327-339, 2014.",
        "[4]  S. M. Alarcao and M. J. Fonseca, \"Emotions recognition using EEG signals: A survey,\" IEEE Trans. Affective Computing, vol. 10, no. 3, pp. 374-393, 2017.",
        "[5]  W. L. Zheng and B. L. Lu, \"Investigating critical frequency bands for EEG-based emotion recognition,\" IEEE Trans. Autonomous Mental Development, vol. 7, no. 3, pp. 162-175, 2015.",
        "[6]  A. Craik, Y. He, and J. L. Contreras-Vidal, \"Deep learning for EEG classification tasks: A review,\" J. Neural Engineering, vol. 16, no. 3, 031001, 2019.",
        "[7]  L. I. Aftanas et al., \"Analysis of evoked EEG synchronization in emotional activation,\" Neuroscience and Behavioral Physiology, vol. 34, no. 8, pp. 859-867, 2004.",
        "[8]  R. J. Davidson, \"Anterior cerebral asymmetry and the nature of emotion,\" Brain and Cognition, vol. 20, no. 1, pp. 125-151, 1992.",
        "[9]  D. O. Bos, \"EEG-based emotion recognition,\" The Influence of Visual and Auditory Stimuli, vol. 56, no. 3, pp. 1-17, 2006.",
        "[10] V. J. Lawhern et al., \"EEGNet: A compact CNN for EEG-based BCIs,\" J. Neural Engineering, vol. 15, no. 5, 056013, 2018.",
        "[11] Y. Li et al., \"A bi-hemisphere domain adversarial neural network for EEG emotion recognition,\" IEEE Trans. Affective Computing, vol. 12, no. 2, pp. 494-504, 2018.",
        "[12] H. Yang, J. Han, and K. Min, \"A multi-column CNN model for emotion recognition from EEG,\" Sensors, vol. 19, no. 21, 4736, 2018.",
        "[13] O. E. Krigolson et al., \"Choosing MUSE: Validation of a low-cost portable EEG system,\" Frontiers in Neuroscience, vol. 11, 109, 2017.",
        "[14] J. J. Bird et al., \"Mental emotional sentiment classification with an EEG-based brain-machine interface,\" in Proc. DISP'19, Springer, 2019.",
        "[15] J. A. Russell, \"A circumplex model of affect,\" J. Personality and Social Psychology, vol. 39, no. 6, pp. 1161-1178, 1980.",
        "[16] K. Beyer et al., \"When is nearest neighbor meaningful?\" in Proc. Int. Conf. Database Theory, Springer, pp. 217-235, 1999.",
        "[17] M. Ienca and R. Andorno, \"Towards new human rights in the age of neuroscience,\" Life Sciences, Society and Policy, vol. 13, no. 1, pp. 1-27, 2017.",
        "[18] S. Koelstra et al., \"DEAP: A database for emotion analysis using physiological signals,\" IEEE Trans. Affective Computing, vol. 3, no. 1, pp. 18-31, 2012.",
        "[19] P. Ekman, \"An argument for basic emotions,\" Cognition and Emotion, vol. 6, no. 3-4, pp. 169-200, 1992.",
        "[20] R. T. Schirrmeister et al., \"Deep learning with CNNs for EEG decoding,\" Human Brain Mapping, vol. 38, no. 11, pp. 5391-5420, 2017.",
    ]
    for ref in refs:
        pdf.multi_cell(0, 4, ref)
        pdf.ln(0.5)
    return pdf


def main():
    print("Generating IEEE-formatted PDF (single column)...")
    pdf = build_pdf()
    pdf = add_methodology(pdf)
    pdf = add_results(pdf)
    pdf = add_conclusion_refs(pdf)

    output_path = "paper/EEG_Emotion_Recognition_IEEE_Format.pdf"
    pdf.output(output_path)
    print(f"PDF saved to: {output_path}")
    print(f"Total pages: {pdf.page_no()}")


if __name__ == "__main__":
    main()
