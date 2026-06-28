"""
Generate Research Paper PDF with embedded figures.
EEG-Based Cognitive State Classification - Cognitive Squad
"""
from fpdf import FPDF
from pathlib import Path

FIGURES_DIR = Path("results/figures")


class ResearchPaperPDF(FPDF):
    """Custom PDF class for the research paper."""

    def __init__(self):
        super().__init__()
        # Use built-in DejaVu font which supports Unicode
        self.add_font("DejaVu", "", "C:/Windows/Fonts/arial.ttf", uni=True)
        self.add_font("DejaVu", "B", "C:/Windows/Fonts/arialbd.ttf", uni=True)
        self.add_font("DejaVu", "I", "C:/Windows/Fonts/ariali.ttf", uni=True)
        self.add_font("DejaVu", "BI", "C:/Windows/Fonts/arialbi.ttf", uni=True)

    def header(self):
        if self.page_no() > 1:
            self.set_font("DejaVu", "I", 8)
            self.cell(0, 5, "EEG-Based Emotion Recognition - Cognitive Squad", align="C")
            self.ln(3)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("DejaVu", "B", 13)
        self.ln(4)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection_title(self, title):
        self.set_font("DejaVu", "B", 11)
        self.ln(3)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def subsubsection_title(self, title):
        self.set_font("DejaVu", "BI", 10)
        self.ln(2)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("DejaVu", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def add_figure(self, image_path, caption, width=170):
        if Path(image_path).exists():
            self.ln(3)
            x = (210 - width) / 2
            self.image(image_path, x=x, w=width)
            self.ln(2)
            self.set_font("DejaVu", "I", 9)
            self.multi_cell(0, 4, caption, align="C")
            self.ln(3)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        # Header
        self.set_font("DejaVu", "B", 9)
        self.set_fill_color(220, 230, 241)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        # Rows
        self.set_font("DejaVu", "", 9)
        for row in rows:
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6, str(val), border=1, align="C")
            self.ln()
        self.ln(3)


def build_pdf():
    pdf = ResearchPaperPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ===================== TITLE PAGE =====================
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("DejaVu", "B", 20)
    pdf.multi_cell(0, 10, "EEG-Based Emotion Recognition\nUsing Comparative Analysis of\nMachine Learning and Deep Learning Models", align="C")
    pdf.ln(12)
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 8, "Cognitive Squad", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 7, "MSc in Information Technology", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Sri Lanka Institute of Information Technology (SLIIT)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Team member table
    col_w = [50, 55, 75]
    table_width = sum(col_w)
    x_start = (210 - table_width) / 2
    pdf.set_font("DejaVu", "B", 10)
    pdf.set_fill_color(200, 210, 230)
    pdf.set_x(x_start)
    pdf.cell(col_w[0], 7, "Student ID", border=1, fill=True, align="C")
    pdf.cell(col_w[1], 7, "Name", border=1, fill=True, align="C")
    pdf.cell(col_w[2], 7, "Email", border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_font("DejaVu", "", 9)
    members = [
        ("MS26906294", "Perera D.S.M.", "MS26906294@my.sliit.lk"),
        ("MS26904214", "Fonseka E.A.R.", "MS26904214@my.sliit.lk"),
        ("MS26917016", "Manduli J.A.L.", "MS26917016@my.sliit.lk"),
        ("MS26917184", "Bandara K.M.H.", "MS26917184@my.sliit.lk"),
    ]
    for sid, name, email in members:
        pdf.set_x(x_start)
        pdf.cell(col_w[0], 6, sid, border=1, align="C")
        pdf.cell(col_w[1], 6, name, border=1, align="C")
        pdf.cell(col_w[2], 6, email, border=1, align="C")
        pdf.ln()

    pdf.ln(15)
    pdf.set_font("DejaVu", "I", 10)
    pdf.cell(0, 6, "Keywords: EEG, Emotion Recognition, Brain-Computer Interface,", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Deep Learning, Machine Learning, Affective Computing", align="C", new_x="LMARGIN", new_y="NEXT")

    # ===================== ABSTRACT =====================
    pdf.add_page()
    pdf.section_title("Abstract")
    pdf.body_text(
        "Emotion recognition from electroencephalogram (EEG) signals has emerged as a critical area of research "
        "in brain-computer interfaces (BCI) and affective computing. This study presents a comprehensive comparative "
        "analysis of six classification models - three traditional machine learning approaches (Support Vector Machine, "
        "Random Forest, and K-Nearest Neighbors) and three deep learning architectures (Multi-Layer Perceptron, "
        "one-dimensional Convolutional Neural Network, and Long Short-Term Memory network) - for classifying emotional "
        "states from EEG brainwave data. Using a publicly available dataset of EEG recordings captured via a consumer-grade "
        "Muse headband during exposure to emotionally evocative stimuli, we classify three emotional states: positive, "
        "neutral, and negative. Our experimental results demonstrate that both the CNN-1D and Random Forest models achieve "
        "the highest classification accuracy of 98.83%, while all six models exceed 93% accuracy. Ten-fold stratified "
        "cross-validation confirms the robustness of these results, with the Random Forest achieving 98.69% (+/-0.81%) "
        "mean accuracy. Feature importance analysis reveals that minimum quantile features and mean signal values are the "
        "most discriminative EEG markers for emotion classification. These findings suggest that carefully engineered features "
        "combined with either traditional or deep learning classifiers can achieve near-perfect emotion recognition from "
        "consumer-grade EEG devices, opening pathways for accessible mental health monitoring and adaptive human-computer "
        "interaction systems."
    )

    # ===================== 1. INTRODUCTION =====================
    pdf.section_title("1. Introduction")
    pdf.body_text(
        "Human emotions play a fundamental role in cognitive processes, decision-making, social interactions, and overall "
        "mental well-being. The ability to accurately recognize and classify emotional states has significant implications "
        "across multiple domains, including mental health diagnostics, adaptive learning environments, human-computer "
        "interaction (HCI), and neurofeedback systems (Picard, 1997; Calvo & D'Mello, 2010)."
    )
    pdf.body_text(
        "Electroencephalography (EEG) has become one of the most widely adopted modalities for emotion recognition due to "
        "its non-invasive nature, high temporal resolution, and direct measurement of neural activity (Jenke et al., 2014). "
        "Unlike facial expression analysis or speech-based approaches, EEG signals capture the underlying neurophysiological "
        "processes associated with emotional experiences, making them less susceptible to voluntary suppression or social "
        "masking of emotions (Alarcao & Fonseca, 2017)."
    )
    pdf.body_text(
        "Recent advances in machine learning and deep learning have significantly improved the accuracy of EEG-based emotion "
        "classification systems. Traditional approaches rely on handcrafted feature extraction \u2014 such as power spectral "
        "density (PSD), differential entropy (DE), and statistical measures \u2014 followed by classifiers like Support Vector "
        "Machines (SVM) and Random Forests (Zheng & Lu, 2015). Deep learning methods, on the other hand, can learn "
        "hierarchical feature representations directly from raw or minimally processed signals, potentially capturing complex "
        "patterns that manual feature engineering might miss (Craik et al., 2019)."
    )
    pdf.body_text(
        "However, a significant gap exists in the literature regarding systematic comparisons between traditional machine "
        "learning and deep learning approaches on the same dataset under identical experimental conditions. Most studies "
        "focus exclusively on either traditional or deep learning methods, making it difficult to draw meaningful conclusions "
        "about their relative strengths and limitations for EEG emotion recognition."
    )
    pdf.body_text(
        "This study addresses this gap by conducting a comprehensive comparative analysis of six classification models "
        "spanning both paradigms. Our contributions are: (1) We evaluate three traditional ML models and three deep learning "
        "architectures on the same EEG emotion dataset under identical conditions. (2) We employ rigorous evaluation "
        "methodology including 10-fold cross-validation. (3) We perform feature importance analysis to identify discriminative "
        "EEG markers. (4) We provide insights into practical trade-offs between model complexity and performance."
    )

    # ===================== 2. LITERATURE REVIEW =====================
    pdf.section_title("2. Literature Review")

    pdf.subsection_title("2.1 EEG and Emotion")
    pdf.body_text(
        "The relationship between brain electrical activity and emotional states has been studied extensively in neuroscience. "
        "EEG signals are typically decomposed into frequency bands \u2014 theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), "
        "and gamma (30-45 Hz) \u2014 each associated with different cognitive and emotional processes (Aftanas et al., 2004). "
        "Research has shown that frontal alpha asymmetry is linked to approach-withdrawal motivation, with greater left frontal "
        "activity associated with positive emotions and right frontal activity with negative emotions (Davidson, 1992). "
        "Emotion models used in EEG research generally follow two paradigms: the dimensional model (valence-arousal) and "
        "the discrete model (distinct emotion categories). The present study adopts a discrete classification approach with "
        "three classes: positive, neutral, and negative emotional states."
    )

    pdf.subsection_title("2.2 Traditional Machine Learning for EEG")
    pdf.body_text(
        "Traditional machine learning approaches for EEG emotion recognition typically involve a two-stage pipeline: feature "
        "extraction followed by classification. Commonly extracted features include time-domain statistics, frequency-domain "
        "features (band power, PSD), and time-frequency features (Jenke et al., 2014). Support Vector Machines have been "
        "widely used due to their effectiveness in high-dimensional feature spaces. Zheng and Lu (2015) achieved 83.99% "
        "accuracy for three-class emotion recognition on the SEED dataset using SVM with differential entropy features. "
        "Random Forests have also shown strong performance due to their robustness to noisy features (Bos, 2006)."
    )

    pdf.subsection_title("2.3 Deep Learning for EEG")
    pdf.body_text(
        "Deep learning has gained significant traction in EEG analysis due to its ability to automatically learn feature "
        "representations. CNNs have been adapted for EEG through architectures like EEGNet (Lawhern et al., 2018). "
        "One-dimensional CNNs have proven effective for processing sequential EEG features. LSTM networks are well-suited "
        "for EEG data due to its temporal nature, with bidirectional LSTMs capturing both forward and backward dependencies "
        "(Li et al., 2018). Hybrid CNN-LSTM architectures leverage CNNs for local feature extraction and LSTMs for temporal "
        "modeling (Yang et al., 2018). MLPs remain competitive when applied to well-engineered feature sets (Subasi, 2007)."
    )

    pdf.subsection_title("2.4 Consumer-Grade EEG Devices")
    pdf.body_text(
        "The emergence of consumer-grade EEG devices such as the Muse headband has democratized EEG research (Krigolson "
        "et al., 2017). While these devices offer fewer channels and lower signal quality compared to clinical-grade systems, "
        "several studies have demonstrated their viability for emotion recognition. Bird et al. (2019) showed that meaningful "
        "emotional patterns can be extracted from just four EEG channels using the Muse headband."
    )

    return pdf


def add_methodology(pdf):
    """Add methodology section."""
    pdf.section_title("3. Methodology")

    pdf.subsection_title("3.1 Dataset Description")
    pdf.body_text(
        "This study utilizes the EEG Brainwave Dataset for Feeling Emotions (Bird et al., 2019), a publicly available "
        "dataset collected using a Muse EEG headband. The dataset comprises EEG recordings from two participants "
        "(one male, one female) during exposure to emotionally evocative stimuli."
    )
    pdf.body_text(
        "Data Collection: The Muse EEG headband with four dry electrodes (TP9, AF7, AF8, TP10) was used at temporal "
        "and frontal regions. Three emotional states were recorded: Positive (evoked by musical numbers and nature "
        "timelapses), Negative (evoked by death scenes from films), and Neutral (resting state). Each state was "
        "recorded for 3 minutes per participant. The original authors applied a sliding window approach to extract "
        "2,548 statistical features including temporal means, standard deviations, FFT coefficients, covariance "
        "matrix elements, and higher-order statistical moments."
    )
    pdf.body_text(
        "The final dataset contains 2,132 samples: Negative (708, 33.2%), Neutral (716, 33.6%), and Positive "
        "(708, 33.2%). The near-equal class distribution eliminates the need for class balancing techniques."
    )
    pdf.add_figure(str(FIGURES_DIR / "class_distribution.png"), "Figure 1: Dataset Class Distribution", width=120)

    pdf.subsection_title("3.2 Data Preprocessing")
    pdf.body_text(
        "The following preprocessing steps were applied: (1) Missing Value Handling: NaN or infinite values were "
        "replaced with zeros. (2) Feature Standardization: Z-score normalization via StandardScaler, fitted on "
        "training data only to prevent data leakage. (3) Data Splitting: 80% training (1,705 samples) and 20% "
        "testing (427 samples) using stratified random sampling."
    )

    pdf.subsection_title("3.3 Model Architectures")

    pdf.subsubsection_title("3.3.1 Support Vector Machine (SVM)")
    pdf.body_text(
        "SVM with RBF kernel (C=10.0, gamma='scale') maps features into higher-dimensional space for linear separation."
    )

    pdf.subsubsection_title("3.3.2 Random Forest")
    pdf.body_text(
        "Ensemble of 300 decision trees (max_depth=30) aggregating predictions from decorrelated trees."
    )

    pdf.subsubsection_title("3.3.3 K-Nearest Neighbors (KNN)")
    pdf.body_text("KNN with k=7 neighbors using Euclidean distance metric.")

    pdf.subsubsection_title("3.3.4 Multi-Layer Perceptron (MLP)")
    pdf.body_text(
        "Four fully connected layers (2548 -> 512 -> 256 -> 128 -> 3) with BatchNorm, ReLU, and Dropout (p=0.4). "
        "Approximately 1.47 million trainable parameters."
    )

    pdf.subsubsection_title("3.3.5 One-Dimensional CNN (CNN-1D)")
    pdf.body_text(
        "Three convolutional blocks (kernel sizes 7, 5, 3) with BatchNorm, ReLU, MaxPool, and Dropout (p=0.4), "
        "followed by adaptive average pooling and FC classifier. Approximately 44,000 parameters."
    )

    pdf.subsubsection_title("3.3.6 LSTM Network")
    pdf.body_text(
        "Bidirectional LSTM (2 layers, hidden_size=128) processing features as sequences of 50 time steps. "
        "Final hidden states concatenated for classification. Approximately 580,000 parameters."
    )

    pdf.subsection_title("3.4 Training Configuration")
    pdf.body_text(
        "All deep learning models: Adam optimizer (lr=0.001, weight_decay=0.0001), Cross-Entropy loss, batch "
        "size 32, max 100 epochs, ReduceLROnPlateau scheduler (factor=0.5, patience=7), early stopping "
        "(patience=15). Hardware: NVIDIA GeForce RTX 3060 (12GB VRAM)."
    )

    pdf.subsection_title("3.5 Evaluation Methodology")
    pdf.body_text(
        "Metrics: Accuracy, weighted F1 Score, confusion matrices, and 10-fold stratified cross-validation "
        "for traditional ML models to assess generalization robustness."
    )
    return pdf


def add_results(pdf):
    """Add results and discussion section."""
    pdf.section_title("4. Results and Discussion")

    pdf.subsection_title("4.1 Overall Classification Performance")
    pdf.body_text(
        "Table 1 presents the classification results for all six models. The CNN-1D and Random Forest achieved "
        "the highest accuracy of 98.83%, followed by MLP and LSTM at 98.36%."
    )

    pdf.set_font("DejaVu", "I", 9)
    pdf.cell(0, 5, "Table 1: Classification Results for EEG Emotion Recognition", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.add_table(
        headers=["Model", "Accuracy (%)", "F1 Score", "Type"],
        rows=[
            ["CNN-1D", "98.83", "0.9883", "Deep Learning"],
            ["Random Forest", "98.83", "0.9883", "Traditional ML"],
            ["MLP", "98.36", "0.9836", "Deep Learning"],
            ["LSTM", "98.36", "0.9836", "Deep Learning"],
            ["SVM (RBF)", "97.66", "0.9765", "Traditional ML"],
            ["KNN (k=7)", "93.44", "0.9335", "Traditional ML"],
        ],
        col_widths=[50, 40, 40, 50],
    )

    pdf.add_figure(str(FIGURES_DIR / "model_comparison_accuracy.png"), "Figure 2: Model Comparison by Accuracy", width=150)

    pdf.subsection_title("4.2 Cross-Validation Results")
    pdf.body_text(
        "Ten-fold stratified cross-validation confirms the robustness of our findings. Random Forest maintains "
        "the highest mean accuracy (98.69%) with the lowest variance (+/-0.81%)."
    )

    pdf.set_font("DejaVu", "I", 9)
    pdf.cell(0, 5, "Table 2: 10-Fold Stratified Cross-Validation Results", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.add_table(
        headers=["Model", "Mean Acc (%)", "Std Dev (%)", "Mean F1"],
        rows=[
            ["Random Forest", "98.69", "+/-0.81", "0.9869"],
            ["SVM (RBF)", "97.94", "+/-0.92", "0.9794"],
            ["KNN (k=7)", "94.18", "+/-1.87", "0.9409"],
        ],
        col_widths=[50, 40, 40, 50],
    )

    pdf.add_figure(str(FIGURES_DIR / "cross_validation_results.png"), "Figure 3: Cross-Validation Results with Error Bars", width=150)
    pdf.add_figure(str(FIGURES_DIR / "cv_boxplot.png"), "Figure 4: Cross-Validation Accuracy Distribution", width=150)

    pdf.subsection_title("4.3 Per-Class Analysis")
    pdf.body_text(
        "The confusion matrix analysis reveals that the Neutral class is consistently the easiest to classify, "
        "achieving near-perfect precision and recall across all models. This is expected, as the neutral resting "
        "state produces distinct EEG patterns that differ markedly from emotionally aroused states. The primary "
        "source of misclassification occurs between the Positive and Negative classes, which share similar arousal "
        "levels and differ primarily in valence (Russell, 1980)."
    )
    pdf.add_figure(str(FIGURES_DIR / "all_confusion_matrices.png"), "Figure 5: Confusion Matrices for All Traditional ML Models", width=170)

    pdf.subsection_title("4.4 Feature Importance Analysis")
    pdf.body_text(
        "Feature importance analysis using Random Forest reveals the most discriminative EEG features. The top "
        "features include: (1) Minimum quantile features capturing lower bounds of signal amplitude distributions, "
        "(2) Mean signal values reflecting baseline shifts in neural activity, (3) Covariance matrix elements "
        "capturing inter-channel functional connectivity, (4) Signal derivative features capturing rate of change "
        "in neural activity, and (5) Higher-order statistical moments reflecting non-linear neural dynamics."
    )
    pdf.add_figure(str(FIGURES_DIR / "feature_importance_top30.png"), "Figure 6: Top 30 Most Important EEG Features", width=160)
    pdf.add_figure(str(FIGURES_DIR / "feature_category_importance.png"), "Figure 7: Feature Category Importance", width=140)

    pdf.subsection_title("4.5 Data Visualization")
    pdf.body_text(
        "PCA visualization reveals clear clustering of the three emotional classes. The first two principal "
        "components explain 46.3% of total variance. The Neutral class forms the most distinct cluster, while "
        "Positive and Negative classes show some overlap, consistent with the confusion matrix findings."
    )
    pdf.add_figure(str(FIGURES_DIR / "pca_2d_visualization.png"), "Figure 8: PCA 2D Visualization of EEG Emotion Data", width=140)
    pdf.add_figure(str(FIGURES_DIR / "pca_explained_variance.png"), "Figure 9: PCA Explained Variance", width=140)

    pdf.subsection_title("4.6 Discussion")
    pdf.body_text(
        "Traditional ML vs. Deep Learning: The most striking finding is that Random Forest achieves performance "
        "equal to the best deep learning model (CNN-1D) at 98.83%. This suggests that when high-quality features "
        "are available, the choice between traditional and deep learning may be less critical than feature quality."
    )
    pdf.body_text(
        "Model Complexity vs. Performance: CNN-1D achieves top performance with only 44,000 parameters compared "
        "to MLP's 1.47 million, indicating that architectural efficiency is more important than raw model capacity."
    )
    pdf.body_text(
        "KNN Limitations: The lower KNN performance (93.44%) is attributed to the curse of dimensionality in the "
        "2,548-feature space, where Euclidean distance becomes less meaningful (Beyer et al., 1999)."
    )
    pdf.body_text(
        "Limitations: The dataset contains only two participants, limiting generalizability. Pre-extracted features "
        "prevent evaluation of raw signal learning. The controlled laboratory setting may not reflect real-world "
        "emotional complexity."
    )
    return pdf


def add_ethics_conclusion_references(pdf):
    """Add ethics, conclusion, and references."""

    pdf.section_title("5. Ethical Considerations")
    pdf.body_text(
        "Privacy and Neural Data Protection: EEG signals contain sensitive information about cognitive and emotional "
        "states. Collection, storage, and processing must adhere to strict privacy standards. Unauthorized access "
        "could enable invasive surveillance of mental states (Ienca & Andorno, 2017)."
    )
    pdf.body_text(
        "Informed Consent: Participants must provide fully informed consent. The dataset used in this study was "
        "collected with appropriate ethical oversight. Potential for Misuse: Emotion detection could be misused in "
        "workplace surveillance or manipulative advertising. Bias and Fairness: Limited demographic representation "
        "means models may not perform equitably across populations."
    )

    pdf.section_title("6. Conclusion and Future Work")

    pdf.subsection_title("6.1 Conclusion")
    pdf.body_text(
        "This study presented a comprehensive comparative analysis of six models for EEG-based emotion recognition. "
        "Key findings: (1) CNN-1D and Random Forest achieved the highest accuracy of 98.83%. (2) Cross-validation "
        "confirmed robustness with 98.69% (+/-0.81%) for Random Forest. (3) Feature importance analysis identified "
        "minimum quantile features and mean signal values as the most discriminative markers. (4) PCA revealed clear "
        "class separation. (5) Feature engineering quality is at least as important as model architecture selection."
    )

    pdf.subsection_title("6.2 Future Work")
    pdf.body_text(
        "Future directions include: (1) Evaluation on larger datasets (DEAP, SEED) for cross-subject generalization. "
        "(2) Applying deep learning to raw EEG signals for end-to-end learning. (3) Real-time classification systems "
        "for practical BCI applications. (4) Exploring attention mechanisms and transformer architectures. "
        "(5) Cross-subject transfer learning. (6) Multimodal fusion with ECG, GSR, and facial expressions."
    )

    pdf.section_title("References")
    pdf.set_font("DejaVu", "", 8)
    refs = [
        "Aftanas, L. I., et al. (2004). Analysis of evoked EEG synchronization and desynchronization in conditions of emotional activation. Neuroscience and Behavioral Physiology, 34(8), 859-867.",
        "Alarcao, S. M., & Fonseca, M. J. (2017). Emotions recognition using EEG signals: A survey. IEEE Trans. Affective Computing, 10(3), 374-393.",
        "Beyer, K., et al. (1999). When is 'nearest neighbor' meaningful? In Int. Conf. Database Theory (pp. 217-235). Springer.",
        "Bird, J. J., et al. (2019). A study on mental state classification using EEG-based brain-machine interface. In 9th Int. Conf. Intelligent Systems. IEEE.",
        "Bird, J. J., et al. (2019). Mental emotional sentiment classification with an EEG-based brain-machine interface. In DISP'19. Springer.",
        "Bos, D. O. (2006). EEG-based emotion recognition. The Influence of Visual and Auditory Stimuli, 56(3), 1-17.",
        "Calvo, R. A., & D'Mello, S. (2010). Affect detection: An interdisciplinary review. IEEE Trans. Affective Computing, 1(1), 18-37.",
        "Craik, A., et al. (2019). Deep learning for EEG classification tasks: A review. J. Neural Engineering, 16(3), 031001.",
        "Davidson, R. J. (1992). Anterior cerebral asymmetry and the nature of emotion. Brain and Cognition, 20(1), 125-151.",
        "Ekman, P. (1992). An argument for basic emotions. Cognition & Emotion, 6(3-4), 169-200.",
        "Ienca, M., & Andorno, R. (2017). Towards new human rights in the age of neuroscience. Life Sciences, Society and Policy, 13(1), 1-27.",
        "Jenke, R., et al. (2014). Feature extraction and selection for emotion recognition from EEG. IEEE Trans. Affective Computing, 5(3), 327-339.",
        "Koelstra, S., et al. (2012). DEAP: A database for emotion analysis using physiological signals. IEEE Trans. Affective Computing, 3(1), 18-31.",
        "Krigolson, O. E., et al. (2017). Choosing MUSE: Validation of a low-cost, portable EEG system. Frontiers in Neuroscience, 11, 109.",
        "Lawhern, V. J., et al. (2018). EEGNet: A compact CNN for EEG-based BCIs. J. Neural Engineering, 15(5), 056013.",
        "Li, Y., et al. (2018). A bi-hemisphere domain adversarial neural network for EEG emotion recognition. IEEE Trans. Affective Computing, 12(2), 494-504.",
        "Lotte, F., et al. (2018). A review of classification algorithms for EEG-based BCIs: A 10 year update. J. Neural Engineering, 15(3), 031005.",
        "Petrantonakis, P. C., & Hadjileontiadis, L. J. (2010). Emotion recognition from EEG using higher order crossings. IEEE Trans. IT in Biomedicine, 14(2), 186-197.",
        "Picard, R. W. (1997). Affective Computing. MIT Press.",
        "Ray, W. J., & Cole, H. W. (1985). EEG alpha activity reflects attentional demands. Science, 228(4700), 750-752.",
        "Russell, J. A. (1980). A circumplex model of affect. J. Personality and Social Psychology, 39(6), 1161-1178.",
        "Schirrmeister, R. T., et al. (2017). Deep learning with CNNs for EEG decoding and visualization. Human Brain Mapping, 38(11), 5391-5420.",
        "Subasi, A. (2007). EEG signal classification using wavelet feature extraction. Expert Systems with Applications, 32(4), 1084-1093.",
        "Yang, H., et al. (2018). A multi-column CNN model for emotion recognition from EEG signals. Sensors, 19(21), 4736.",
        "Zheng, W. L., & Lu, B. L. (2015). Investigating critical frequency bands for EEG-based emotion recognition. IEEE Trans. Autonomous Mental Development, 7(3), 162-175.",
    ]
    for ref in refs:
        pdf.multi_cell(0, 4, ref)
        pdf.ln(1)

    return pdf


def main():
    print("Generating PDF...")
    pdf = build_pdf()
    pdf = add_methodology(pdf)
    pdf = add_results(pdf)
    pdf = add_ethics_conclusion_references(pdf)

    output_path = "paper/EEG_Emotion_Recognition_Research_Paper.pdf"
    pdf.output(output_path)
    print(f"PDF saved to: {output_path}")
    print(f"Total pages: {pdf.page_no()}")


if __name__ == "__main__":
    main()
