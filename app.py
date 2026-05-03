"""
EEG Emotion Recognition - Web Dashboard
Cognitive Squad | MSc in Information Technology | SLIIT

Run: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="EEG Emotion Recognition - Cognitive Squad",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1a237e;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the dataset."""
    df = pd.read_csv("data/raw/emotions.csv")
    X = df.drop("label", axis=1).values.astype(np.float32)
    X = np.nan_to_num(X)
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    class_names = list(le.classes_)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, class_names, scaler, le


def train_model(model_name, X_train, y_train):
    """Train selected model."""
    models = {
        "SVM (RBF Kernel)": SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=30, random_state=42, n_jobs=-1),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    }
    model = models[model_name]
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    return model, train_time


def plot_confusion(y_true, y_pred, class_names):
    """Generate confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance for Random Forest."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    ax.barh(range(top_n), importances[indices], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features", fontweight="bold")
    plt.tight_layout()
    return fig


# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<p class="main-header">🧠 EEG Emotion Recognition</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Cognitive Squad | MSc in Information Technology | SLIIT</p>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.image("results/figures/cognitive_squad_logo_v2.png", width=200)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", [
        "📊 Dashboard",
        "🤖 Train & Predict",
        "🔍 Live Prediction",
        "📈 Analysis",
    ])

    # Load data
    X_train, X_test, y_train, y_test, class_names, scaler, le = load_data()
    df = pd.read_csv("data/raw/emotions.csv")
    feature_names = df.drop("label", axis=1).columns.tolist()

    if page == "📊 Dashboard":
        show_dashboard(df, class_names, y_train, y_test)
    elif page == "🤖 Train & Predict":
        show_train(X_train, X_test, y_train, y_test, class_names, feature_names)
    elif page == "🔍 Live Prediction":
        show_live_prediction(X_test, y_test, class_names)
    elif page == "📈 Analysis":
        show_analysis()


def show_dashboard(df, class_names, y_train, y_test):
    """Dashboard page with dataset overview."""
    st.header("Dataset Overview")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", f"{len(df):,}")
    col2.metric("Features", f"{df.shape[1] - 1:,}")
    col3.metric("Classes", len(class_names))
    col4.metric("EEG Channels", "4")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df["label"].value_counts()
        colors = ["#e74c3c", "#3498db", "#2ecc71"]
        bars = ax.bar(counts.index, counts.values, color=colors)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(val), ha="center", fontweight="bold")
        ax.set_ylabel("Samples")
        ax.set_title("Emotion Class Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Dataset Details")
        st.markdown("""
        | Property | Value |
        |----------|-------|
        | **Source** | Kaggle EEG Brainwave Dataset |
        | **Device** | Muse EEG Headband |
        | **Channels** | TP9, AF7, AF8, TP10 |
        | **Subjects** | 2 (1 male, 1 female) |
        | **Emotions** | Positive, Neutral, Negative |
        | **Train Set** | 1,705 samples (80%) |
        | **Test Set** | 427 samples (20%) |
        | **Features** | 2,548 statistical features |
        """)

    st.divider()
    st.subheader("Sample Data (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True)


def show_train(X_train, X_test, y_train, y_test, class_names, feature_names):
    """Train and evaluate models."""
    st.header("Train & Evaluate Models")

    model_name = st.selectbox("Select Model:", [
        "SVM (RBF Kernel)", "Random Forest", "KNN (k=7)"
    ])

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner(f"Training {model_name}..."):
            model, train_time = train_model(model_name, X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        # Results
        st.success(f"Training complete in {train_time:.2f} seconds!")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.2%}")
        col2.metric("F1 Score", f"{f1:.4f}")
        col3.metric("Training Time", f"{train_time:.2f}s")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Confusion Matrix")
            fig = plot_confusion(y_test, preds, class_names)
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Classification Report")
            report = classification_report(y_test, preds, target_names=class_names)
            st.code(report)

        # Feature importance for Random Forest
        if model_name == "Random Forest":
            st.divider()
            st.subheader("Feature Importance (Top 15)")
            fig = plot_feature_importance(model, feature_names)
            st.pyplot(fig)
            plt.close()

        # Store model in session
        st.session_state["model"] = model
        st.session_state["model_name"] = model_name


def show_live_prediction(X_test, y_test, class_names):
    """Live prediction on random samples."""
    st.header("🔍 Live Prediction")
    st.write("Click the button to predict emotion from a random EEG sample.")

    # Need a trained model
    if "model" not in st.session_state:
        st.warning("Please train a model first in the 'Train & Predict' page.")
        return

    model = st.session_state["model"]
    model_name = st.session_state["model_name"]
    st.info(f"Using model: **{model_name}**")

    col1, col2 = st.columns([1, 2])

    with col1:
        n_samples = st.slider("Number of predictions:", 1, 20, 5)
        if st.button("🎲 Predict Random Samples", type="primary"):
            indices = np.random.choice(len(X_test), n_samples, replace=False)
            results = []
            for idx in indices:
                actual = class_names[y_test[idx]]
                predicted = class_names[model.predict(X_test[idx:idx+1])[0]]
                match = actual == predicted
                results.append({
                    "Actual": actual,
                    "Predicted": predicted,
                    "Correct": "✅" if match else "❌",
                })

            results_df = pd.DataFrame(results)
            correct = sum(1 for r in results if r["Correct"] == "✅")
            st.session_state["live_results"] = results_df
            st.session_state["live_acc"] = correct / len(results)

    with col2:
        if "live_results" in st.session_state:
            st.metric("Live Accuracy", f"{st.session_state['live_acc']:.0%}")
            st.dataframe(st.session_state["live_results"], use_container_width=True, hide_index=True)


def show_analysis():
    """Show pre-generated analysis figures."""
    st.header("📈 Detailed Analysis")

    figures = {
        "Model Comparison (Accuracy)": "model_comparison_accuracy.png",
        "Model Comparison (F1 Score)": "model_comparison_f1.png",
        "Cross-Validation Results": "cross_validation_results.png",
        "Cross-Validation Box Plot": "cv_boxplot.png",
        "All Confusion Matrices": "all_confusion_matrices.png",
        "Feature Importance (Top 30)": "feature_importance_top30.png",
        "Feature Category Importance": "feature_category_importance.png",
        "PCA Visualization": "pca_2d_visualization.png",
        "PCA Explained Variance": "pca_explained_variance.png",
        "System Architecture": "system_architecture.png",
    }

    selected = st.selectbox("Select Figure:", list(figures.keys()))
    fig_path = Path("results/figures") / figures[selected]

    if fig_path.exists():
        st.image(str(fig_path), caption=selected, use_container_width=True)
    else:
        st.warning(f"Figure not found. Run `python analyze_results.py` first.")


if __name__ == "__main__":
    main()
