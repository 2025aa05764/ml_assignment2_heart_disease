import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("🫀 Heart Disease Prediction – ML Models")

# Sidebar configuration
st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox(
    "Select Model",
    ["logistic_regression", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
)

# File handling
test_file_path = "data/heart_test_with_target.csv"
uploaded_data = st.file_uploader("Upload CSV test data", type=["csv"])

with open(test_file_path, "rb") as f:
    sample_csv = f.read()

st.download_button(
    label="⬇️ Download Test CSV",
    data=sample_csv,
    file_name="test_dataset_without_target.csv",
    mime="text/csv"
)

# Main logic
if uploaded_data is not None:
    df = pd.read_csv(uploaded_data)
    if 'target' in df.columns:
        X_test = df.drop('target', axis=1)
        y_test = df['target']
    else:
        X_test = df
        y_test = None

    model = joblib.load(f"model/{selected_model}.pkl")
    predictions = model.predict(X_test)

    # Layout for results
    col_metrics, col_matrix = st.columns([3, 3])

    if y_test is not None:
        with col_metrics:
            st.subheader("Evaluation Metrics")
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            mcc = matthews_corrcoef(y_test, predictions)
            auc = roc_auc_score(y_test, predictions)

            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{accuracy:.3f}")
            m2.metric("Precision", f"{precision:.3f}")

            m3, m4 = st.columns(2)
            m3.metric("Recall", f"{recall:.3f}")
            m4.metric("F1 Score", f"{f1:.3f}")

            m5, m6 = st.columns(2)
            m5.metric("MCC Score", f"{mcc:.3f}")
            m6.metric("AUC Score", f"{auc:.3f}" if auc is not None else "N/A")

            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(y_test, predictions, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"))

        with col_matrix:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, predictions)
            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
