import streamlit as st
import pandas as pd
import joblib
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mobile Price Classification", layout="wide")

st.title("📱 Mobile Price Classification – Assignment 2")
st.write("Upload test data and choose a model to predict mobile price ranges.")

# Sidebar for model selection
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# File uploader
uploaded_file = st.file_uploader("Upload CSV test data", type="csv")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(test_data.head())

    # Load model
    model_filename = f"model/{model_choice.lower().replace(' ', '_')}.pkl"
    model = joblib.load(model_filename)

    # Predictions (drop target column if present)
    X_test = test_data.drop(columns=["price_range"], errors="ignore")
    predictions = model.predict(X_test)

    st.subheader("Predicted Price Ranges")
    st.write(predictions)

    # Show counts of predictions
    st.bar_chart(pd.Series(predictions).value_counts())

    # Load metrics
    metrics_filename = f"model/{model_choice.lower().replace(' ', '_')}_metrics.json"
    with open(metrics_filename, "r") as f:
        metrics = json.load(f)

    st.subheader("Evaluation Metrics")
    st.json(metrics)

    # Confusion Matrix / Classification Report (if ground truth available)
    if "price_range" in test_data.columns:
        y_true = test_data["price_range"]
        y_pred = predictions

        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
