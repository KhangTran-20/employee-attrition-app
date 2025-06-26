import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import io

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("📊 Employee Attrition Prediction System")

# Help
with st.expander("ℹ️ Help - How to Use This App", expanded=False):
    st.markdown("""
    1. Upload your employee CSV file
    2. Clean and explore data
    3. Select a machine learning model
    4. View predictions, metrics, visualizations
    5. Download performance report
    """)

# Upload CSV
uploaded_file = st.file_uploader("Upload your employee dataset (CSV file)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Data Preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Dataset Summary
    st.subheader("📈 Dataset Summary")
    st.write("**Data Types:**")
    st.write(df.dtypes)

    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

    st.write("**Statistical Summary:**")
    st.write(df.describe())

    # Visualizations
    st.subheader("📊 Exploratory Visualizations")
    if df.select_dtypes(include='object').shape[1] > 0:
        cat_col = st.selectbox("Categorical column to plot", df.select_dtypes(include='object').columns)
        st.bar_chart(df[cat_col].value_counts())

    if df.select_dtypes(include='number').shape[1] > 0:
        num_col = st.selectbox("Numeric column to plot", df.select_dtypes(include='number').columns)
        fig, ax = plt.subplots()
        df[num_col].hist(bins=20, ax=ax)
        st.pyplot(fig)

    # Data Cleaning
    st.subheader("🧹 Data Cleaning")
    df_clean = df.copy()
    df_clean.dropna(axis=1, thresh=int(len(df)*0.5), inplace=True)
    df_clean.fillna(method='ffill', inplace=True)

    if "Attrition" in df_clean.columns:
        df_clean["Attrition"] = df_clean["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)

    df_clean = pd.get_dummies(df_clean)

    st.success("✅ Data cleaned and encoded.")
    st.dataframe(df_clean.head())

    # Modeling
    st.subheader("🤖 Model Training and Evaluation")
    if "Attrition" not in df_clean.columns:
        st.error("The dataset must contain an 'Attrition' column.")
    else:
        X = df_clean.drop("Attrition", axis=1)
        y = df_clean["Attrition"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_type = st.selectbox("Choose model:", ["Logistic Regression", "Random Forest"])

        if model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation Metrics
        st.markdown(f"### 🔍 Performance Metrics: {model_type}")
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"**Precision:** {precision_score(y_test, y_pred):.2f}")
        st.write(f"**Recall:** {recall_score(y_test, y_pred):.2f}")

        # Confusion Matrix
        st.markdown("**Confusion Matrix:**")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # Classification Report
        report = classification_report(y_test, y_pred, target_names=["Stayed", "Left"])
        st.subheader("📄 Classification Report")
        st.text(report)

        # Report Generation and Download
        st.subheader("📥 Download Report")
        report_io = io.StringIO()
        report_io.write("Employee Attrition Prediction Report\n")
        report_io.write("-----------------------------------\n")
        report_io.write(f"Model: {model_type}\n")
        report_io.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
        report_io.write(f"Precision: {precision_score(y_test, y_pred):.2f}\n")
        report_io.write(f"Recall: {recall_score(y_test, y_pred):.2f}\n\n")
        report_io.write("Classification Report:\n")
        report_io.write(report)

        report_bytes = report_io.getvalue().encode()

        st.download_button(
            label="📥 Download Report as TXT",
            data=report_bytes,
            file_name="attrition_model_report.txt",
            mime="text/plain"
        )
else:
    st.info("👈 Please upload a CSV file to begin.")
