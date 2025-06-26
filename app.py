import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Title
st.title("Employee Attrition Prediction App")

# Help Section
with st.expander("ℹ️ Help - How to Use This App", expanded=False):
    st.markdown("""
    1. Upload a CSV file of employee data.
    2. Select a model (Random Forest or Logistic Regression).
    3. View prediction results and model evaluation metrics.
    """)

# Upload section
uploaded_file = st.file_uploader("Upload employee dataset (CSV)", type="csv")

# Load dataset
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # Show preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Cleaning and Encoding")
    # Drop columns with >50% missing values
    missing_percent = df.isnull().mean()
    drop_cols = missing_percent[missing_percent > 0.5].index
    df_clean = df.drop(columns=drop_cols)

    # Forward fill missing values
    df_clean.fillna(method='ffill', inplace=True)

    # Encode categorical variables
    df_encoded = pd.get_dummies(df_clean)

    st.success("Data cleaned and encoded.")
    st.dataframe(df_encoded.head())

    # Model selection
    st.subheader("Choose Prediction Model")
    model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"])

    # Load model and features
    if model_choice == "Random Forest":
        with open("rf_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("rf_features.pkl", "rb") as f:
            feature_cols = pickle.load(f)
    else:
        with open("logistic_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("logistic_features.pkl", "rb") as f:
            feature_cols = pickle.load(f)

    # Check required columns
    missing = [col for col in feature_cols if col not in df_encoded.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        X = df_encoded[feature_cols]
        y_pred = model.predict(X)

        # Prediction output
        st.subheader("Prediction Results")
        st.write(pd.DataFrame({'Prediction': y_pred}).replace({0: 'Stay', 1: 'Leave'}))

        # Optionally compare with actual 'Attrition' if present
        if 'Attrition' in df.columns:
            y_actual = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
            if len(y_actual) == len(y_pred):
                st.markdown("### Model Evaluation")
                st.write(f"**Accuracy:** {accuracy_score(y_actual, y_pred):.2f}")
                st.write(f"**Precision:** {precision_score(y_actual, y_pred):.2f}")
                st.write(f"**Recall:** {recall_score(y_actual, y_pred):.2f}")

                st.markdown("**Confusion Matrix**")
                cm = confusion_matrix(y_actual, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                st.text("Classification Report:")
                st.text(classification_report(y_actual, y_pred, target_names=["Stayed", "Left"]))