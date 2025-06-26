
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Set the title of the app
st.title("Employee Attrition Data Uploader")

# Help section for users
with st.expander("ℹ️ Help - How to Use This App", expanded=False):
    st.markdown("""
    **Welcome to the Employee Attrition Predictor App!**

    1. Upload or paste a CSV link
    2. Explore and clean data
    3. Choose model (Random Forest or Logistic Regression)
    4. View predictions and download report
    """)

# Upload CSV file
uploaded_file = st.file_uploader("Upload your employee dataset (CSV file)", type="csv")

# Option to load dataset via URL
url = st.text_input("Or enter a URL to a CSV file")

# Load dataset from file or URL
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
elif url:
    try:
        df = pd.read_csv(url)
        st.success("Data loaded from URL!")
    except Exception as e:
        st.error(f"Error loading from URL: {e}")

# Only proceed if data is loaded
if df is not None:
    # Preview the uploaded dataset
    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    # Show data types, missing values, and summary statistics
    st.subheader("Dataset Summary")
    st.markdown("**Data Types:**")
    st.write(df.dtypes)

    st.markdown("**Missing Values:**")
    st.write(df.isnull().sum())

    st.markdown("**Summary Statistics:**")
    st.write(df.describe())

    # Show unique values for categorical features
    st.markdown("**Unique Categorical Values:**")
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        st.write(f"**{col}**: {df[col].nunique()} unique values")

    # Plot bar chart for selected categorical column
    st.subheader("Basic Visualizations")
    plot_col = st.selectbox("Choose a categorical column to plot", df.select_dtypes(include='object').columns)
    st.bar_chart(df[plot_col].value_counts())

    # Plot histogram for selected numerical column
    num_col = st.selectbox("Choose a numerical column to plot", df.select_dtypes(include='number').columns)
    fig, ax = plt.subplots()
    df[num_col].hist(ax=ax, bins=20)
    st.pyplot(fig)

    # Data cleaning section
    st.subheader("Automated Data Cleaning")
    missing_percent = df.isnull().mean()
    drop_cols = missing_percent[missing_percent > 0.5].index
    df_clean = df.drop(columns=drop_cols)
    df_clean.fillna(method='ffill', inplace=True)
    df_clean = pd.get_dummies(df_clean)
    st.success("Data cleaned successfully.")
    st.dataframe(df_clean.head())

    # Model selection section
    st.subheader("Modeling: Predict Attrition")
    model_choice = st.selectbox("Choose a model", ["Random Forest", "Logistic Regression"])

    # Only proceed if Attrition column is present
    if "Attrition" in df.columns:
        # Encode target column
        df_clean["Attrition"] = df["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)
        X = df_clean.drop("Attrition", axis=1)
        y = df_clean["Attrition"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train the selected model
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = LogisticRegression(max_iter=1000)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display model evaluation metrics
        st.markdown(f"**Model Performance: {model_choice}**")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
        st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")

        # Display confusion matrix
        st.markdown("**Confusion Matrix:**")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # Report generation and download section
        st.subheader("Generate and Save Report")
        report = classification_report(y_test, y_pred, target_names=["Stayed", "Left"], output_dict=False)
        st.text("Classification Report:")
        st.text(report)

        if st.button("Download Report as TXT"):
            with open("attrition_model_report.txt", "w") as f:
                f.write("Employee Attrition Prediction Report\n")
                f.write("------------------------------------\n")
                f.write(f"Model: {model_choice}\n")
                f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
                f.write(f"Precision: {precision_score(y_test, y_pred):.2f}\n")
                f.write(f"Recall: {recall_score(y_test, y_pred):.2f}\n\n")
                f.write("Classification Report:\n")
                f.write(report)
            st.success("Report saved as 'attrition_model_report.txt' in your app folder.")
    else:
        st.warning("The 'Attrition' column is missing from the original dataset.")
