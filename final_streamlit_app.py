
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# Set page configuration
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("📊 Employee Attrition Prediction System")

# Sidebar instructions
with st.sidebar:
    st.header("📌 Instructions")
    st.markdown("1. Upload a CSV file of employee data.")
    st.markdown("2. Explore the data and visualizations.")
    st.markdown("3. Select a model and run predictions.")
    st.markdown("4. View performance metrics.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload employee attrition dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data successfully uploaded!")

    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head())

    st.markdown("### 📌 Data Summary")
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())
    st.write("**Data Types:**")
    st.write(df.dtypes)
    st.write("**Descriptive Statistics:**")
    st.write(df.describe())

    st.markdown("### 📊 Visualizations")
    if df.select_dtypes(include='object').shape[1] > 0:
        cat_col = st.selectbox("Select a categorical column", df.select_dtypes(include='object').columns)
        st.bar_chart(df[cat_col].value_counts())

    num_col = st.selectbox("Select a numeric column", df.select_dtypes(include='number').columns)
    fig, ax = plt.subplots()
    df[num_col].hist(ax=ax, bins=20)
    st.pyplot(fig)

    st.subheader("⚙️ Data Cleaning & Encoding")
    clean_df = df.copy()
    clean_df.dropna(inplace=True)
    clean_df.reset_index(drop=True, inplace=True)

    categorical_cols = clean_df.select_dtypes(include='object').columns
    clean_df = pd.get_dummies(clean_df, columns=categorical_cols, drop_first=True)

    st.success("✅ Data cleaned and encoded.")
    st.dataframe(clean_df.head())

    if "Attrition_Yes" in clean_df.columns:
        y = clean_df["Attrition_Yes"]
        X = clean_df.drop("Attrition_Yes", axis=1)

        model_option = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader(f"📈 Performance Metrics: {model_option}")
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"**Precision:** {precision_score(y_test, y_pred):.2f}")
        st.write(f"**Recall:** {recall_score(y_test, y_pred):.2f}")

        st.markdown("### 📌 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.markdown("### 🔁 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        st.pyplot(fig_cm)
    else:
        st.error("❌ 'Attrition' column missing or not in expected format. Ensure it's present and labeled as 'Attrition'.")
else:
    st.warning("⬆ Please upload a dataset to begin.")
