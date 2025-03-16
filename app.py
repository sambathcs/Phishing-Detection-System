import os
import pandas as pd
import streamlit as st
from scipy.io import arff

# Streamlit App Title
st.title("🔍 Phishing Detection System")
st.write("Enter website features to check if it's **safe** or **phishing**.")

# Ensure Streamlit correctly loads the dataset
dataset_path = os.path.join(os.path.dirname(__file__), "Training Dataset.arff")

# Load the ARFF file
try:
    data, meta = arff.loadarff(dataset_path)
    df = pd.DataFrame(data)

    # Show the first few rows of the dataset
    st.subheader("📊 Sample Dataset")
    st.write(df.head())

except Exception as e:
    st.error(f"Error loading dataset: {e}")

# Add input fields for user features (Example)
st.sidebar.header("🔹 Enter Website Features")
feature_1 = st.sidebar.selectbox("Feature 1", [-1, 0, 1])
feature_2 = st.sidebar.selectbox("Feature 2", [-1, 0, 1])

# Display User Input
st.write("### User Input Features")
st.write(f"Feature 1: {feature_1}, Feature 2: {feature_2}")

st.success("✅ App is running successfully!")
