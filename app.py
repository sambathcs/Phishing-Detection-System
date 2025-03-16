import streamlit as st
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt
import numpy as np

# Load model with caching for performance
@st.cache_resource
def load_model():
    return joblib.load("phishing_model.pkl")

model = load_model()

# Get the number of features expected by the model
num_features = model.n_features_in_

# Streamlit UI Improvements
st.set_page_config(page_title="Phishing Detection System", page_icon="🔍", layout="wide")

# Sidebar for Input Features
st.sidebar.title("🔹 Enter Website Features")
feature_inputs = []

for i in range(num_features):
    feature_inputs.append(st.sidebar.selectbox(f"Feature {i+1}", [0, 1]))

# Convert user inputs into NumPy array
user_input = np.array([feature_inputs]).reshape(1, -1)

# Main Page Header
st.markdown("<h1 style='text-align: center;'>🔍 Phishing Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>Enter website features to check if it's <span style='color: green;'>safe</span> or <span style='color: red;'>phishing</span>.</h4>", unsafe_allow_html=True)

# Show Sample Dataset
st.subheader("📊 Sample Dataset")
df = pd.DataFrame({
    "having_IP_Address": [1, 0, 1, 0, 1],
    "URL_Length": [1, 0, 1, 0, 1],
    "Shortening_Service": [0, 1, 0, 1, 0],
    "having_At_Symbol": [1, 0, 1, 0, 1],
    "double_slash_redirecting": [0, 1, 0, 1, 0]
})
st.dataframe(df)

# Model Prediction
try:
    prediction = model.predict(user_input)
    probability = model.predict_proba(user_input)[0]

    # Display Prediction Results
    st.subheader("📝 User Input Features")
    st.write(f"**Entered Features:** {feature_inputs}")

    st.subheader("🔍 Prediction Result")
    if prediction[0] == 1:
        st.error("🚨 The website is **Phishing**!")
    else:
        st.success("✅ The website is **Safe**!")

    # Show Confidence Score
    st.subheader("📊 Confidence Score")
    fig, ax = plt.subplots()
    ax.bar(["Safe", "Phishing"], [probability[0], probability[1]], color=["green", "red"])
    st.pyplot(fig)

except ValueError as e:
    st.error(f"⚠️ Model Error: {str(e)}")
    st.stop()

# Live URL Check
st.subheader("🌐 Check Website URL")
url = st.text_input("Enter a website URL to analyze:")
if st.button("Check Website"):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            st.success("✅ Website is accessible!")
        else:
            st.warning("⚠️ Website might be down! Status Code: " + str(response.status_code))
    except requests.RequestException:
        st.error("🚨 Unable to reach the website!")

# Success Message
st.markdown("<div style='text-align: center; color: green; font-weight: bold;'>✅ App is running successfully!</div>", unsafe_allow_html=True)
