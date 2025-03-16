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

# Streamlit UI Improvements
st.set_page_config(page_title="Phishing Detection System", page_icon="🔍", layout="wide")

# Sidebar for Input Features
st.sidebar.title("🔹 Enter Website Features")
feature1 = st.sidebar.selectbox("Feature 1", [0, 1])
feature2 = st.sidebar.selectbox("Feature 2", [0, 1])

# Main Page Header
st.markdown("<h1 style='text-align: center;'>🔍 Phishing Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>Enter website features to check if it's <span style='color: green;'>safe</span> or <span style='color: red;'>phishing</span>.</h4>", unsafe_allow_html=True)

# Show Sample Dataset
st.subheader("📊 Sample Dataset")
df = pd.DataFrame({
    "having_IP_Address": [45, 49, 49, 49, 49],
    "URL_Length": [49, 49, 48, 48, 48],
    "Shortening_Service": [49, 49, 49, 49, 45],
    "having_At_Symbol": [49, 49, 49, 49, 49],
    "double_slash_redirecting": [45, 49, 49, 49, 49]
})
st.dataframe(df)

# Model Prediction
user_input = np.array([[feature1, feature2]])
prediction = model.predict(user_input)
probability = model.predict_proba(user_input)[0]

# Display Prediction Results
st.subheader("📝 User Input Features")
st.write(f"**Feature 1:** {feature1}, **Feature 2:** {feature2}")

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

# Live URL Check
st.subheader("🌐 Check Website URL")
url = st.text_input("Enter a website URL to analyze:")
if st.button("Check Website"):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            st.success("✅ Website is accessible!")
        else:
            st.warning("⚠️ Website might be down!")
    except:
        st.error("🚨 Unable to reach the website!")

# Success Message
st.markdown("<div style='text-align: center; color: green; font-weight: bold;'>✅ App is running successfully!</div>", unsafe_allow_html=True)
