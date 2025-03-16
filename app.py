import streamlit as st
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup  # Extracts website content

# Load model with caching for performance
@st.cache_resource
def load_model():
    return joblib.load("phishing_model.pkl")

model = load_model()

# Get the number of features expected by the model
num_features = model.n_features_in_

# Streamlit UI
st.set_page_config(page_title="Phishing Detection System", page_icon="🔍", layout="wide")

# Sidebar for Input Features
st.sidebar.title("🔹 Enter Website Features")
feature_inputs = [st.sidebar.selectbox(f"Feature {i+1}", [0, 1]) for i in range(num_features)]
user_input = np.array([feature_inputs]).reshape(1, -1)

# Main Page Header
st.markdown("<h1 style='text-align: center;'>🔍 Phishing Detection System</h1>", unsafe_allow_html=True)

# Model Prediction
try:
    prediction = model.predict(user_input)
    probability = model.predict_proba(user_input)[0]

    st.subheader("🔍 Prediction Result")
    if prediction[0] == 1:
        st.error("🚨 The website is **Phishing**!")
    else:
        st.success("✅ The website is **Safe**!")

    # Confidence Score
    st.subheader("📊 Confidence Score")
    fig, ax = plt.subplots()
    ax.bar(["Safe", "Phishing"], [probability[0], probability[1]], color=["green", "red"])
    st.pyplot(fig)

except ValueError as e:
    st.error(f"⚠️ Model Error: {str(e)}")
    st.stop()

# 🚀 VirusTotal API Key
VT_API_KEY = "YOUR_VIRUSTOTAL_API_KEY"  # Replace with your actual API key

# Function to check URL using VirusTotal API
def check_url_virustotal(url):
    vt_url = "https://www.virustotal.com/api/v3/urls"
    headers = {"x-apikey": VT_API_KEY}
    data = {"url": url}

    response = requests.post(vt_url, headers=headers, data=data)

    if response.status_code == 200:
        analysis_id = response.json()["data"]["id"]
        report_url = f"https://www.virustotal.com/api/v3/analyses/{analysis_id}"
        report_response = requests.get(report_url, headers=headers)

        if report_response.status_code == 200:
            results = report_response.json()
            malicious_count = results["data"]["attributes"]["stats"]["malicious"]

            if malicious_count > 0:
                return f"🚨 **This URL is flagged as malicious by {malicious_count} security vendors!**"
            else:
                return "✅ **This URL is safe according to VirusTotal!**"
    
    return "⚠️ Unable to check URL at the moment."

# Function to analyze website content for phishing keywords
def analyze_website_content(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text().lower()

        # Common phishing indicators
        phishing_keywords = ["verify", "login", "bank", "account", "password", "security", "update", "urgent", "restricted"]
        
        matches = [word for word in phishing_keywords if word in text]
        
        return matches
    except:
        return None

# 🌐 URL Check Section
st.subheader("🌐 Check Website URL")
url = st.text_input("Enter a website URL to analyze:")

if st.button("🔍 Analyze Website"):
    if url:
        try:
            # Step 1: Check if website is accessible
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                st.success("✅ Website is accessible!")
            else:
                st.warning(f"⚠️ Website might be down! Status Code: {response.status_code}")

            # Step 2: Check website using VirusTotal
            vt_result = check_url_virustotal(url)
            st.write(vt_result)

            # Step 3: Analyze website content
            phishing_matches = analyze_website_content(url)
            if phishing_matches is None:
                st.warning("⚠️ Unable to extract website content.")
            elif len(phishing_matches) > 0:
                st.error(f"🚨 Warning! Phishing-related words found: {', '.join(phishing_matches)}")
            else:
                st.success("✅ No obvious phishing keywords detected on this website.")
        
        except:
            st.error("🚨 Unable to reach the website!")

# ✅ Success Message
st.markdown("<div style='text-align: center; color: green; font-weight: bold;'>✅ App is running successfully!</div>", unsafe_allow_html=True)
