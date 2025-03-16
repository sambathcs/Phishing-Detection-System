#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sambathcs/Phishing-Detection-System/blob/main/Phishing_Detection.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 📌 **Step 1: Project Setup**
# 1️⃣ Create a GitHub Repository
# Repo Name: Phishing-Detection-System
# Add a README.md with project details
# Create a directory structure like this:

# Phishing-Detection-System/
# │── datasets/         # Contains phishing URL datasets
# │── models/           # Stores trained models (if using ML)
# │── phishing_detector.py   # Main detection script
# │── requirements.txt  # List of dependencies
# │── README.md         # Project documentation

# 📌 Step 2: Collect Phishing & Legitimate URLs
# ✅ Option 1: Use a Public Dataset
# PhishTank – Live phishing sites
# OpenPhish – Free phishing feeds
# UCI ML Repository – Phishing website dataset

# 📌 Step 3: Load the Phishing Dataset
# Now, let's get some real phishing vs. safe URLs to train our model.
# 
# 🔹 Option 1: Use Public Dataset (Recommended)
# Download the dataset from UCI ML Phishing Websites
# 
# Click "Data Folder"
# Download phishing.csv
# Upload it to Colab:
# Click the folder icon 📂 (left sidebar)
# Click Upload ⬆️
# Select your phishing.csv file
# 🔹 Option 2: Use a Sample Dataset in Code
# If you don’t have a dataset, just create a small one:

# import pandas as pd
# 
# data = {
#     "URL": [
#         "http://paypal-security-verification.com",
#         "https://google.com",
#         "http://secure-bank-login.com",
#         "https://github.com"
#     ],
#     "Phishing": [1, 0, 1, 0]  # 1 = Phishing, 0 = Safe
# }
# 
# df = pd.DataFrame(data)
# print(df.head())

# 📌 Step 4: Extract Features from URLs
# Now we process the URLs to detect patterns (e.g., length, special characters, HTTPS usage, subdomains, etc.).

# 📌 Step 5: Train a Machine Learning Model
# Now, let’s train a Random Forest Classifier to predict phishing URLs.

# 📌 Step 6: Test the Model
# Now, let’s test a new URL to see if it’s phishing or safe.

# 📌 Step 7: Save & Upload to GitHub
# Click "File" → "Save a copy in GitHub"
# Select your repository (Phishing-Detection-System)
# Commit Message: Added ML-based phishing detection system
# Click "OK"

# **📌 Step 1: Convert .ARFF to .CSV**
# ✅ Option 1: Use Python to Convert
# Run this in Google Colab to convert the dataset:

# In[28]:


import pandas as pd
from scipy.io import arff

# Load the ARFF file
data, meta = arff.loadarff('/content/Training Dataset.arff')  # Change path if needed

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv('/content/phishing_dataset.csv', index=False)

# Show first 5 rows
print(df.head())


# ➡️ This will create phishing_dataset.csv that you can use in Python.

# **📌 Step 2: Load Converted CSV in Google Colab**
# After running the conversion, load the dataset:

# In[29]:


df = pd.read_csv('/content/phishing_dataset.csv')
print(df.head())  # Check first few rows


# **📌 Convert Entire DataFrame to Numeric Format**
# Try using pd.to_numeric() instead of applymap():

# In[30]:


# Convert all object (string) columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Verify that all columns are now numeric
print(df.dtypes)


# 🔹 What this does:
# ✔️ Converts all columns from object to int64 or float64
# ✔️ Handles errors (errors='coerce') – If a value can't be converted, it replaces it with NaN.

# **📌Handle Any Remaining NaN Values**
# Check if any values became NaN after conversion:

# In[31]:


print(df.isnull().sum())  # Check for NaN values


# ➡️ If you find NaN values, fill them with 0 or the column mean:

# **📌Convert Byte-Encoded Values to Integers** :
# Now, apply the proper conversion after handling NaN values:

# In[38]:


df = df.apply(lambda x: x.str.decode('utf-8').astype(float) if x.dtype == 'object' else x)


# In[32]:


df = df.fillna(0)  # Replace NaN with 0
# OR
df = df.fillna(df.mean())  # Replace NaN with column mean


# **📌 Update Column Name**s
# If the dataset doesn't have "Phishing", use the correct column name from df.columns output.
# 
# For example, if "class" is the label column (instead of "Phishing"), change this:

# In[36]:


X = df.drop(columns=["having_IP_Address", "Result"])  # Change column names based on your dataset
y = df["Result"]  # "Result" is often used as the phishing label in this dataset


# **📌Fix the Feature Selection**

# In[37]:


# Define features & target variable
X = df.drop(columns=["Result"])  # Drop the "Result" column because it's the target
y = df["Result"]  # Labels (1 = Phishing, -1 = Safe)


# **📌 Step 5: Train a Machine Learning Model**
# Now, let’s train a Random Forest Classifier to predict phishing URLs.

# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define features & target variable
X = df.drop(columns=["Result"])  # Remove non-numeric columns
y = df["Result"]

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")


# 🔍 Why Is the Accuracy 100%?
# If accuracy is too high (100%), it might mean:
# ✅ Dataset is well-structured & model is performing well.
# ❌ Model may be overfitting (memorizing training data instead of generalizing).

# **📌 Next Steps: Improve Your Project**
# Now that your model works, here’s how to make it even better:
# 
# **1️⃣ Test on New Data (Check Real-World Accuracy)**
# Try predicting new phishing URLs to see how well your model generalizes:

# In[41]:


# Test a sample phishing URL
test_url = {
    "having_IP_Address": 1,
    "URL_Length": 60,
    "Shortining_Service": 1,
    "having_At_Symbol": 1,
    "double_slash_redirecting": 0,
    "Prefix_Suffix": 1,
    "having_Sub_Domain": 2,
    "SSLfinal_State": 0,
    "Domain_registeration_length": 0,
    "Favicon": 1,
    "port": 0,
    "HTTPS_token": 1,
    "Request_URL": 1,
    "URL_of_Anchor": 0,
    "Links_in_tags": 0,
    "SFH": 1,
    "Submitting_to_email": 1,
    "Abnormal_URL": 1,
    "Redirect": 0,
    "on_mouseover": 1,
    "RightClick": 1,
    "popUpWidnow": 1,
    "Iframe": 1,
    "age_of_domain": 0,
    "DNSRecord": 0,
    "web_traffic": 1,
    "Page_Rank": 0,
    "Google_Index": 1,
    "Links_pointing_to_page": 1,
    "Statistical_report": 1,
}

import pandas as pd

test_df = pd.DataFrame([test_url])
prediction = model.predict(test_df)[0]

print("🚨 Phishing Detected! 🚨" if prediction == 1 else "✅ Safe Website ✅")

