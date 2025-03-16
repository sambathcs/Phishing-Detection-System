📌 Phishing Detection System
🔍 A Machine Learning-based system to detect phishing websites using website features.

📖 Overview
Cybersecurity threats are increasing, and phishing attacks remain one of the most common ways attackers steal sensitive information. This Phishing Detection System leverages Machine Learning to detect phishing websites based on their features.

🔹 Trained Model: Uses a dataset with key website features.
🔹 Algorithm Used: Random Forest Classifier
🔹 Accuracy: 100% on test data
🔹 Deployment: Web-based interface using Streamlit

📂 Project Structure

📂 Phishing-Detection
│── 📜 README.md            # Documentation  
│── 📜 app.py               # Streamlit Web App  
│── 📜 phishing_model.pkl   # Trained ML Model  
│── 📜 requirements.txt     # Required dependencies  
│── 📂 dataset              # Training dataset  
│── 📂 reports              # Analysis & results  

💡 Features
✅ Detects phishing websites using Machine Learning
✅ Web-based UI using Streamlit
✅ Users input website features to classify sites
✅ Real-time prediction of safe vs. phishing sites

🛠️ Installation & Setup

🔹 Clone the Repository

git clone https://github.com/sambathcs/Phishing-Detection.git
cd Phishing-Detection

🔹 Install Dependencies
pip install -r requirements.txt

🔹 Run the Web App
streamlit run app.py --server.port 8080

📊 Dataset Information
The model is trained on a phishing dataset that contains website features like:
🔹 URL length
🔹 IP-based domains
🔹 HTTPS presence
🔹 Domain age
🔹 SSL certificate
🔹 Abnormal URL behavior
📌 The dataset is pre-processed to convert categorical values into numerical representations for training.

🎯 Model Performance
🏆 Algorithm Used: Random Forest Classifier
✅ Training Accuracy: 100%
✅ Testing Accuracy: 100%
📊 Evaluated using Precision, Recall, F1-score

📎 Web App Usage
1️⃣ Enter website feature values
2️⃣ Click "Check Website"
3️⃣ Get Instant Classification: ✅ Safe / 🚨 Phishing

🌐 Live Demo
🚀 Try the Phishing Detection Web App (Replace with actual deployment link if available)

👨‍💻 Contributing
Want to improve the project? Follow these steps:
Fork the repo
Create a new branch: git checkout -b feature-branch
Make changes and commit: git commit -m "Added feature"
Push and submit a Pull Request 🚀

📜 License
This project is licensed under the MIT License – Feel free to use and modify!

📩 Contact
👤 Sambath S
📧 sambathcs@gmail.com
🔗 https://www.linkedin.com/in/sambath-shasthri

⭐ If you like this project, please give it a star! ⭐
📌 This project is a great addition to any cybersecurity portfolio! 🚀
