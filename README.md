# 🏥 AI Healthcare Centre

AI Healthcare Centre is a web-based healthcare management system that combines **Artificial Intelligence, Machine Learning, and healthcare management** in one platform.

## 🚀 Features

- 👤 Patient Management
- 👨‍⚕️ Doctor Management
- 📅 Appointment Management
- 📋 Medical Records
- 💊 Prescription Management
- 🛒 Medicine Delivery & Cart
- 🤖 AI-Based Preliminary Disease Screening
- 📊 Admin Dashboard & Statistics

## 🤖 AI Module

The project uses a **Random Forest Classifier** to provide preliminary respiratory disease screening.

### Input Features
- Age
- Gender
- Fever
- Cough
- Fatigue
- Difficulty Breathing

The model contains **7 disease classes** and is trained on a dataset of **35 records**.

> ⚠️ The AI result is only a preliminary screening result and is **not a medical diagnosis**.

## 🛠️ Technologies

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, Flask
- **Database:** SQLite
- **Machine Learning:** Scikit-learn
- **Algorithm:** Random Forest
- **Data Processing:** Pandas
- **Model Saving:** Joblib

## 📁 Project Structure

```text
AI Healthcare Centre/
├── app.py
├── train_model.py
├── dataset.csv
├── model.pkl
├── healthcare.db
└── index.html
