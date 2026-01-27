# Customer Churn Prediction using ANN 🧠📊

This project focuses on predicting **customer churn** in the banking domain using an  
**Artificial Neural Network (ANN)**.  
The goal is to identify customers who are likely to leave the bank so that proactive
retention strategies can be applied.


🔗 GitHub Repo: [Link]()

🌐 Live Demo: [Link](https://churn-rate-prediction-using-ann-8kzvnrusn3glfebxcmzzqw.streamlit.app/)

## 🔍 Problem Statement
Customer churn is a major challenge in the banking industry.  
Acquiring a new customer is far more expensive than retaining an existing one.

This project helps answer:
> *Which customers are most likely to leave the bank in the near future?*

---

## 🛠️ Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-FA0F00?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

---

## 📂 Dataset
The dataset contains customer information such as:
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Active Membership Status
- Estimated Salary

**Target Variable:**  
- `Exited = 1` → Customer churned  
- `Exited = 0` → Customer retained  

---

## ⚙️ Project Workflow

1. **Data Cleaning & Preprocessing**
   - Removed irrelevant columns
   - Encoded categorical features
   - Applied feature scaling

2. **Handling Imbalanced Data**
   - Observed class imbalance in churned vs non-churned customers
   - Applied **SMOTE (Synthetic Minority Oversampling Technique)**

3. **Model Building**
   - Built an Artificial Neural Network using TensorFlow/Keras
   - Used sigmoid activation for binary classification

4. **Model Evaluation**
   - Confusion Matrix
   - Precision, Recall, F1-Score
   - Accuracy

---

## 🚀 How to Run the Project

### Create virtual environment
```
conda create -n ml_env python=3.10
```

### Install environment
```
conda activate ml_env
```

### Installdependencies
```
pip install -r requirements.txt
```

### Run Jupyter Notebook
```
python -m jupyter notebook
```
Open the notebook churn-rate-prediction-using-ann.ipynb and run all cells.

---

## 📈 Output
The model predicts whether a customer is likely to:

Stay with the bank (0)
Leave the bank (1)

This can help banks design targeted retention strategies.

---

## 📄 License
This project is licensed under the MIT License.

---

## 👤 Piyush
Piyush Kumar


