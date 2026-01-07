import streamlit as st
import pickle
import numpy as np

st.title("Customer Churn Prediction App")

st.write("Enter customer details to predict churn")

credit_score = st.number_input("Credit Score", 300, 900)
age = st.number_input("Age", 18, 100)
balance = st.number_input("Balance", 0.0)
salary = st.number_input("Estimated Salary", 0.0)

if st.button("Predict"):
    st.success("Prediction logic will run here")
