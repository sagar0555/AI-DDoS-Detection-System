import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# 1. SHOW THE UI IMMEDIATELY (Prevent White Page)
st.set_page_config(page_title="AI DDoS Shield")
st.title("🛡️ AI-Based Network Intrusion System")
st.write("Initializing AI Engines... Please wait.")

# 2. LOAD MODELS ONLY AFTER UI IS DRAWN
@st.cache_resource # This saves the models in RAM so they load fast next time
def load_brains():
    try:
        # Load small files first
        rf = joblib.load("models/rf_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return rf, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

with st.spinner("🧠 Loading AI Models into RAM..."):
    rf_model, data_scaler = load_brains()
    if rf_model:
        st.success("✅ AI Engines Online!")

# 3. YOUR APP LOGIC STARTS HERE
st.header("Step 1: Upload Traffic Log")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")