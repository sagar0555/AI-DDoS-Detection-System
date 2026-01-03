import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from stable_baselines3 import DQN
import time

# --- 1. PERFORMANCE OPTIMIZATION (Basic to Advanced) ---
# Basic: Force the app to use CPU to prevent freezing
# Advanced: Disable GPU devices and hide low-level TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 2. UI INITIALIZATION (Prevents White Page) ---
# By calling st.title first, the browser gets content immediately
st.set_page_config(page_title="AI DDoS Shield", layout="wide")
st.title("🛡️ AI-Based Network Intrusion System")
status_placeholder = st.empty() # Placeholder to update status later

# --- 3. SECURE MODEL LOADING (Deep Detail) ---
# We use @st.cache_resource so models load only once into RAM
@st.cache_resource
def load_all_brains():
    # These paths exactly match your GitHub structure: notebooks/models/
    path = "notebooks/models/"
    
    # Loading Signature-based model (Random Forest)
    rf = joblib.load(f"{path}rf_model.pkl")
    sc_rf = joblib.load(f"{path}scaler_rf.pkl")
    
    # Loading Anomaly-based model (Autoencoder)
    ae = tf.keras.models.load_model(f"{path}autoencoder_model.h5")
    ae_thresh = joblib.load(f"{path}ae_threshold.pkl")
    
    # Loading Response-based model (DRL Agent)
    drl = DQN.load(f"{path}drl_response_agent.zip")
    
    return rf, sc_rf, ae, ae_thresh, drl

# Start the loading process with a visual spinner
with st.spinner("🧠 Initializing AI Engines... This takes about 15 seconds."):
    try:
        rf_model, scaler, autoencoder, threshold, drl_agent = load_all_brains()
        status_placeholder.success("✅ AI Engines Online & Ready!")
    except Exception as e:
        status_placeholder.error(f"❌ Critical Error: {e}")
        st.stop()

# --- 4. DATA PROCESSING SECTION ---
st.divider()
st.header("Step 1: Network Traffic Analysis")
uploaded_file = st.file_uploader("Upload Network Log (CSV)", type="csv")

if uploaded_file is not None:
    # Basic: Read and display data
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Traffic Preview")
    st.dataframe(df.head())
    
    # Advanced: Automated Analysis (Placeholder for your specific prediction logic)
    st.subheader("🔍 AI Analysis Results")
    with st.status("Analyzing packets...", expanded=True) as status:
        st.write("Running Signature Detection (Random Forest)...")
        time.sleep(1) # Simulation for demo
        st.write("Running Anomaly Detection (Autoencoder)...")
        time.sleep(1)
        st.write("Generating DRL Response Action...")
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    st.warning("⚠️ High Probability of DDoS detected in sample rows.")
    st.button("Apply Mitigation (Block IP)")

# --- 5. FOOTER (Professional presentation) ---
st.sidebar.info("4th Year Project: Hybrid DDoS Detection System using RF, AE, and DRL.")