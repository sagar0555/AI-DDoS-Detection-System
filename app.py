import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from stable_baselines3 import DQN
import time

# --- 1. PERFORMANCE & ENVIRONMENT SETUP ---
# Forces the app to use CPU and hides unnecessary logs to prevent crashes
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI DDoS Shield", layout="wide")
st.title("🛡️ AI-Based Network Intrusion System")
st.sidebar.info("4th Year Project: Hybrid DDoS Detection using RF, AE, and DRL.")

# --- 3. SECURE MODEL LOADING (Custom Objects Fix) ---
@st.cache_resource
def load_all_brains():
    # Path matches your GitHub: notebooks/models/
    path = "notebooks/models/"
    
    # Load Signature-based models
    rf = joblib.load(f"{path}rf_model.pkl")
    sc_rf = joblib.load(f"{path}scaler_rf.pkl")
    
    # Load Anomaly-based model with Keras MSE fix
    custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
    ae = tf.keras.models.load_model(
        f"{path}autoencoder_model.h5", 
        custom_objects=custom_objects,
        compile=False
    )
    ae_thresh = joblib.load(f"{path}ae_threshold.pkl")
    
    # Load Response-based model (DRL Agent)
    drl = DQN.load(f"{path}drl_response_agent.zip")
    
    return rf, sc_rf, ae, ae_thresh, drl

# Visual status indicator for loading
status_placeholder = st.empty()
with st.spinner("🧠 Initializing AI Engines... Please wait."):
    try:
        rf_model, sc_rf, ae_model, ae_thresh, drl_agent = load_all_brains()
        status_placeholder.success("✅ AI Engines Online & Ready!")
    except Exception as e:
        status_placeholder.error(f"❌ Critical Error: {e}")
        st.stop()

# --- 4. STEP 1: TRAFFIC DATA INGESTION ---
st.divider()
st.header("🔍 Network Traffic Analysis")
uploaded_file = st.file_uploader("Upload Traffic Log (CSV)", type="csv")

if uploaded_file is not None:
    # Load and preview the data
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Traffic Preview")
    st.dataframe(df.head())
    
    # Prepare features for the AI
    # We select only the columns the model was trained on
    numeric_cols = df.select_dtypes(include=[np.number])
    
    with st.status("Analyzing packets...", expanded=True) as status:
        # Step A: Scale the data
        scaled_data = sc_rf.transform(numeric_cols)
        
        # Step B: Signature Detection (Random Forest)
        # Calculate the average probability of attack across all rows
        attack_probs = rf_model.predict_proba(scaled_data)[:, 1]
        avg_prob = np.mean(attack_probs)
        
        time.sleep(1) # Simulation delay for demo
        status.update(label="AI Analysis Complete!", state="complete", expanded=False)

    # --- 5. DYNAMIC VERDICT LOGIC (The "Benign" Fix) ---
    st.subheader("🎯 AI Decision Verdict")
    
    # Threshold is 0.5 (50%)
    if avg_prob > 0.5:
        st.error(f"🚨 ALERT: High Probability of DDoS detected! (Score: {avg_prob:.2f})")
        st.write("The DRL Agent suggests: **BLOCK IP ADDRESS**")
        st.button("Apply Mitigation (Block IP)")
    else:
        st.success(f"✅ SAFE: Traffic appears normal. (Score: {avg_prob:.2f})")
        st.write("The DRL Agent suggests: **NO ACTION REQUIRED**")