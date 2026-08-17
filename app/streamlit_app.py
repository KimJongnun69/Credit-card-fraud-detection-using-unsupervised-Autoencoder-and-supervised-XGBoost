import streamlit as st
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import FraudInferencePipeline
from src.preprocessing import split_data_anomaly_detection, preprocess_features

st.title("Credit Card Fraud Detection Dashboard")
st.markdown("This dashboard compares Autoencoder Anomaly Detection with XGBoost on custom transactions.")

@st.cache_resource
def load_pipeline():
    return FraudInferencePipeline()

@st.cache_resource
def load_scalers(_pipeline):
    config = _pipeline.config
    base_dir = _pipeline.base_dir
    df = pd.read_csv(os.path.join(base_dir, config['data']['subset_path']))
    train_df, val_df, test_df = split_data_anomaly_detection(
        df, 
        val_size=config['data']['val_size'], 
        test_size=config['data']['test_size'], 
        random_state=config['random_seed']
    )
    _, _, _, scalers = preprocess_features(train_df, val_df, test_df)
    return scalers[0], scalers[1]  # scaler_time, scaler_amount

try:
    pipeline = load_pipeline()
    scaler_time, scaler_amount = load_scalers(pipeline)
    st.success(f"Models loaded successfully! Using device: {pipeline.device}")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

def preprocess_input(input_df):
    """Applies the exact same preprocessing to new data."""
    df_copy = input_df.copy()
    
    # Scale
    df_copy['Time_scaled'] = scaler_time.transform(df_copy[['Time']])
    df_copy['Amount_scaled'] = scaler_amount.transform(df_copy[['Amount']])
    
    # Ensure all V1-V28 are present
    v_cols = [f'V{i}' for i in range(1, 29)]
    for v in v_cols:
        if v not in df_copy.columns:
            df_copy[v] = 0.0 # Default if missing, though we'll validate later
            
    final_features = ['Time_scaled'] + v_cols + ['Amount_scaled']
    return df_copy[final_features].values

# Sidebar
st.sidebar.header("INPUT TRANSACTION")
input_mode = st.sidebar.radio("Mode", ["Sample Data", "Manual Input", "Upload CSV"])

st.sidebar.markdown("---")

# Render Sidebar Options
if input_mode == "Sample Data":
    st.sidebar.subheader("Sample Data")
    if st.sidebar.button("Generate Random Legitimate Transaction"):
        st.session_state['mode'] = 'single'
        st.session_state['tx'] = np.random.normal(0, 0.5, size=(1, 30))
        st.session_state['tx'][0, -1] = abs(np.random.normal(5, 2)) # Amount
    if st.sidebar.button("Generate Random 'Fraud-like' Transaction"):
        st.session_state['mode'] = 'single'
        st.session_state['tx'] = np.random.normal(3, 2.0, size=(1, 30))
        st.session_state['tx'][0, -1] = abs(np.random.normal(100, 50)) # Amount

elif input_mode == "Manual Input":
    st.sidebar.subheader("Manual Input")
    with st.sidebar.form("manual_form"):
        time_val = st.number_input("Time", value=0.0)
        amount_val = st.number_input("Amount", value=10.0)
        st.markdown("**V1 - V28 Features**")
        
        # A simple data editor or just columns
        v_dict = {}
        cols = st.columns(4)
        for i in range(1, 29):
            with cols[(i-1) % 4]:
                v_dict[f'V{i}'] = st.number_input(f"V{i}", value=0.0, step=0.1, key=f"v{i}")
                
        submitted = st.form_submit_button("Analyze Transaction")
        if submitted:
            data = {'Time': [time_val]}
            data.update({k: [v] for k, v in v_dict.items()})
            data['Amount'] = [amount_val]
            
            raw_df = pd.DataFrame(data)
            processed_tx = preprocess_input(raw_df)
            
            st.session_state['mode'] = 'single'
            st.session_state['tx'] = processed_tx

elif input_mode == "Upload CSV":
    st.sidebar.subheader("Upload CSV")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            req_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
            missing = [c for c in req_cols if c not in df_upload.columns]
            
            if missing:
                st.sidebar.error(f"Missing required columns: {missing}")
            else:
                if st.sidebar.button("Analyze Batch"):
                    # Strip Class if exists
                    if 'Class' in df_upload.columns:
                        df_upload = df_upload.drop(columns=['Class'])
                    
                    processed_tx = preprocess_input(df_upload)
                    st.session_state['mode'] = 'batch'
                    st.session_state['tx_batch'] = processed_tx
                    st.session_state['df_upload'] = df_upload
        except Exception as e:
            st.sidebar.error(f"Error parsing CSV: {e}")


# Main Content Area
if st.session_state.get('mode') == 'single' and 'tx' in st.session_state:
    tx = st.session_state['tx']
    
    # Inference
    results = pipeline.predict(tx)
    
    col1, col2 = st.columns(2)
    
    # Autoencoder results
    with col1:
        st.header("Autoencoder (Unsupervised)")
        score = float(results['ae_anomaly_scores'][0])
        decision = int(results['ae_decisions'][0])
        threshold = pipeline.threshold
        
        st.metric("Anomaly Score (MSE)", f"{score:.4f}")
        st.metric("Threshold", f"{threshold:.4f}")
        
        if decision == 1:
            st.error("🚨 FRAUD DETECTED")
        else:
            st.success("✅ LEGITIMATE")
            
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.barh(["Score"], [score], color='red' if decision else 'green')
        ax.axvline(threshold, color='black', linestyle='--', label='Threshold')
        ax.set_xlim(0, max(score * 1.5, threshold * 1.5))
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        
    # XGBoost results
    with col2:
        st.header("XGBoost (Supervised)")
        prob = float(results['xgb_probabilities'][0])
        decision_xgb = int(results['xgb_decisions'][0])
        
        st.metric("Fraud Probability", f"{prob:.4f}")
        
        if decision_xgb == 1:
            st.error("🚨 FRAUD DETECTED")
        else:
            st.success("✅ LEGITIMATE")
            
        fig2, ax2 = plt.subplots(figsize=(6, 2))
        ax2.barh(["Probability"], [prob], color='red' if decision_xgb else 'green')
        ax2.axvline(0.5, color='black', linestyle='--', label='Threshold (0.5)')
        ax2.set_xlim(0, 1)
        ax2.legend()
        st.pyplot(fig2)
        plt.close(fig2)

elif st.session_state.get('mode') == 'batch' and 'tx_batch' in st.session_state:
    st.header("Batch Prediction Results")
    
    tx_batch = st.session_state['tx_batch']
    df_upload = st.session_state['df_upload']
    
    # Inference
    results = pipeline.predict(tx_batch)
    
    # Construct Results DataFrame
    out_df = df_upload.copy()
    out_df['AE_Anomaly_Score'] = results['ae_anomaly_scores']
    out_df['AE_Decision'] = results['ae_decisions']
    out_df['XGB_Probability'] = results['xgb_probabilities']
    out_df['XGB_Decision'] = results['xgb_decisions']
    
    st.dataframe(out_df)
    
    # Download Button
    csv = out_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='fraud_predictions.csv',
        mime='text/csv',
    )
    
else:
    st.info("Please select an input mode from the sidebar and generate/analyze a transaction.")

st.markdown("---")
st.subheader("Model Performance Summary (Test Set)")
st.markdown("""
| Metric | Autoencoder | XGBoost |
|---|---|---|
| Approach | Unsupervised | Supervised |
| PR-AUC | 0.767 | 0.971 |
| ROC-AUC | 0.941 | 0.985 |

*Note: The XGBoost model performs better overall, but requires labeled fraud data to train. The Autoencoder can detect novel fraud patterns without ever seeing them during training.*
""")
