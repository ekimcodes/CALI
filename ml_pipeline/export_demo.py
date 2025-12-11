import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import json
import os
from utils import calculate_hrv_features

# Config
WINDOW_SEC = 60
STEP_SEC = 10 # Predict every 10 seconds, but for viz we need continuous data
# We will generate continuous data by interpolation or justraw.
FS_WESAD = 700

def export_demo():
    print("Loading WESAD raw data for demo...")
    path = r"c:/Users/edwin/Desktop/CALI/WESAD/S10/S10.pkl"
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    chest = data['signal']['chest']
    # Take a 10 min slice: 5 min Baseline (Label 1) -> 5 min Stress (Label 2)
    # Find indices
    labels = data['label']
    
    # Indices for baseline
    base_idxs = np.where(labels == 1)[0]
    stress_idxs = np.where(labels == 2)[0]
    
    if len(base_idxs) < 210000: # 5 mins * 60 * 700 = 210000
        print("Not enough baseline")
    
    # Take chunk
    # We want a transition.
    # We find where label changes from 1 to 2?
    # WESAD structure: usually Base(1) -> Stress(2).
    # Let's verify.
    # changes = np.where(np.diff(labels) != 0)[0]
    
    # Simplification: Concatenate 2 mins Baseline + 2 mins Stress
    mins = 2
    samples = mins * 60 * FS_WESAD
    
    b_start = base_idxs[len(base_idxs)//2] # Middle of baseline
    s_start = stress_idxs[0] # Start of stress
    
    # Data Chunks
    # Baseline
    ecg_b = chest['ECG'][b_start : b_start+samples].flatten()
    eda_b = chest['EDA'][b_start : b_start+samples].flatten()
    temp_b = chest['Temp'][b_start : b_start+samples].flatten()
    
    # Stress
    ecg_s = chest['ECG'][s_start : s_start+samples].flatten()
    eda_s = chest['EDA'][s_start : s_start+samples].flatten()
    temp_s = chest['Temp'][s_start : s_start+samples].flatten()
    
    # Concat
    ecg = np.concatenate([ecg_b, ecg_s])
    eda = np.concatenate([eda_b, eda_s])
    temp = np.concatenate([temp_b, temp_s])
    
    # Downsample for JS (700Hz -> 50Hz)
    factor = 14
    ecg_viz = ecg[::factor]
    eda_viz = eda[::factor]
    temp_viz = temp[::factor]
    
    print(f"Viz Data Points: {len(ecg_viz)}")
    
    # Run Inference
    # Model expects features from 10 consecutive windows...
    # We need to slide a window over the RAW data, extract features, feed to model.
    # This is slow, but we pre-compute it.
    
    print("Loading Model...")
    model = load_model("model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    predictions = []
    
    # We iterate the raw data in steps equivalent to our Viz/Downsampled rate? 
    # No, model prediction is sparse (every 10s?). 
    # But for a smooth graph, we want prediction to interpolate?
    # Or just hold the value.
    
    # We'll compute prediction every 10s (STEP_SEC) based on previous 60s (WINDOW_SEC).
    
    seq_len = 10 # Model Sequence Length
    window_samples = WINDOW_SEC * FS_WESAD
    step_samples = STEP_SEC * FS_WESAD
    
    # Pre-compute features for all windows
    all_features = []
    
    # Define features columns
    cols = ['MEAN_RR', 'SDRR', 'RMSSD', 'pNN50', 'HR', 'MEAN_EDA', 'STD_EDA', 'MEAN_TEMP']
    
    num_windows = (len(ecg) - window_samples) // step_samples
    
    print(f"Computing features for {num_windows} windows...")
    
    for i in range(num_windows):
        start = i * step_samples
        end = start + window_samples
        
        w_ecg = ecg[start:end]
        w_eda = eda[start:end]
        w_temp = temp[start:end]
        
        # Peaks
        peaks = np.where((w_ecg[1:-1] > w_ecg[:-2]) & (w_ecg[1:-1] > w_ecg[2:]) & (w_ecg[1:-1] > 0.5))[0] + 1
        rr = np.diff(peaks) / FS_WESAD * 1000
        feat = calculate_hrv_features(rr)
        
        feat['MEAN_EDA'] = np.mean(w_eda)
        feat['STD_EDA'] = np.std(w_eda)
        feat['MEAN_TEMP'] = np.mean(w_temp)
        
        # To list in correct order
        vec = [feat[k] for k in cols]
        all_features.append(vec)
        
    # Scale
    all_features = scaler.transform(all_features)
    
    # Create Sequences for model
    # Model needs (1, 10, 8)
    preds = []
    # Pad beginning
    for i in range(len(all_features)):
        if i < seq_len - 1:
            # Not enough history
            p = 0.0 # Default
        else:
            # Seq from i-9 to i
            seq = all_features[i-seq_len+1 : i+1]
            seq = seq.reshape(1, seq_len, 8)
            p = model.predict(seq, verbose=0)[0][0]
            
        preds.append(float(p))
        
    # Now map predictions back to Viz Timeline
    # predictions[i] corresponds to window ending at (i * step) + window_size
    # We will expand this to per-sample array by interpolation
    
    # viz data length
    viz_len = len(ecg_viz)
    
    # Map predictions to time
    # Pred[0] is for time t = 60s.
    # Viz data is 0s to 4mins (240s).
    
    pred_curve = np.zeros(viz_len)
    
    # Fill
    # Step in seconds = 10. Viz Hz = 50. Step in indices = 500.
    step_viz_idx = int(STEP_SEC * 50)
    
    current_pred = 0.0
    
    # Alignment might be tricky. Let's just create JSON objects
    
    json_data = []
    
    for i in range(viz_len):
        t_sec = i / 50.0
        
        # Find relevant prediction
        # Pred index ~ (t_sec - 60) / 10
        p_idx = int((t_sec - WINDOW_SEC) / STEP_SEC)
        if p_idx < 0: p_idx = 0
        if p_idx >= len(preds): p_idx = len(preds) - 1
        
        val = preds[p_idx]
        
        # Smooth transition
        # simple hold
        
        obj = {
            't': t_sec,
            'ecg': float(ecg_viz[i]),
            'eda': float(eda_viz[i]),
            'temp': float(temp_viz[i]),
            'hr': 70.0, # Placeholder, dynamic HR hard to calc per sample without history
            'pred': val
        }
        json_data.append(obj)
        
    # Save as JS file
    print("Saving to dashboard/data.js...")
    content = f"const DEMO_DATA = {json.dumps(json_data)};"
    with open("dashboard/data.js", "w") as f:
        f.write(content)
        
    print("Done.")

if __name__ == "__main__":
    export_demo()
