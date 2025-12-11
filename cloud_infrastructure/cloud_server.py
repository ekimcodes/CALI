from flask import Flask, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os
import time

app = Flask(__name__)

# Globals
model = None
scaler = None
MODEL_PATH = "../model.keras" # Assume model is in root or ml_pipeline
SCALER_PATH = "../scaler.pkl"

def load_inference_artifacts():
    global model, scaler
    print("Loading Model Artifacts...")
    try:
        model = load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("Model Loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.route('/upload_telemetry', methods=['POST'])
def upload_telemetry():
    """
    Receives a batch of packets.
    JSON Body: {
        "device_id": "NeuroPatch-001",
        "packets": [
            { "ts": 1234567890, "seq": 1, "ecg": [...], "eda": ..., "temp": ... },
            ...
        ]
    }
    """
    data = request.json
    device_id = data.get('device_id')
    packets = data.get('packets', [])
    
    print(f"Received {len(packets)} packets from {device_id}")
    
    # Process for Inference
    # We need a sequence of 10 windows to predict?
    # Or we assume the 'packets' contain features? 
    # For this sim, let's assume packets contain RAW data windows or pre-computed features?
    # To keep "device" dumb (as per typical BLE), device sends Raw or lightly processed.
    # But for our ML model (BiLSTM) we need sequences.
    
    # Let's assume the 'packets' are actually Feature Vectors for simplicity in this demo,
    # OR we buffer them here.
    
    # Let's assume each packet = 1 Time Step of Features (10s window computed on device or hub)
    # The Hub might do feature extraction? 
    # Let's stick to the Plan: Cloud does inference.
    # We will assume the 'packets' sent by Base Station are pre-processed Feature Vectors 
    # tailored for the model (so Base Station does compute features).
    
    results = []
    
    if model and scaler:
        # Extract features
        # Sequence of packets
        # We need 10 steps for 1 prediction.
        # If batch size < 10, we can't predict unless we have state.
        # For demo, we just predict on what we have if it matches shape.
        
        # Expecting packets to have 'features': [mean_rr, sdrr, ...]
        
        feature_batch = []
        for p in packets:
            if 'features' in p:
                feature_batch.append(p['features'])
                
        if len(feature_batch) > 0:
            # Scale
            X = scaler.transform(feature_batch)
            
            # Reshape for model (Batch, 10, 8)
            # Actually model expects (10, 8) for one sample. 
            # If we receive a stream, we slide a window.
            # Simplified: We treat each batch as a potential sequence source.
            
            if len(X) >= 10:
                # Take last 10
                seq = np.array(X[-10:]).reshape(1, 10, 8)
                pred = model.predict(seq, verbose=0)[0][0]
                results.append({"ts": time.time(), "stress_prob": float(pred)})
                print(f" >>> PREDICTION: Stress Probability = {pred:.2f}")
            else:
                print("Buffer filling... (need 10 samples)")
    
    return jsonify({"status": "received", "predictions": results}), 200

if __name__ == '__main__':
    load_inference_artifacts()
    # Run on 0.0.0.0 to be accessible
    app.run(host='0.0.0.0', port=5000)
