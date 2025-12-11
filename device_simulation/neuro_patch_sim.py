import socket
import time
import json
import pickle
import numpy as np
import pandas as pd
import sys
sys.path.append('../ml_pipeline')
from utils import calculate_hrv_features

# Config
HUB_IP = "127.0.0.1"
HUB_PORT = 5555
WESAD_PATH = "../WESAD/S10/S10.pkl"

# Simulation Speed
SPEED_FACTOR = 1.0 # 1x Real-time (approx)
TRANSMISSION_INTERVAL = 1.0 # Send data every second (e.g. 1Hz feature updates)
# If features are window-based (10s window), we slide?
# For smooth demo, we'll send features every 1s, based on previous 60s window.

def load_data():
    print("Loading WESAD for simulation...")
    with open(WESAD_PATH, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data['signal']['chest'], data['label']

def simulate_device():
    # Setup UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Load Data
    chest, labels = load_data()
    ecg = chest['ECG'].flatten()
    eda = chest['EDA'].flatten()
    temp = chest['Temp'].flatten()
    
    # FS = 700
    FS = 700
    WINDOW_SEC = 60
    
    # Find a Stress event transition
    # Labels: 1=Base, 2=Stress
    # Just iterate through the whole file or a specific slice
    # Let's iterate from index 0 or find a good start
    
    start_idx = 0 
    # Use same logic as export_demo to find a transition if possible
    # But linear is fine.
    
    current_idx = start_idx + (WINDOW_SEC * FS)
    
    print(f"[Device] Starting advertisement & transmission to {HUB_IP}:{HUB_PORT}")
    
    seq = 0
    
    while current_idx < len(ecg):
        start_time = time.time()
        
        # Extract Window
        w_start = current_idx - (WINDOW_SEC * FS)
        w_end = current_idx
        
        w_ecg = ecg[w_start:w_end]
        w_eda = eda[w_start:w_end]
        w_temp = temp[w_start:w_end]
        
        # Calc Features (On-device Edge Compute Simulation)
        # Or we send raw?
        # Plan says: "Packetize data". 
        # If we send raw 60s window * 700hz = 42000 float points, too big for UDP packet usually.
        # "BLE" usually sends chunks. 
        # For this prototype, we assume "Smart Sensor" sending calculated features OR we assume high throughput.
        # Let's send FEATURES to match the Cloud Server expectation.
        
        peaks = np.where((w_ecg[1:-1] > w_ecg[:-2]) & (w_ecg[1:-1] > w_ecg[2:]) & (w_ecg[1:-1] > 0.5))[0] + 1
        rr = np.diff(peaks) / FS * 1000
        feat = calculate_hrv_features(rr)
        
        feat['MEAN_EDA'] = np.mean(w_eda)
        feat['STD_EDA'] = np.std(w_eda)
        feat['MEAN_TEMP'] = np.mean(w_temp)
        
        # Convert feature vector to native float
        cols = ['MEAN_RR', 'SDRR', 'RMSSD', 'pNN50', 'HR', 'MEAN_EDA', 'STD_EDA', 'MEAN_TEMP']
        feature_vector = [float(feat[k]) for k in cols]
        
        # Create Packet
        packet = {
            "ts": time.time(),
            "seq": seq,
            "features": feature_vector,
            "battery": 98.5
        }
        
        # Send
        msg = json.dumps(packet).encode('utf-8')
        sock.sendto(msg, (HUB_IP, HUB_PORT))
        print(f"[Device] Sent Packet #{seq}")
        
        seq += 1
        current_idx += int(TRANSMISSION_INTERVAL * FS) # Step
        
        # Sleep
        elapsed = time.time() - start_time
        sleep_time = (TRANSMISSION_INTERVAL / SPEED_FACTOR) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == "__main__":
    simulate_device()
