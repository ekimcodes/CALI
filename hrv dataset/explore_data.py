import pickle
import pandas as pd
import os
import numpy as np

def explore_wesad():
    print("--- WESAD Dataset ---")
    wesad_path = r"c:/Users/edwin/Desktop/CALI/WESAD/S10/S10.pkl"
    if not os.path.exists(wesad_path):
        print(f"File not found: {wesad_path}")
        return

    try:
        with open(wesad_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        print(f"Keys: {data.keys()}")
        if 'signal' in data:
            print(f"Signal Keys: {data['signal'].keys()}")
            if 'chest' in data['signal']:
                print(f"Chest Keys: {data['signal']['chest'].keys()}")
            if 'wrist' in data['signal']:
                print(f"Wrist Keys: {data['signal']['wrist'].keys()}")
        if 'label' in data:
            print(f"Label Shape: {data['label'].shape}")
            print(f"Unique Labels: {np.unique(data['label'])}")
    except Exception as e:
        print(f"Error loading WESAD: {e}")

def explore_hrv():
    print("\n--- HRV Dataset ---")
    hrv_path = r"c:/Users/edwin/Desktop/CALI/hrv dataset/data/final/train.csv"
    if not os.path.exists(hrv_path):
        print(f"File not found: {hrv_path}")
        return

    try:
        df = pd.read_csv(hrv_path, nrows=5)
        print("Columns:", df.columns.tolist())
        print(df.head())
    except Exception as e:
        print(f"Error loading HRV: {e}")

def explore_exam_stress():
    print("\n--- Exam Stress Dataset ---")
    base_path = r"c:/Users/edwin/Desktop/CALI/a-wearable-exam-stress-dataset-for-predicting-cognitive-performance-in-real-world-settings-1.0.0/Data"
    if not os.path.exists(base_path):
        print(f"Directory not found: {base_path}")
        return
    
    # List files in the directory
    files = os.listdir(base_path)
    print(f"Files found: {files[:5]}")
    
    # Try to read one CSV if it exists
    csv_files = [f for f in files if f.endswith('.csv')]
    if csv_files:
        first_csv = os.path.join(base_path, csv_files[0])
        print(f"Reading: {first_csv}")
        try:
            df = pd.read_csv(first_csv, nrows=5)
            print("Columns:", df.columns.tolist())
            print(df.head())
        except Exception as e:
            print(f"Error reading CSV: {e}")
    else:
        # Check subdirectories
        subdirs = [d for d in files if os.path.isdir(os.path.join(base_path, d))]
        if subdirs:
            print(f"Looking into subdir: {subdirs[0]}")
            sub_path = os.path.join(base_path, subdirs[0])
            sub_files = os.listdir(sub_path)
            print(f"Files in subdir: {sub_files[:5]}")

if __name__ == "__main__":
    explore_wesad()
    explore_hrv()
    explore_exam_stress()
