import numpy as np
import pandas as pd
import pickle
import os
import glob
from utils import calculate_hrv_features

# Configuration
WINDOW_SEC = 60  # Window size in seconds
STEP_SEC = 10     # Step size in seconds
SR_WESAD = 700    # WESAD Chest Sampling Rate (approx, varies by sensor)

def load_wesad(path):
    """
    Load WESAD dataset (Subject 10).
    Returns: DataFrame with calculated features per window.
    """
    print(f"Loading WESAD from {path}")
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Failed to load WESAD: {e}")
        return pd.DataFrame()

    # Extract signals
    chest = data['signal']['chest']
    ecg = chest['ECG'].flatten()
    eda = chest['EDA'].flatten()
    temp = chest['Temp'].flatten()
    labels_raw = data['label']
    
    # WESAD Sampling Rate is 700Hz
    fs = 700
    window_samples = WINDOW_SEC * fs
    step_samples = STEP_SEC * fs
    
    windows = []
    
    # Iterate windows
    for i in range(0, len(ecg) - window_samples, step_samples):
        w_ecg = ecg[i:i+window_samples]
        w_eda = eda[i:i+window_samples]
        w_temp = temp[i:i+window_samples]
        w_label = labels_raw[i:i+window_samples]
        
        # Determine label
        unique, counts = np.unique(w_label, return_counts=True)
        label = unique[np.argmax(counts)]
        
        # Map to Binary
        if label == 2: # Stress
            binary_label = 1
        elif label in [1, 3, 4]: # Baseline/Amusement/Med
            binary_label = 0
        else:
            continue 
            
        peaks = np.where((w_ecg[1:-1] > w_ecg[:-2]) & (w_ecg[1:-1] > w_ecg[2:]) & (w_ecg[1:-1] > 0.5))[0] + 1
        rr_intervals = np.diff(peaks) / fs * 1000 
        
        hrv_feats = calculate_hrv_features(rr_intervals)
        
        features = hrv_feats.copy()
        features['MEAN_EDA'] = np.mean(w_eda)
        features['STD_EDA'] = np.std(w_eda)
        features['MEAN_TEMP'] = np.mean(w_temp)
        features['label'] = binary_label
        features['source'] = 'WESAD'
        features['subject'] = 'S10'
        features['timestamp'] = i / fs
        
        windows.append(features)
        
    return pd.DataFrame(windows)

def load_exam_stress(base_path):
    """
    Load Exam Stress Dataset (E4 data).
    Returns: DataFrame with features.
    """
    print(f"Loading Exam Stress from {base_path}")
    student_dirs = glob.glob(os.path.join(base_path, "S*"))
    all_windows = []
    
    for s_dir in student_dirs:
        student_id = os.path.basename(s_dir)
        # Check subdirs: Final, Midterm 1, Midterm 2
        for session in ['Final', 'Midterm 1', 'Midterm 2']:
            session_path = os.path.join(s_dir, session)
            if not os.path.exists(session_path): continue
            
            try:
                # Load IBI
                ibi_path = os.path.join(session_path, 'IBI.csv')
                if not os.path.exists(ibi_path): continue
                # IBI.csv: 1st row start time, then time,ibi
                if os.stat(ibi_path).st_size < 10: continue
                
                start_time_df = pd.read_csv(ibi_path, nrows=1, header=None)
                if start_time_df.empty: continue
                start_time = float(start_time_df.iloc[0, 0])
                
                ibi_data = pd.read_csv(ibi_path, skiprows=1, header=None, names=['time', 'ibi'])
                if ibi_data.empty: continue
                ibi_data['abs_time'] = start_time + ibi_data['time']
                
                # Load EDA
                eda_path = os.path.join(session_path, 'EDA.csv')
                if not os.path.exists(eda_path): continue
                eda_start = float(pd.read_csv(eda_path, nrows=1, header=None).iloc[0, 0])
                eda_fs = float(pd.read_csv(eda_path, nrows=2, header=None).iloc[1, 0])
                eda_vals = pd.read_csv(eda_path, skiprows=2, header=None)[0].values
                eda_times = eda_start + np.arange(len(eda_vals)) / eda_fs
                
                # Load Temp
                temp_path = os.path.join(session_path, 'TEMP.csv')
                if not os.path.exists(temp_path): continue
                temp_start = float(pd.read_csv(temp_path, nrows=1, header=None).iloc[0, 0])
                temp_fs = float(pd.read_csv(temp_path, nrows=2, header=None).iloc[1, 0])
                temp_vals = pd.read_csv(temp_path, skiprows=2, header=None)[0].values
                temp_times = temp_start + np.arange(len(temp_vals)) / temp_fs
                
                # Assume Exam = Stress (1)
                label = 1 
                
                # Sliding window
                if len(eda_times) == 0: continue
                end_time = eda_times[-1]
                
                for t in np.arange(eda_times[0], end_time - WINDOW_SEC, STEP_SEC):
                    # Features logic
                    idx_start = int((t - eda_start) * eda_fs)
                    idx_end = int((t + WINDOW_SEC - eda_start) * eda_fs)
                    
                    if idx_start < 0 or idx_end > len(eda_vals): continue
                    w_eda = eda_vals[idx_start:idx_end]
                    
                    idx_start_t = int((t - temp_start) * temp_fs)
                    idx_end_t = int((t + WINDOW_SEC - temp_start) * temp_fs)
                    w_temp = temp_vals[idx_start_t:idx_end_t] if (idx_start_t >= 0 and idx_end_t <= len(temp_vals)) else []
                    
                    w_ibis = ibi_data[(ibi_data['abs_time'] >= t) & (ibi_data['abs_time'] < t + WINDOW_SEC)]['ibi'].values * 1000
                    
                    hrv_feats = calculate_hrv_features(w_ibis)
                    
                    features = hrv_feats.copy()
                    features['MEAN_EDA'] = np.mean(w_eda) if len(w_eda) > 0 else 0
                    features['STD_EDA'] = np.std(w_eda) if len(w_eda) > 0 else 0
                    features['MEAN_TEMP'] = np.mean(w_temp) if len(w_temp) > 0 else 0
                    features['label'] = label
                    features['source'] = 'ExamStress'
                    features['subject'] = student_id
                    features['timestamp'] = t
                    
                    all_windows.append(features)
                    
            except Exception as e:
                print(f"Error processing {s_dir}/{session}: {e}")
                continue
            
    return pd.DataFrame(all_windows)

def load_hrv_dataset(path):
    """
    Load HRV dataset (train.csv).
    Returns: DataFrame with features.
    """
    print(f"Loading HRV Dataset from {path}")
    try:
        df = pd.read_csv(path)
        df['label'] = df['datasetId'].apply(lambda x: 0) 
        
        def map_label(cond):
            if cond == 'no stress': return 0
            if cond in ['interruption', 'time pressure']: return 1
            return 0
            
        df['label'] = df['condition'].apply(map_label)
        df['MEAN_EDA'] = 0.0
        df['STD_EDA'] = 0.0
        df['MEAN_TEMP'] = 0.0
        df['source'] = 'HRV_Dataset'
        df['subject'] = df['datasetId'].astype(str)
        df['timestamp'] = df.index # Dummy time
        
        cols = ['MEAN_RR', 'SDRR', 'RMSSD', 'pNN50', 'HR', 'MEAN_EDA', 'STD_EDA', 'MEAN_TEMP', 'label', 'source', 'subject', 'timestamp']
        return df[cols]
        
    except Exception as e:
        print(f"Error loading HRV Dataset: {e}")
        return pd.DataFrame()

def load_and_merge_data():
    wesad_path = r"c:/Users/edwin/Desktop/CALI/WESAD/S10/S10.pkl"
    hrv_path = r"c:/Users/edwin/Desktop/CALI/hrv dataset/data/final/train.csv"
    exam_path = r"c:/Users/edwin/Desktop/CALI/a-wearable-exam-stress-dataset-for-predicting-cognitive-performance-in-real-world-settings-1.0.0/Data/Data"
    
    df1 = load_wesad(wesad_path)
    df2 = load_hrv_dataset(hrv_path)
    df3 = load_exam_stress(exam_path)
    
    # Concatenate
    print(f"WESAD samples: {len(df1)}")
    print(f"HRV samples: {len(df2)}")
    print(f"Exam samples: {len(df3)}")
    
    full_df = pd.concat([df1, df2, df3], ignore_index=True)
    full_df = full_df.fillna(0)
    
    return full_df

if __name__ == "__main__":
    df = load_and_merge_data()
    print("Combined Data Shape:", df.shape)
    print(df['source'].value_counts())
    print(df['label'].value_counts())
    
    # Save processed data
    df.to_csv("c:/Users/edwin/Desktop/CALI/ml_pipeline/processed_data.csv", index=False)
