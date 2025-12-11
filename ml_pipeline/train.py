import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Masking
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Config
SEQ_LEN = 10
FEATURES = ['MEAN_RR', 'SDRR', 'RMSSD', 'pNN50', 'HR', 'MEAN_EDA', 'STD_EDA', 'MEAN_TEMP']
TARGET = 'label'

def create_sequences(df, seq_len):
    """
    Create sequences from DataFrame.
    """
    sequences = []
    labels = []
    
    # Process WESAD and Exam Stress (Time Series)
    ts_df = df[df['source'].isin(['WESAD', 'ExamStress'])]
    
    # Group by subject and source
    # We assume 'timestamp' is sorted or we sort it
    ts_df = ts_df.sort_values(['source', 'subject', 'timestamp'])
    
    for (source, subject), group in ts_df.groupby(['source', 'subject']):
        data = group[FEATURES].values
        target = group[TARGET].values
        
        # Create sliding windows
        for i in range(len(data) - seq_len):
            sequences.append(data[i:i+seq_len])
            # Label is the label of the last step (or majority?)
            # Usually last step for prediction
            labels.append(target[i+seq_len-1])
            
    # Process HRV Dataset (Static)
    hrv_df = df[df['source'] == 'HRV_Dataset']
    # Use a subset if too large? 369k is fine.
    # Tile the data to match seq_len
    hrv_data = hrv_df[FEATURES].values
    hrv_target = hrv_df[TARGET].values
    
    # Vectorized tiling: (N, Features) -> (N, SeqLen, Features)
    # This might consume memory: 370k * 10 * 8 * 4bytes ~ 118MB. Safe.
    hrv_seqs = np.tile(hrv_data[:, np.newaxis, :], (1, seq_len, 1))
    
    # Convert properly
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    print(f"Time Series Seqs: {sequences.shape}")
    print(f"HRV Static Seqs: {hrv_seqs.shape}")
    
    # Merging
    if len(sequences) > 0:
        X = np.concatenate([sequences, hrv_seqs], axis=0)
        y = np.concatenate([labels, hrv_target], axis=0)
    else:
        X = hrv_seqs
        y = hrv_target
        
    return X, y

def train():
    print("Loading Data...")
    df = pd.read_csv("ml_pipeline/processed_data.csv")
    
    # Normalize
    print("Normalizing...")
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    
    # Create Sequences
    print("Creating Sequences...")
    X, y = create_sequences(df, SEQ_LEN)
    
    # Shuffle and Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    print(f"Train Shape: {X_train.shape}")
    print(f"Test Shape: {X_test.shape}")
    
    # Build Model
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(SEQ_LEN, len(FEATURES))), # Optional if we pad
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Training...")
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
    
    # Save
    print("Saving artifacts...")
    model.save("model.keras") # .keras is new format
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    print("Done.")

if __name__ == "__main__":
    train()
