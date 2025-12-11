import numpy as np
import pandas as pd

def calculate_hrv_features(rr_intervals):
    """
    Calculate basic HRV features from RR intervals (in ms).
    """
    if len(rr_intervals) < 2:
        return {
            'MEAN_RR': 0, 'SDRR': 0, 'RMSSD': 0, 'pNN50': 0, 'HR': 0
        }
    
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    sdrr = np.std(rr_intervals)
    mean_rr = np.mean(rr_intervals)
    hr = 60000 / mean_rr if mean_rr > 0 else 0
    
    # pNN50
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = (nn50 / len(diff_rr)) * 100
    
    return {
        'MEAN_RR': mean_rr,
        'SDRR': sdrr,
        'RMSSD': rmssd,
        'pNN50': pnn50,
        'HR': hr
    }

def get_sliding_windows(signal_data, window_size, step_size):
    """
    Generate sliding windows from signal data.
    """
    num_windows = (len(signal_data) - window_size) // step_size + 1
    if num_windows <= 0:
        return []
        
    windows = []
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        windows.append(signal_data[start:end])
        
    return np.array(windows)
