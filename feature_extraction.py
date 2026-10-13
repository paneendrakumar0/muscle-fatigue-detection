import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy

def extract_emg_features(emg_window, fs):
    """
    Extracts time and frequency domain features from an EMG window.
    Returns a dictionary of features.
    """
    features = {}
    
    # Time Domain
    features['emg_rms'] = np.sqrt(np.mean(emg_window**2))
    features['emg_mav'] = np.mean(np.abs(emg_window))
    
    # Zero-Crossing Rate (ZCR)
    # Number of times the signal changes sign
    zero_crossings = np.where(np.diff(np.sign(emg_window)))[0]
    features['emg_zcr'] = len(zero_crossings) / len(emg_window)
    
    # Waveform Length (WL)
    # Cumulative length of the waveform
    features['emg_wl'] = np.sum(np.abs(np.diff(emg_window)))
    
    # Frequency Domain (Welch's Method)
    f, Pxx = welch(emg_window, fs=fs, nperseg=len(emg_window))
    
    # Mean Frequency (MNF)
    features['emg_mnf'] = np.sum(f * Pxx) / np.sum(Pxx) if np.sum(Pxx) > 0 else 0
    
    # Median Frequency (MDF)
    cumulative_power = np.cumsum(Pxx)
    total_power = cumulative_power[-1]
    mdf_idx = np.where(cumulative_power >= total_power / 2)[0]
    features['emg_mdf'] = f[mdf_idx[0]] if len(mdf_idx) > 0 else 0
    
    # Spectral Entropy
    # Normalizing power spectrum to use as a probability distribution
    Pxx_norm = Pxx / np.sum(Pxx)
    features['emg_spectral_entropy'] = entropy(Pxx_norm)
    
    return features

def extract_imu_features(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z):
    """
    Extracts features from IMU data.
    Returns a dictionary of features.
    """
    features = {}
    
    # Variance of the acceleration vector magnitude
    accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    features['accel_var'] = np.var(accel_mag)
    
    # Signal Magnitude Area (SMA) - quantifies overall movement intensity
    features['accel_sma'] = np.sum(np.abs(accel_x) + np.abs(accel_y) + np.abs(accel_z)) / len(accel_x)
    
    # RMS of the gyroscope axes
    features['gyro_rms_x'] = np.sqrt(np.mean(gyro_x**2))
    features['gyro_rms_y'] = np.sqrt(np.mean(gyro_y**2))
    features['gyro_rms_z'] = np.sqrt(np.mean(gyro_z**2))
    
    return features

def extract_features(df, fs=200, window_size_sec=1.0, overlap_sec=0.5):
    """
    Extracts features using a sliding window across the entire dataframe.
    """
    window_samples = int(window_size_sec * fs)
    step_samples = int((window_size_sec - overlap_sec) * fs)
    
    features_list = []
    
    for start_idx in range(0, len(df) - window_samples + 1, step_samples):
        end_idx = start_idx + window_samples
        window = df.iloc[start_idx:end_idx]
        
        # Ensure we have a full window
        if len(window) < window_samples:
            continue
            
        emg_raw = window['emg_raw'].values
        accel_x = window['accel_x'].values
        accel_y = window['accel_y'].values
        accel_z = window['accel_z'].values
        gyro_x = window['gyro_x'].values
        gyro_y = window['gyro_y'].values
        gyro_z = window['gyro_z'].values
        
        # Label (take the most common label in the window)
        label = window['label'].mode()[0]
        
        emg_feats = extract_emg_features(emg_raw, fs)
        imu_feats = extract_imu_features(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
        
        # Combine dictionaries
        window_features = {**emg_feats, **imu_feats}
        window_features['label'] = label
        
        features_list.append(window_features)
        
    return pd.DataFrame(features_list)

if __name__ == "__main__":
    from data_generation import generate_synthetic_data
    print("Generating data...")
    df = generate_synthetic_data()
    print("Extracting features with advanced metrics...")
    features_df = extract_features(df)
    print(f"Features shape: {features_df.shape}")
    print(features_df.head())
    print("\nFeature distribution per class:")
    print(features_df.groupby('label').mean())
