import numpy as np
import pandas as pd

def generate_synthetic_data(duration_minutes=5, fs=200):
    """
    Generates synthetic EMG and IMU data for muscle fatigue classification.
    """
    np.random.seed(42)
    
    total_samples = duration_minutes * 60 * fs
    timestamps = np.arange(total_samples) / fs
    
    # Split the duration into 3 equal parts for the 3 states
    samples_per_state = total_samples // 3
    
    labels = np.array(['Rested'] * samples_per_state + 
                      ['Fatiguing'] * samples_per_state + 
                      ['Exhausted'] * (total_samples - 2 * samples_per_state))
    
    # Initialize arrays
    emg_raw = np.zeros(total_samples)
    accel_x = np.zeros(total_samples)
    accel_y = np.zeros(total_samples)
    accel_z = np.zeros(total_samples)
    gyro_x = np.zeros(total_samples)
    gyro_y = np.zeros(total_samples)
    gyro_z = np.zeros(total_samples)
    
    # Time arrays for each segment
    t_rested = timestamps[:samples_per_state]
    t_fatiguing = timestamps[samples_per_state:2*samples_per_state]
    t_exhausted = timestamps[2*samples_per_state:]
    
    # -- Rested State --
    # EMG: Baseline amplitude and higher median frequency (e.g., 100 Hz)
    f_rested = 100
    emg_raw[:samples_per_state] = 0.5 * np.sin(2 * np.pi * f_rested * t_rested) + np.random.normal(0, 0.1, samples_per_state)
    accel_x[:samples_per_state] = np.random.normal(0, 0.05, samples_per_state)
    accel_y[:samples_per_state] = np.random.normal(0, 0.05, samples_per_state)
    accel_z[:samples_per_state] = np.random.normal(9.81, 0.05, samples_per_state) # Gravity
    
    # -- Fatiguing State --
    # EMG: Amplitude increases progressively, frequency decreases progressively
    # Frequency drops from 100 Hz to 70 Hz, Amplitude increases from 0.5 to 1.0
    progress = np.linspace(0, 1, samples_per_state)
    f_fatiguing = 100 - 30 * progress
    amp_fatiguing = 0.5 + 0.5 * progress
    # To properly simulate changing frequency, we integrate frequency over time to get phase
    phase_fatiguing = np.cumsum(2 * np.pi * f_fatiguing / fs)
    emg_raw[samples_per_state:2*samples_per_state] = amp_fatiguing * np.sin(phase_fatiguing) + np.random.normal(0, 0.15, samples_per_state)
    
    # Tremors start increasing
    tremor_variance = 0.05 + 0.2 * progress
    accel_x[samples_per_state:2*samples_per_state] = np.random.normal(0, tremor_variance)
    accel_y[samples_per_state:2*samples_per_state] = np.random.normal(0, tremor_variance)
    accel_z[samples_per_state:2*samples_per_state] = np.random.normal(9.81, tremor_variance)
    
    # -- Exhausted State --
    # EMG: High amplitude (e.g., 1.2), low frequency (e.g., 60 Hz)
    f_exhausted = 60
    emg_raw[2*samples_per_state:] = 1.2 * np.sin(2 * np.pi * f_exhausted * t_exhausted) + np.random.normal(0, 0.2, len(t_exhausted))
    
    # High tremors
    accel_x[2*samples_per_state:] = np.random.normal(0, 0.3, len(t_exhausted))
    accel_y[2*samples_per_state:] = np.random.normal(0, 0.3, len(t_exhausted))
    accel_z[2*samples_per_state:] = np.random.normal(9.81, 0.3, len(t_exhausted))
    
    # Gyroscope data (random noise, slightly increasing with fatigue)
    gyro_x[:samples_per_state] = np.random.normal(0, 0.01, samples_per_state)
    gyro_y[:samples_per_state] = np.random.normal(0, 0.01, samples_per_state)
    gyro_z[:samples_per_state] = np.random.normal(0, 0.01, samples_per_state)
    
    gyro_x[samples_per_state:2*samples_per_state] = np.random.normal(0, 0.01 + 0.04 * progress)
    gyro_y[samples_per_state:2*samples_per_state] = np.random.normal(0, 0.01 + 0.04 * progress)
    gyro_z[samples_per_state:2*samples_per_state] = np.random.normal(0, 0.01 + 0.04 * progress)
    
    gyro_x[2*samples_per_state:] = np.random.normal(0, 0.05, len(t_exhausted))
    gyro_y[2*samples_per_state:] = np.random.normal(0, 0.05, len(t_exhausted))
    gyro_z[2*samples_per_state:] = np.random.normal(0, 0.05, len(t_exhausted))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'emg_raw': emg_raw,
        'accel_x': accel_x,
        'accel_y': accel_y,
        'accel_z': accel_z,
        'gyro_x': gyro_x,
        'gyro_y': gyro_y,
        'gyro_z': gyro_z,
        'label': labels
    })
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    print("Synthetic Data Generation Complete")
    print(f"Shape: {df.shape}")
    print(df.head())
    print("\nClass distribution:")
    print(df['label'].value_counts())
