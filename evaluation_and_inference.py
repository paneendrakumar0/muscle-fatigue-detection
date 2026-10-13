import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from feature_extraction import extract_features

def evaluate_model(pipeline, test_df):
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['Rested', 'Fatiguing', 'Exhausted'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rested', 'Fatiguing', 'Exhausted'],
                yticklabels=['Rested', 'Fatiguing', 'Exhausted'])
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to 'confusion_matrix.png'")
    
def save_model(pipeline, filename='fatigue_model.pkl'):
    joblib.dump(pipeline, filename)
    print(f"Model saved to {filename}")

def predict_live_data(live_window_df, model_path='fatigue_model.pkl', fs=200):
    """
    Simulates receiving a single window of live data from an ESP32 and predicting its state.
    live_window_df should be a DataFrame with 1 second of data (e.g., 200 rows).
    """
    # Load the trained model
    pipeline = joblib.load(model_path)
    
    # Extract features from the single window (without 'label')
    emg_raw = live_window_df['emg_raw'].values
    accel_x = live_window_df['accel_x'].values
    accel_y = live_window_df['accel_y'].values
    accel_z = live_window_df['accel_z'].values
    gyro_x = live_window_df['gyro_x'].values
    gyro_y = live_window_df['gyro_y'].values
    gyro_z = live_window_df['gyro_z'].values
    
    # Use our feature extraction functions
    from feature_extraction import extract_emg_features, extract_imu_features
    emg_feats = extract_emg_features(emg_raw, fs)
    imu_feats = extract_imu_features(
        accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    )
    
    # Create feature dataframe for the pipeline
    # The keys in the dictionaries exactly match the column names needed
    feature_dict = {**emg_feats, **imu_feats}
    
    # Convert scalar values to lists for DataFrame construction
    for key in feature_dict:
        feature_dict[key] = [feature_dict[key]]
        
    single_window_features = pd.DataFrame(feature_dict)
    
    # Predict
    prediction = pipeline.predict(single_window_features)[0]
    return prediction

if __name__ == "__main__":
    from data_generation import generate_synthetic_data
    from feature_extraction import extract_features
    from model_training import split_chronologically, build_and_train_model
    
    print("Running Full Pipeline...")
    # Step 1
    df = generate_synthetic_data()
    
    # Step 2
    features_df = extract_features(df)
    
    # Step 3
    train_df, test_df = split_chronologically(features_df)
    pipeline = build_and_train_model(train_df)
    
    # Step 4 - Evaluation
    print("\n--- Evaluation ---")
    evaluate_model(pipeline, test_df)
    
    # Step 4 - Save Model
    print("\n--- Persistence ---")
    save_model(pipeline, 'fatigue_model.pkl')
    
    # Step 4 - Live Inference Simulation
    print("\n--- Live Inference Simulation ---")
    # Take a random 1-second window from the test data (e.g., from the 'Fatiguing' section)
    # The original df has 60000 samples. The 'Fatiguing' section is between index 20000 and 40000
    start_idx = 35000
    window_samples = 200 # 1 second at 200Hz
    live_window = df.iloc[start_idx:start_idx+window_samples].copy()
    
    actual_state = live_window['label'].iloc[0]
    predicted_state = predict_live_data(live_window, 'fatigue_model.pkl')
    
    print(f"Simulated live data - Actual State: {actual_state}")
    print(f"Simulated live data - Predicted State: {predicted_state}")
    
    if actual_state == predicted_state:
        print("Success: Prediction matches actual state!")
    else:
        print("Warning: Prediction does not match actual state.")
