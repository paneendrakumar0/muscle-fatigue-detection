import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def split_chronologically(features_df, test_size=0.2):
    """
    Splits the dataframe chronologically within each class to ensure
    the test set is representative and there is no time-series leakage.
    """
    train_dfs = []
    test_dfs = []
    
    # We maintain the order within each class
    for label in ['Rested', 'Fatiguing', 'Exhausted']:
        class_df = features_df[features_df['label'] == label]
        split_idx = int(len(class_df) * (1 - test_size))
        
        train_dfs.append(class_df.iloc[:split_idx])
        test_dfs.append(class_df.iloc[split_idx:])
        
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)
    
    return train_df, test_df

def build_and_train_model(train_df):
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

if __name__ == "__main__":
    from data_generation import generate_synthetic_data
    from feature_extraction import extract_features
    
    print("Generating data...")
    df = generate_synthetic_data()
    print("Extracting features...")
    features_df = extract_features(df)
    
    print("Splitting data chronologically...")
    train_df, test_df = split_chronologically(features_df)
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    print("Training distribution:")
    print(train_df['label'].value_counts())
    print("Testing distribution:")
    print(test_df['label'].value_counts())
    
    print("Training model...")
    pipeline = build_and_train_model(train_df)
    
    print("Model trained successfully.")
