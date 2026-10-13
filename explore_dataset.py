import pandas as pd
import os

base_path = r"d:\Downloads\Fatigue\EMG Dataset for Muscle Fatigue Analysis in Biceps\EMG Dataset for Muscle Fatigue Analysis in Biceps\BBDD_EMG_Accel"

# Let's read one data file
sample_file = os.path.join(base_path, "Suj_1_B1.xlsx")
df_sample = pd.read_excel(sample_file)
print(f"File: Suj_1_B1.xlsx")
print(f"Shape: {df_sample.shape}")
print(f"Columns: {df_sample.columns.tolist()}")
print(df_sample.head())

# Let's read the labels file
labels_file = os.path.join(base_path, "Dataset_labels.xlsx")
df_labels = pd.read_excel(labels_file)
print("\nFile: Dataset_labels.xlsx")
print(f"Shape: {df_labels.shape}")
print(f"Columns: {df_labels.columns.tolist()}")
print(df_labels.head())

# Let's read ManualSegmentation
seg_file = r"d:\Downloads\Fatigue\EMG Dataset for Muscle Fatigue Analysis in Biceps\EMG Dataset for Muscle Fatigue Analysis in Biceps\ManualSegmentation.xlsx"
df_seg = pd.read_excel(seg_file)
print("\nFile: ManualSegmentation.xlsx")
print(f"Shape: {df_seg.shape}")
print(f"Columns: {df_seg.columns.tolist()}")
print(df_seg.head())
