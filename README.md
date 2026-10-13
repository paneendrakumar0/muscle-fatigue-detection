# Wearable Muscle Fatigue Detection via sEMG & IMU Sensor Fusion

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Validation Accuracy](https://img.shields.io/badge/Accuracy-98.64%25-brightgreen.svg)]()

An end-to-end multi-modal machine learning system and digital signal processing (DSP) pipeline designed to non-invasively detect and classify human neuromuscular fatigue in real time. The system fuses **Surface Electromyography (sEMG)** signals and **6-axis Inertial Measurement Unit (IMU)** kinematic tremor data from dynamic bicep exertion trials.

---

## 🚀 Key Highlights & Results

- **98.64% Mean Accuracy:** Validated on the real-world [Mendeley Biceps Dataset](https://data.mendeley.com/) across 30 distinct subjects.
- **Overcoming Inter-Subject Variability:** Employs **Dense Sliding Windows** (1.0s window, 0.95s overlap) paired with **Subject-Specific Baseline Normalization**, neutralizing anatomical differences across individuals (skin impedance, fat distribution, electrode placement).
- **Multi-Modal Sensor Fusion:** Combines electrical motor unit action potential (MUAP) degradation from sEMG with high-frequency mechanical tremor signatures from 3-axis accelerometers.
- **Edge AI / TinyML Ready:** Low-latency classification with time complexity $\mathcal{O}(M \cdot D)$, optimized for microcontroller deployment (ESP32 / ARM Cortex-M).
- **Dual Visual Interfaces:** Includes both an interactive Python **Streamlit** dashboard and a standalone modern **Web Dashboard** (`dashboard_web`).

---

## 🧠 Scientific & Mathematical Methodology

### 1. The Physiological Fatigue Mechanism
During sustained muscle contraction:
1. **Intracellular pH Drop & Conduction Velocity Slowing:** Accumulation of metabolic byproducts widens Motor Unit Action Potentials (MUAPs) in the time domain, which manifests as a **downward shift in median/mean frequency** in the frequency domain.
2. **Motor Unit Synchronization & Recruitment:** To sustain force, the central nervous system recruits additional, slower motor units, producing an **increase in time-domain amplitude (RMS/MAV)**.
3. **Kinematic Tremor Emergence:** As localized motor units undergo fatigue, smooth motor control degrades into distinct mechanical oscillations captured by IMU acceleration variance.

### 2. Feature Engineering Pipeline

| Domain | Feature | Mathematical Description / Purpose |
| :--- | :--- | :--- |
| **Time Domain** | **Root Mean Square (RMS)** | $\text{RMS} = \sqrt{\frac{1}{N}\sum_{n=1}^N x_n^2}$ (measures total electrical power) |
| | **Mean Absolute Value (MAV)** | $\text{MAV} = \frac{1}{N}\sum_{n=1}^N \|x_n\|$ |
| | **Zero-Crossing Rate (ZCR)** | Captures waveform frequency slowing without FFT overhead |
| | **Waveform Length (WL)** | Cumulative signal complexity and amplitude variation |
| **Frequency Domain** | **Welch's PSD** | Periodogram averaging to eliminate biological signal variance |
| | **Median Frequency (MDF)** | Frequency dividing spectral power into two equal halves |
| | **Mean Frequency (MNF)** | Centroid of the power spectral distribution |
| | **Spectral Entropy** | Quantifies spectral flattening vs. narrow-band fatigue concentration |
| **Kinematics (IMU)** | **Tremor Variance** | Variance of 3-axis acceleration magnitude vector $\sigma^2(\|\mathbf{a}\|)$ |
| | **Signal Magnitude Area (SMA)** | Normalized sum of acceleration integrals measuring mechanical exertion |

---

## 📁 Repository Structure

```text
├── EMG Dataset for Muscle Fatigue Analysis in Biceps/  # 30-Subject real-world benchmark dataset
├── dashboard_web/                                      # Standalone modern web dashboard UI
│   ├── index.html
│   ├── style.css
│   └── app.js
├── build_academic_report.py                           # 20+ page academic thesis PDF generator
├── dashboard.py                                       # Real-time Streamlit streaming dashboard
├── data_generation.py                                 # Synthetic signal generation & simulation
├── evaluation_and_inference.py                        # Model evaluation & cross-validation metrics
├── explore_dataset.py                                 # Dataset inspection & distribution analysis
├── feature_extraction.py                              # Signal processing and feature extraction modules
├── model_training.py                                  # Core ML training workflows
├── real_data_pipeline.py                              # Subject-specific baseline pipeline (98.64% accuracy)
├── Muscle_Fatigue_Extended_Academic_Thesis.pdf         # Comprehensive academic thesis report
├── requirements.txt                                   # Python environment dependencies
└── .gitignore                                         # Git ignore specifications
```

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/paneendrakumar0/muscle-fatigue-detection.git
cd muscle-fatigue-detection
```

### 2. Set Up Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Real Data Pipeline (98.64% Model)
```bash
python real_data_pipeline.py
```

### 4. Launch Real-Time Streamlit Dashboard
```bash
streamlit run dashboard.py
```

### 5. Open Web Dashboard
Simply open `dashboard_web/index.html` in any modern web browser to view live simulated streaming, fatigue gauges, and time-frequency charts.

---

## 📊 Documentation & Thesis
A publication-quality, 20+ page academic thesis PDF generated from this work is included directly in this repository:
- 📄 [Muscle_Fatigue_Extended_Academic_Thesis.pdf](Muscle_Fatigue_Extended_Academic_Thesis.pdf)

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
