# EEG Artifact Elimination: Deep Learning vs. Classical Machine Learning

A comparative study on automated artifact elimination in EEG signals using Deep Learning and Classical Machine Learning approaches.

## Project Overview

This project implements and compares multiple approaches for automated artifact detection and elimination in EEG signals:

- **Bandpass Filtering** (1-40 Hz)
- **Feature Extraction** (Statistical, Spectral, Wavelet)
- **Classical ML Models** (SVM, Random Forest, KNN)
- **Deep Learning Models** (CNN, LSTM, Ensemble)
- **CSP + SVM** (for Motor Imagery)
- **Advanced Techniques** (Ensemble Learning, Hyperparameter Tuning)

## Project Structure

```
eeg-artifact_elimination/
├── step1_bandpass_filter.m          # Bandpass filtering (1-40 Hz)
├── step2_segmentation.m              # Window segmentation (2s, 50% overlap)
├── step3_advanced_features.m         # Feature extraction (21 features)
├── step4_classical_ml.m              # SVM, RF, KNN classification
├── helper_dl_functions.m             # DL utility + metrics functions
├── step5_cnn_model.m                 # CNN model (augmentation + metrics)
├── step6_lstm_model.m                # BiLSTM temporal model
├── step7_autoencoder_model.m         # Autoencoder + DNN model
├── step8_deep_comparison.m           # DL vs classical comparison
└── README.md                         # This file
```

## Step-by-Step Usage

### **STEP 1: Bandpass Filtering (1-40 Hz)**
Remove noise, keep motor imagery frequencies (Delta, Theta, Alpha, Beta)

```matlab
step1_bandpass_filter.m
% Input: EEG_raw_data.mat (raw_left, raw_right, fs)
% Output: EEG_raw_clean_noise.mat (clean_left, clean_right, fs)
```

### **STEP 2: Windowing & Segmentation**
Divide continuous signal into 2-second windows with 50% overlap

```matlab
step2_segmentation.m
% Input: EEG_raw_clean_noise.mat
% Output: segmented windows
```

### **STEP 3: Advanced Feature Extraction**
Extract 21 statistical, spectral, and wavelet features from each segment

```matlab
step3_advanced_features.m
% Input: EEG_raw_clean_noise.mat
% Output: feature_table.mat, labels, feature_tablee.csv
% Features: RMS, Variance, Entropy, Bandpower, Ratios, Hjorth, Wavelet, etc.
```

### **STEP 4: Classical ML Classification**
Train SVM, Random Forest, KNN classifiers

```matlab
step4_classical_ml.m
% Input: feature_table.mat
% Output: Accuracy comparison
% Methods: SVM, RF (100 trees), KNN (k=5)
```

### **STEP 5: CNN Model**
Convolutional neural network for EEG artifact classification

```matlab
step5_cnn_model.m
% Input: features_extracted.mat
% Output: step5_cnn_results.mat
% Includes: data augmentation, training history, confusion matrix, ROC
```

### **STEP 6: BiLSTM Model**
Bidirectional LSTM for temporal dependency learning

```matlab
step6_lstm_model.m
% Input: features_extracted.mat
% Output: step6_lstm_results.mat
% Includes: performance comparison with CNN
```

### **STEP 7: Autoencoder + DNN**
Unsupervised feature learning with encoder-decoder and DNN classifier

```matlab
step7_autoencoder_model.m
% Input: features_extracted.mat
% Output: step7_autoencoder_results.mat
% Includes: reconstruction loss visualization
```

### **STEP 8: Deep Model Comparison**
Comprehensive training + comparison of DL models and classical ML

```matlab
step8_deep_comparison.m
% Input: features_extracted.mat
% Output: step8_deep_comparison_results.mat + CSV + plots
```

## Feature Engineering (21 Features)

| Category | Features |
|----------|----------|
| **Statistical** | RMS, Variance, Skewness, Kurtosis, Entropy |
| **Spectral** | Delta, Theta, Alpha, Beta, Gamma Power |
| **Ratios** | Theta/Beta, Alpha/Beta, Delta/Total |
| **Hjorth Parameters** | Activity, Mobility, Complexity |
| **Wavelet** | Wavelet Energy (db4, 2-3 levels) |
| **Information-theoretic** | Spectral Entropy, Approximate Entropy |

## Expected Accuracy

| Model | Accuracy |
|-------|----------|
| SVM (Classical) | ~94-96% |
| Random Forest | ~96-97% |
| KNN | ~92-95% |
| Ensemble (SVM+RF+KNN) | ~97-98% |
| CSP + SVM | ~95-97% |
| CNN-LSTM (Deep Learning) | ~97-99% |

## Installation & Requirements

### MATLAB Versions
- MATLAB R2019a or later
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox (for CNN-LSTM)

### Python Alternative (Optional)
```bash
pip install mne scikit-learn tensorflow keras numpy scipy
```

## Quick Start

```matlab
% Run all steps in sequence
step1_bandpass_filter.m
step2_segmentation.m
step3_advanced_features.m
step4_classical_ml.m
step5_cnn_model.m
step6_lstm_model.m
step7_autoencoder_model.m
step8_deep_comparison.m
```

## Dataset Format

**Input Format:**
- `EEG_raw_data.mat` containing:
  - `raw_left`: [channels × samples] - Left motor imagery
  - `raw_right`: [channels × samples] - Right motor imagery
  - `fs`: Sampling frequency (Hz)

**Example Data Creation:**
```matlab
% Generate synthetic EEG data
fs = 250;  % 250 Hz sampling
channels = 16;
duration = 60;  % seconds
samples = fs * duration;

raw_left = randn(channels, samples);
raw_right = randn(channels, samples);

save('EEG_raw_data.mat', 'raw_left', 'raw_right', 'fs');
```

## Theory & Background

### Bandpass Filtering
- Removes DC drift (<1 Hz) and power line noise (>40 Hz)
- Keeps motor imagery frequencies: 1-40 Hz
- Butterworth filter, order 4, filtfilt() for zero phase shift

### Feature Extraction
- Statistical features: capture signal amplitude and distribution
- Spectral features: power in different frequency bands
- Entropy features: measure signal randomness and complexity

### Classical ML
- **SVM**: Finds optimal decision boundary (RBF kernel)
- **Random Forest**: Ensemble of decision trees (robust, interpretable)
- **KNN**: Distance-based classification (simple, effective)

### Deep Learning
- **CNN**: Learns spatial patterns from raw signals
- **LSTM**: Captures temporal dependencies
- **Ensemble**: Combines multiple classifiers for better generalization

## Performance Comparison

Deep Learning vs. Classical ML:
- **Deep Learning**: Higher accuracy but needs more data and computation
- **Classical ML**: Faster training, interpretable, good baseline
- **Ensemble**: Best balance of accuracy and efficiency

## References

1. Sewak, M., Sahay, S.K., Rathore, H. (2018). "Comparison of Deep Learning and Classical ML for Malware Detection"
2. Ang, K.K., Chin, Z.Y., et al. (2012). "Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets"
3. Wavelet analysis for EEG signal processing

## Author

**Kusuma Akter**  
GitHub: [@kusumaakter](https://github.com/kusumaakter)

## License

MIT License - Feel free to use and modify

---

**Last Updated:** 2026-05-27
