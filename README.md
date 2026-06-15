<div align="center">

![Model Type](https://img.shields.io/badge/Model-1D--CNN-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-97.17%25-success)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange)
![Domain](https://img.shields.io/badge/Domain-Biomedical%20AI-red)

</div>
# Ventricula ML: Deep Learning for Clinical ECG Arrhythmia Classification

This project implements an end-to-end machine learning pipeline to automate the detection and classification of cardiac arrhythmias from single-lead ECG signals. Manual analysis of continuous ECG time-series data is incredibly time-consuming and prone to human fatigue. This pipeline solves that by using a deep learning architecture to extract spatial morphology features from raw biopotentials, providing instant, clinical-grade diagnostics.

## Results

After establishing the data pipeline and handling severe clinical class imbalances, a 1D Convolutional Neural Network (1D-CNN) was designed to act as the feature extractor. The network successfully learned the underlying biophysics of the waveforms without overfitting.

* **Winning Model:** 1D Convolutional Neural Network (1D-CNN)
* **Validation Accuracy:** 97.17%
* **Key Achievement:** Successfully isolated and identified critical anomalies, such as Premature Ventricular Contractions (PVCs), which are notoriously difficult to distinguish due to class imbalance.

## Dataset & Engineering

The model processes continuous cardiac electrical activity separated into individual heartbeat segments. Each sample is a temporal sequence representing the normalized amplitude of the heartbeat. 

| Feature / Target | Type | Description |
| :--- | :--- | :--- |
| **Time Steps (0-186)** | Temporal Array | 187 continuous voltage readings capturing the P-wave, QRS complex, and T-wave morphology. |
| **Class 0 (N)** | Target | Normal Beat |
| **Class 1 (S)** | Target | Supraventricular Premature Beat |
| **Class 2 (V)** | Target | Premature Ventricular Contraction |
| **Class 3 (F)** | Target | Fusion Beat |
| **Class 4 (Q)** | Target | Unclassifiable Beat |

## Pipeline Architecture

This repository contains a full production-ready environment, moving far beyond a basic training script:
1.  **Preprocessing & Resampling:** Addressed heavy bias toward "Normal" beats using resampling techniques to ensure the AI learned minority classes (like Fusion Beats).
2.  **Model Training:** A 542k-parameter sequential network utilizing `Conv1D`, `BatchNormalization`, `MaxPooling1D`, and `Dropout` layers to prevent memorization.
3.  **Visual Diagnostic UI:** A custom visualization function built with `Matplotlib` that plots the electrical waveform, shades the area under the curve, and dynamically color-codes the AI's clinical decision against the actual truth.
4.  **Cloud Persistence:** Automated `.keras` weight exportation for instant inference without the need to retrain.
