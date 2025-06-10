
# 📊 Determination of the Invisible Width of the Z Boson using CMS Open Data

This repository contains the code and documentation for an analysis performed to estimate the **invisible decay width of the Z boson**, using public 13 TeV proton-proton collision data collected by the CMS experiment in 2016 and made available through the [CERN Open Data Portal](https://opendata.cern.ch).

---

## Analysis Objective

The goal of this analysis is to estimate the decay width of the invisible channel Z → νν̄ from Z(νν̄ ) + jets events. These events are characterized by jets recoiling against an invisible system, represented by the missing transverse momentum vector **pTmiss**. The estimation is performed through a **simultaneous fit** that combines:

- **Signal region**: events with jets + pTmiss 
- **Control regions**:
  - Single Muon, Electron, Tau → to estimate the W → ℓν background
  - Double Muon, Electron → for the Z → ℓℓ reference channel
  - QCD-enriched region → for multijet background estimation

These control regions are used to compute **transfer factors**, allowing background extrapolation into the signal region.

---

## Methodology

### 1. Cut-based Selection

- Application of direct physics-based event selection criteria

### 2. Machine Learning with GNNs

- Supervised classification using **Graph Neural Networks (GNNs)**
- Each event is represented as a graph with 4 nodes (leading jet variables)
- Balancing strategy: **undersampling** + **SMOTE**
- Decision threshold: 0.6 to reduce false positives
- Evaluation metrics: Accuracy, F1-score, confusion matrix

### 3. Data Processing

- Event normalization using weights derived from integrated luminosity and cross section
- Corrections to the DYJetsToLL dataset
- Pileup reweighting

### 4. Statistical Fit

- **Simultaneous fit via MCMC**, based on:
  - Likelihood function combining signal and control regions
  - Gaussian priors for systematic uncertainties
  - Fitted parameters: r, r_Z, θ

---

## 📈 Results

- **Fitted scale factor**: 
  r = 1.272 (+0.076 / –0.073)

- **Estimated invisible width**: 
  Γ(Z → νν̄) = 634.6 (+44.5 / –43.3) MeV

- **Comparison with CMS Collaboration measurement** 
  This analysis yields a result approximately **2.4σ above** the official value reported by the CMS Collaboration:

  - **CMS (2022)**: 
    Γ(Z → νν̄) = 523 ± 16 MeV 
    📄 [arXiv:2206.07110](https://arxiv.org/abs/2206.07110)

  ⚠️ This discrepancy can largely be attributed to the statistical limitations of the dataset — which includes only the 2016 eras G and H as provided by CERN Open Data — and the limited Monte Carlo statistics in the double lepton control region, which affects the robustness of the r_Z transfer factor estimation.

---

## 📂 Repository Structure

```bash
ZinvWidth/
├── code/
│   ├── ZinvWidthAnalysis.ipynb          # Main notebook: applies weights, physical corrections, and computes Z invisible width
│   ├── Pileup_Correction_Example.ipynb  # Pileup correction example
│   ├── Event_Selection_*.py/.ipynb      # Region-specific selection scripts (Ptmiss, QCD, Single/Double Lepton) for data and MC
│   ├── geometry.py                      # Auxiliary geometry functions
│   └── dpoa_utilities.py                # Utilities for interacting with CERN Open Data
│
├── data/
│   ├── raw/                             # Lists of .root files from CERN Open Data, plus real pileup histogram
│
├── ml/
│   ├── code/                                    # Machine Learning scripts:
│   │   ├── GNN_Model.ipynb                      # GNN architecture implementation
│   │   ├── EventSelection_training_ML.py        # Signal training event selection
│   │   ├── EventSelection_Negative_training_ML.py # Background training event selection
│   │   ├── Event_Selection_validation_ML.ipynb  # Model-based validation event selection
│   │   ├── Event_Selection_cuts_Ptmiss_validation.ipynb # Cut-based event selection for comparison
│   │   └── CutsVsML.ipynb                       # ML vs cut-based selection comparison
│   ├── model/                # Trained model architecture and weights
│   ├── raw/                  # Raw .root file lists from CERN Open Data
│   └── plots/                # ML model output plots
│
├── plots/                    # Plots and figures from the final analysis
│
├── LICENSE.txt               # License information
├── README.md                 # This file
└── requirements.txt          # List of required Python packages
```
---

### 📎 Supplementary Documentation

📄 The **associated master's thesis**: https://doi.org/10.17181/sjdsz-79773

The document covers:

- The **theoretical background** on the Standard Model and invisible Z boson decay 
- A detailed description of the **methodology** 
- Technical aspects of the **GNN-based machine learning approach** 
- A **comprehensive discussion of the results**

---

### Author

This project was developed by **Maria Gabriela Gomes**. 
For more information or to connect, visit [Maria Gabriela Gomes on LinkedIn](https://www.linkedin.com/in/maria-gabriela-gomes-27097431b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app).
