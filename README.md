# Quantum Kernel-Based Battery Anomaly Detection

**Early-Stage Degradation Detection in Lithium-Ion Batteries using Quantum One-Class Support Vector Machines**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.2.4-purple.svg)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository presents a novel approach to battery health monitoring using **quantum kernel methods** combined with one-class support vector machines (OC-SVM) for anomaly detection. The framework demonstrates early detection of battery degradation with significant lead-time before reaching the 80% capacity threshold, trained on only nominal (healthy) battery cycles.

### Key Highlights

- **Quantum Feature Mapping**: ZZ/Pauli entangling circuit with depth-2, 8-qubit encoding
- **Early Detection**: Quantum kernel achieves **944-cycle lead-time** (first alarm at cycle 21, 98.78% capacity remaining)
- **One-Class Learning**: Trained exclusively on first 20 nominal cycles (ν=0.05)
- **Comprehensive Benchmarking**: Comparison against RBF, Laplacian, and Polynomial kernels
- **Real Battery Data**: Analysis of 1,241 discharge cycles from lithium-ion cell degradation

---

## Methodology

### Phase 1: Data Processing & Segmentation

- **Input**: Raw current-voltage time-series (22.7M samples, 1 Hz)
- **Segmentation**: Automated discharge cycle identification using current polarity
- **Filtering**: Minimum duration (1,800s) and magnitude (0.1A) thresholds
- **Output**: 1,241 validated discharge cycles

### Phase 2: Feature Engineering

Extracted 8 physical features per cycle:

| Feature          | Description                          | Unit  |
|------------------|--------------------------------------|-------|
| `capacity_Ah`    | Discharge capacity                   | Ah    |
| `energy_Wh`      | Discharge energy                     | Wh    |
| `duration_s`     | Cycle duration                       | s     |
| `v_min`          | Minimum voltage                      | V     |
| `v_max`          | Maximum voltage                      | V     |
| `v_mean`         | Mean voltage                         | V     |
| `i_rms`          | RMS current                          | A     |
| `dVdt_abs_mean`  | Mean absolute voltage rate of change| V/s   |

### Phase 3: Quantum Kernel Training

1. **Feature Scaling**: Min-max normalization to [0, π] using training statistics
2. **Quantum Feature Map**:
   - 8-qubit parameterized circuit
   - ZZ entangling gates with nearest-neighbor + wrap-around topology
   - Pauli rotations (RZ, RY) encoding
   - Circuit depth: 52, Total gates: 80
3. **Kernel Computation**: K(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|² via statevector simulation
4. **OC-SVM Training**: ν-OCSVM with precomputed kernel, ν=0.05
5. **Threshold Calibration**: 5th percentile on training scores (5% FPR, anomaly when score < threshold)

### Phase 4: Evaluation & Visualization

- **Lead-Time Analysis**: Time to first anomaly vs. 80%-capacity threshold (C80)
- **AUROC**: Early vs. Late cycle discrimination
- **Visualizations**: Capacity curves, anomaly scores, kernel heatmaps, kernel-PCA projections

---

## Results

### Performance Summary

| Model       | First Alarm | Lead-Time | AUROC  | Support Vectors |
|-------------|-------------|-----------|--------|-----------------|
| **Quantum** | **Cycle 21** | **944** | 0.4954 | 19              |
| RBF         | Cycle 21    | 944       | 0.5756 | 7               |
| Laplacian   | Cycle 21    | 944       | **0.9998** | 9           |
| Poly (d=2)  | Cycle 24    | 941       | 0.0005 | 2               |
| Poly (d=3)  | Cycle 24    | 941       | 0.9961 | 2               |

### Key Findings

- **Best Early Warning**: Quantum, RBF, and Laplacian kernels detect anomaly at cycle 21 (98.78% capacity), 944 cycles before C80
- **High Discriminative Power**: Laplacian kernel achieves 0.9998 AUROC for early/late separation
- **Comparable Performance**: Quantum, RBF, and Laplacian kernels all provide early warning with similar lead times
- **Training Efficiency**: Only 20 nominal cycles required for effective anomaly detection

### Reference Metrics

- **C_ref** (cycle 1): 3.270 Ah
- **C80 threshold**: 2.616 Ah
- **80%-capacity cycle**: 965
- **Total cycles analyzed**: 1,241

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/1bytess/quantum-ocsvm-battery-anomaly-detection.git
   cd quantum-ocsvm-battery-anomaly-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the raw data file**:

   The raw battery cycling data file (`data/data.csv`, 387.89 MB) is not included in this repository due to size constraints.

   **Options to obtain the data:**
   - Contact the author at [project@ezrahernowo.com](mailto:project@ezrahernowo.com) for access
   - Use publicly available battery datasets (e.g., [NASA Prognostics Center](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository), [Battery Archive](https://batteryarchive.org/))
   - The data should be a CSV file with two columns: `Current` (A) and `Voltage` (V) sampled at 1 Hz

   Place the data file at: `data/data.csv`

### Dependencies

```
qiskit==1.2.4
qiskit-machine-learning==0.8.0
qiskit-aer==0.15.1
scikit-learn==1.5.2
numpy==1.26.4
pandas==2.2.3
matplotlib==3.9.2
seaborn==0.13.2
scipy==1.14.1
jupyter==1.1.1
notebook==7.2.2
```

---

## Usage

### Running the Pipeline

Execute the Jupyter notebooks sequentially:

```bash
jupyter notebook
```

Then run in order:

1. **`phase_1_data_processing_segmentation.ipynb`**
   - Loads raw battery data from `data/data.csv`
   - Segments discharge cycles
   - Saves cycles to `result/phase_1/data/cycles.pkl`

2. **`phase_2_feature_engineering.ipynb`**
   - Computes 8 features per cycle
   - Saves feature matrix to `result/phase_2/data/features.csv`

3. **`phase_3_load_feature_select_training_cycles.ipynb`**
   - Trains quantum and baseline OC-SVM models
   - Computes kernel matrices
   - Calibrates detection thresholds
   - Saves models to `result/phase_3/data/`

4. **`phase_4_analysis_visualization.ipynb`**
   - Scores all cycles
   - Computes lead-time and AUROC
   - Generates plots and metrics
   - Saves outputs to `result/phase_4/`

### Expected Outputs

- **Plots**: `result/phase_4/plot/`
  - `capacity_vs_cycle.png`: Battery degradation curve
  - `anomaly_score_vs_cycle_quantum_vs_rbf.png`: Score evolution
  - `gram_heatmap_quantum.png`, `gram_heatmap_rbf.png`: Kernel structure
  - `kernel_pca_scatter_quantum.png`, `kernel_pca_scatter_rbf.png`: 2D projections

- **Metrics**: `result/phase_4/data/`
  - `metrics_summary.csv`: Performance comparison
  - `first_alarms.json`: Detection timing details

---

## Project Structure

```
quantum-ocsvm-battery-anomaly-detection/
├── data/
│   └── data.csv                                      # Raw battery time-series data
├── result/
│   ├── phase_1/
│   │   ├── data/
│   │   │   ├── cycles.pkl                            # Segmented discharge cycles
│   │   │   └── summary.csv                           # Cycle statistics
│   │   └── plot/                                     # Phase 1 visualizations
│   ├── phase_2/
│   │   └── data/
│   │       └── features.csv                          # Engineered features
│   ├── phase_3/
│   │   └── data/
│   │       ├── scaler.pkl                            # Feature scaler
│   │       ├── scaling_params.json                   # Scaling parameters
│   │       ├── quantum_kernel_params.json            # Quantum kernel configuration
│   │       ├── baseline_kernel_params.json           # Baseline kernel parameters
│   │       ├── K_quantum_train.npy                   # Quantum kernel (train)
│   │       ├── K_quantum_full.npy                    # Quantum kernel (full)
│   │       ├── K_rbf_*.npy, K_laplacian_*.npy, ...  # Baseline kernels
│   │       ├── ocsvm_quantum.pkl                     # Trained quantum OC-SVM
│   │       ├── ocsvm_rbf.pkl, ...                    # Baseline models
│   │       ├── thresholds.json                       # Calibrated thresholds
│   │       ├── threshold_summary.json                # Threshold calibration details
│   │       └── train_scores.pkl                      # Training scores
│   └── phase_4/
│       ├── data/
│       │   ├── metrics_summary.csv                   # Performance metrics
│       │   ├── metrics_summary_enhanced.csv          # Enhanced metrics
│       │   └── first_alarms.json                     # Detection results
│       └── plot/                                     # Visualization outputs
│           ├── capacity_vs_cycle.png
│           ├── anomaly_score_vs_cycle_quantum_vs_rbf.png
│           ├── gram_heatmap_quantum.png
│           ├── gram_heatmap_rbf.png
│           ├── kernel_pca_scatter_quantum.png
│           └── kernel_pca_scatter_rbf.png
├── phase_1_data_processing_segmentation.ipynb
├── phase_2_feature_engineering.ipynb
├── phase_3_load_feature_select_training_cycles.ipynb
├── phase_4_analysis_visualization.ipynb
├── requirements.txt
└── README.md
```

---

## Technical Details

### Quantum Feature Map

The quantum encoding circuit applies:

```
For each layer d ∈ {1, 2}:
  1. Pauli rotations: RZ(xᵢ), RY(xᵢ) for each qubit i
  2. ZZ entanglement: CNOT-RZ(2(xᵢ - xⱼ))-CNOT for neighboring qubits (i, j)
  3. Wrap-around coupling for periodic boundary
```

- **Hilbert Space Dimension**: 2⁸ = 256
- **Parameter Count**: 8 (one per feature)
- **Simulation**: Exact statevector (Qiskit AerSimulator)

### Anomaly Detection Convention

- **Score Interpretation**: Higher score = more nominal (closer to training distribution), Lower score = more anomalous
- **Anomaly Flag**: Triggered when `score < threshold` (5th percentile)
- **Training FPR**: Calibrated to 5% on nominal cycles

### Baseline Kernels

- **RBF**: k(x, x') = exp(-γ||x - x'||²), γ = 1/(2 × median²) (median heuristic)
- **Laplacian**: k(x, x') = exp(-γ||x - x'||₁)
- **Polynomial**: k(x, x') = (⟨x, x'⟩ + c)^d, d ∈ {2, 3}, c = 1

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ezrah2024quantum,
  author = {H, Ezra},
  title = {Quantum Kernel-Based Battery Anomaly Detection},
  year = {2024},
  url = {https://github.com/1bytess/quantum-ocsvm-battery-anomaly-detection}
}
```

---

## Author

**Ezra H**
Email: [project@ezrahernowo.com](mailto:project@ezrahernowo.com)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Battery cycling data sourced from NASA Prognostics Data Repository or similar public datasets
- Quantum circuits implemented using IBM Qiskit framework
- Anomaly detection framework built on scikit-learn OC-SVM

---

## Future Work

- [ ] Real-time deployment on edge devices
- [ ] Hardware quantum processor evaluation (IBM Quantum, Rigetti)
- [ ] Multi-cell battery pack anomaly detection
- [ ] Hybrid quantum-classical kernel optimization
- [ ] Transfer learning across different battery chemistries

---

**Last Updated**: October 2025
