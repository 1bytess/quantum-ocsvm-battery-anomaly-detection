## Introduction

After receiving the actual GUIDELINES.MD file, this addendum revises the earlier review of the Quantum-Kernel SVM for Battery Degradation project. The guidelines prescribe a strict phased notebook structure, specific feature definitions, and explicit outputs. Here we reassess each phase for adherence to these requirements and adjust the recommendations accordingly.

## 1. Alignment with the Provided Guidelines

### Features (Phase 2)

The guidelines list eight features to compute per discharge cycle:

1.  capacity_Ah - total charge removed in $h(\Sigma|I|\times dt/3600)$
2.  energy_Wh - discharge energy $(\begin{matrix}\Sigma~V\times|I|\times dt/3600)$.
3.  duration_s - length of the discharge in seconds.
4.  v_min, v_max v_mean - minimum, maximum and average voltage over the discharge.
5.  i_rms-root-mean-square current $(\sqrt{}mean(I^{2})$.
6.  |dV/dt_mean mean absolute voltage derivative (mean (AV|/At)

In my previous review I erroneously referenced features such as range_V, range_I and separate dI/ dt statistics. These are not part of the mandated feature set. Any code implementing additional features could deviate from the guidelines unless clearly marked as exploratory. Ensure the Phase 2 notebook computes exactly the features above and saves them to /result/phase_2/data/ features.csv with the specified column order.

### Phase Structure & File Management

The guidelines enforce a strict sequence of code and markdown cells across four phases (e.g., 1.1 Load Libraries, 1.2 Stream-Load CSV, ...). Each phase must finish with a markdown summary enumerating actions taken, file paths of saved artifacts, key statistics (e.g., number of cycles segmented, training N), and variables needed in the next phase. If the existing notebooks stray from this order or omit summaries, they violate the guidelines.

All outputs must reside under the /result/phase_#/... directory. Check that the code does not write to arbitrary paths (e.g., ./data or the repository root). Each artifact (like scaler.pkl or metrics_summary.csv) should be saved with the exact filenames specified.

### Model & Kernel Implementation (Phase 3)

The project objective requires a v-OCSVM using a simulated ZZ/Pauli feature map (depth 1-2) with 8 qubits to match the eight features. Classical baselines include RBF (with y computed via the median heuristic on the training data), Laplacian and polynomial kernels (degree 2-3). The guidelines emphasise precomputed kernels and calibrating thresholds to achieve 5% false positive rate on the nominal training data. Ensure the code fits the scaler using only the first $N=10-20$ nominal cycles, applies that transform to all cycles, computes quantum and classical kernels on the scaled data, trains v-OCSVM models, and calibrates thresholds accordingly.

### Evaluation & Metrics (Phase 4)

Lead-time and AUROC must be computed using the guideline definitions: lead-time is the difference between the 80% capacity cycle (first cycle where capacity_Ah ≤ 0.8 C_ref) and the first alarm cycle (earliest cycle after training with anomaly score beyond threshold). AUROC should use the 80% capacity split to define "Early" vs "Late." Required figures include capacity vs cycle, anomaly score vs cycle (Quantum vs RBF), Gram matrix heatmaps, and Kernel-PCA scatter plots. These should be saved under / result/phase_4/plot/.

## 2. Updated Recommendations & Corrections

1.  **Feature Computation** - Replace any non-guideline features (e.g., range_V, dI/dt | statistics) with the correct set listed above. Verify that energy_Wh and i_rms are implemented properly; energy computation must multiply voltage by absolute current and integrate over time.
2.  **Segmentation & Preprocessing** The segmentation logic remains as previously recommended (vectorised detection of continuous negative-current blocks with noise filtering). Ensure segmentation yields cycle boundaries saved as cycles.pkl and summary statistics. Confirm the Phase 1 notebook includes the eight mandated steps and ends with a markdown summary describing the segmentation and file paths.
3.  **Scaling** - Continue to fit the min-max scaler only on the first N cycles and map feature values into [0, π]. The guidelines specify this scaling as mandatory before kernel embedding. Robust scalers can still be explored as improvements but must be justified and, if used, multiply results by π to maintain the [0, π] range.
4.  **Model & Kernel Hyperparameters** - Validate that the code sets $v\approx0.05$, selects quantum circuit depth (1-2) consistently with eight qubits, and uses the median heuristic for RBF y. Additional tuning (grid search on y or v) can be performed but should not override the baseline specified values without analysis.
5.  **Threshold Calibration** - The guidelines are explicit about calibrating thresholds to achieve a 5% false positive rate on training data. Ensure the calibration uses the training anomaly scores (not labels) and that thresholds are saved in thresholds.json
6.  **File Naming & Summaries** - Audit each notebook to confirm it follows the specified sequence of cell titles (e.g., 2.1 Load Segmentation, 2.2 Define Feature Computations, ...). At the end of each phase, a markdown summary cell must list all saved artifacts (with paths like /result/phase_3/data/ ocsvm_quantum.pkl), key parameters (training N, scaler ranges, kernel depths), and next-phase inputs. Without this, the pipeline may be fragile and hinder reproducibility.
7.  **Metric Enhancements** The guidelines remain silent about advanced metrics. The additional evaluation metrics suggested previously (rolling AUROC, time-to-first-false-positive, alarm persistence, etc.) still hold; incorporate them as supplementary analysis beyond the mandated deliverables. They provide a more nuanced picture of detection stability and should be presented alongside the core lead-time and AUROC results.

## Conclusion

The GUIDELINES. MD file clarifies the precise expectations for data processing, feature engineering, model training, evaluation and output organisation. Any code within the repository should be audited against this specification: confirm that the eight prescribed features are used, the phased cell sequence and file structure are followed, and the v-OCSVM and kernel implementations respect the guideline hyperparameters and thresholds. With these corrections and the earlier recommendations (vectorisation, robust scaling, hyperparameter tuning and additional metrics), the project can yield a rigorous and reproducible quantum-kernel anomaly detection pipeline for early battery degradation.