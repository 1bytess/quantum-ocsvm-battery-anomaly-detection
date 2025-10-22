# **Project Guidelines: Quantum-Kernel SVM for Battery Degradation**

---

### **1. Project Objective**

Implement a **One-Class Quantum-Kernel SVM (ν-OCSVM, ν≈0.05)** to detect **early degradation** on a **single-cell, 1 Hz, full-life** dataset using **one sample per discharge cycle**. Train only on **early nominal cycles** and evaluate on subsequent cycles.

**Scope & Hypothesis**

* **Model:** ν-OCSVM with a **simulated ZZ/Pauli entangling feature map** (depth 1–2), **6–8 “qubits”** aligned to the feature dimension (here: **8**).
* **Data regime:** **One cell**; treat the **first N = 10–20 cycles** as nominal for training; test on all remaining cycles.
* **Features (8 dims):**
  `capacity_Ah, energy_Wh, duration_s, v_min, v_max, v_mean, i_rms, |dV/dt|_mean`
  **All features must be scaled to** ([0,\pi]) **before kernel embedding**.
* **Baselines:** RBF (**γ via median heuristic**), Laplacian, Polynomial (degree **2–3**).
* **Primary metrics:**

  * **Lead-time:** cycles between the **first alarm** (first anomaly detected after training range) and the **80%-capacity cycle**.
  * **AUROC:** “Early” vs “Late” cycle discrimination using model scores.
  * **FPR control:** **5%** on nominal **training** data (threshold calibration).
* **Required figures (saved in Phase 4):**

  1. **Capacity vs. cycle** (mark 80%-capacity cycle).
  2. **Anomaly score vs. cycle** (**Quantum vs. RBF**), with threshold line(s) and first-alarm markers.
  3. **Kernel Gram matrix heatmaps** (**Quantum vs. RBF**) showing early/late block structure.
  4. **Kernel-PCA 2-D scatter** of cycles (colored by cycle index).

---

### **2. Dataset Information**

Use the **ESCL, Chungnam National University** single-cell, full-life dataset.

* **Source:** Single cell, continuous cycling **to End-of-Life** (EoL).
* **File:** Single `.csv`, **no header row**.
* **Columns (2):**
  `Column 1 = Current (A)`
  `Column 2 = Voltage (V)`
* **Sampling:** 1 Hz (implicit `dt = 1 s`).
* **Size:** **22,714,175 rows**.
* **Data characteristics:**

  * **Zero current samples:** 10,145,126 (**44.66%**)
  * **Positive current samples:** 8,564,265
  * **Negative current samples:** 4,004,784
  * **Current min/max:** **−3.2257 A** / **+3.2291 A**
  * **Mean (non-zero current):** **0.0058 A**

---

### **3. Project Structure (Phased Notebooks)**

The project is executed as **four Jupyter Notebooks**, each producing **saved artifacts** for the next phase. **Do not skip phases.** Each phase ends with a **markdown summary cell** (see §7).

#### **Phase 1 — Data Preprocessing & Cycle Segmentation**

**Goal:** Load the 1 Hz raw stream, assign headers, segment into **individual discharge cycles**, and persist the segmentation.

**High-level tasks**

1. Load raw `.csv` (no header) and assign `Current`, `Voltage`.
2. Basic sanity checks (row counts; zero/± current counts) to confirm integrity.
3. **Cycle segmentation** (discharge-focused): identify continuous **discharge blocks** (negative current), using 1 Hz continuity and appropriate minimal duration and magnitude criteria to avoid noise.
4. Save results:

   * `/result/phase_1/data/cycles.pkl` — list of discharge cycle boundaries and any metadata (e.g., `(start_idx, end_idx)` for each discharge).
   * `/result/phase_1/data/summary.csv` — optional sanity statistics (counts per cycle, durations, etc.).

#### **Phase 2 — Feature Engineering**

**Goal:** Compute **one feature vector per discharge cycle** and compile to a single table.

**High-level tasks**

1. Load `cycles.pkl`, iterate **discharge cycles** only.
2. Compute features (per cycle; `dt=1 s`):

   * `capacity_Ah = (Σ |I| * dt) / 3600` over the discharge portion.
   * `energy_Wh = (Σ V * |I| * dt) / 3600` over the discharge.
   * `duration_s = (#samples in discharge) * dt`.
   * `v_min, v_max, v_mean` over the discharge.
   * `i_rms = sqrt(mean(I^2))` over the discharge.
   * `|dV/dt|_mean = mean(|ΔV/Δt|)` over the discharge (Δt = 1 s).
3. Save results:

   * `/result/phase_2/data/features.csv` with columns
     `cycle_idx, capacity_Ah, energy_Wh, duration_s, v_min, v_max, v_mean, i_rms, dVdt_abs_mean`.

#### **Phase 3 — Model Implementation & Training**

**Goal:** Define kernels, scale features to ([0,\pi]), **train** ν-OCSVM models on the **first N=20 cycles** as nominal, and **calibrate threshold** to 5% FPR.

**High-level tasks**

1. Load `/result/phase_2/data/features.csv`.
2. Select training cycles: indices **1..N** (default **N=20**). Treat **N∈[10,20]** as a tunable setup (start with 20).
3. **Scale to ([0,\pi])** per feature using **min–max scaling** computed **on the training subset**; apply the same transform to all cycles. Persist scaling parameters.
4. **Quantum kernel:** Simulated **ZZ/Pauli entangling feature map** (depth 1–2, **8 qubits** for 8 features). Compute **precomputed kernel matrix** for training and train+test sets.
5. **Baselines:**

   * **RBF**: γ via **median heuristic** (on training set pairwise distances).
   * **Laplacian** and **Polynomial (deg 2–3)** similarly prepared (precomputed or callable).
6. Train **ν-OCSVM** for **Quantum** and each baseline (use `kernel="precomputed"` where appropriate).
7. **Threshold calibration:** Using **training scores**, set anomaly threshold so **FPR = 5%** (e.g., 95th percentile of the training anomaly score definition you adopt).
8. Save artifacts:

   * `/result/phase_3/data/scaler.pkl`
   * `/result/phase_3/data/kernel_params.json` (quantum map depth, qubit count, classical kernel params incl. γ)
   * `/result/phase_3/data/ocsvm_quantum.pkl` and baseline model files
   * `/result/phase_3/data/thresholds.json` (per model)
   * (optional) `/result/phase_3/data/K_quantum_train.npy`, `/result/phase_3/data/K_quantum_full.npy`, etc.

#### **Phase 4 — Analysis & Visualization**

**Goal:** Score **all cycles**, compute **lead-time**, **AUROC**, and generate **all required figures**.

**High-level tasks**

1. Load features, scaler, kernels, trained models, and thresholds.
2. Produce **per-cycle anomaly scores** for **Quantum** and **RBF** (and other baselines).
3. Compute **C_ref** (capacity at **cycle 1**) and find **C80** = 0.8 × C_ref; identify the **80%-capacity cycle** (= first cycle where `capacity_Ah ≤ C80`).
4. **First alarm** per model: earliest **post-training** cycle with score beyond the calibrated threshold.
5. **Lead-time** per model: `(80%-capacity cycle) − (first-alarm cycle)`.
6. **AUROC:** define **Early** vs **Late** using the **80% capacity split** (`cycle ≤ 80% cycle` = Early; `>` = Late); compute **AUROC** from the continuous anomaly scores.
7. Generate & save all required figures:

   * `/result/phase_4/plot/capacity_vs_cycle.png`
   * `/result/phase_4/plot/anomaly_score_vs_cycle_quantum_vs_rbf.png`
   * `/result/phase_4/plot/gram_heatmap_quantum.png`, `/result/phase_4/plot/gram_heatmap_rbf.png`
   * `/result/phase_4/plot/kernel_pca_scatter_quantum.png`, `/result/phase_4/plot/kernel_pca_scatter_rbf.png`
8. Save metrics:

   * `/result/phase_4/data/metrics_summary.csv` with `model, first_alarm_cycle, c80_cycle, lead_time, auroc, fpr_train=0.05` (and any supporting stats).

---

### **4. STRICT Interaction Protocol (Mandatory)**

**Claude must follow these rules exactly:**

* **A. One Cell at a Time:** Provide code in **exactly one** code cell per response.
* **B. Wait for User Feedback:** After providing a code cell, **STOP**. The user will run it and return output (e.g., stdout, `.head()`, or errors).
* **C. Sequential & Contextual:** Generate the **next** code cell **only after** receiving the previous cell’s outputs, and **condition** it on those results.
* **D. No Code Artifacts:** **All code** must be inside a **markdown code block** in the chat. **Do not** use any artifact/file-generation features for code.
* **E. No Debug Jargon:** Do **not** add filler like “Fixed,” “Optimized,” “Updated,” etc. **Only** output the formatted cell.

---

### **5. STRICT Formatting Protocol (Mandatory)**

Every code response must use **exactly** this markdown structure:

```markdown
## **[Phase.Step] [Descriptive Title]**
```

```python
# Your Python code for this single step goes here.
# The code style must be simple and script-like.
```

**Example:**

```markdown
## **1.1. Load Libraries**
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

---

### **6. STRICT File Management (Mandatory)**

All outputs **must** be saved under this directory schema:

```
/result/
  phase_1/
    data/
  phase_2/
    data/
  phase_3/
    data/
  phase_4/
    data/
    plot/
```

* **Phase 1 example:** `/result/phase_1/data/cycles.pkl`
* **Phase 4 example:** `/result/phase_4/plot/capacity_vs_cycle.png`

**Do not** write outputs outside `/result/phase_#/…`.

---

### **7. Phase Summary (Mandatory)**

The **very last cell** of **each phase** must be a **markdown cell** that summarizes:

* **What was done** and **where data/models were saved** (list exact file paths).
* **Key figures/statistics** (e.g., number of cycles, min/max values used, N for training).
* **Key variables/parameters** to **carry forward** (e.g., `N`, scaler bounds, kernel depths/γ).
* **Next-phase inputs** (explicit filenames the next notebook will load).

---

## **Per-Phase Step Map (Exact Cell Sequence)**

Below is the **strict sequence of cells** (titles) that Claude must follow. Claude will emit **one cell** at a time, in order, and wait for the user after each.

### **Phase 1 — Data Preprocessing & Cycle Segmentation**

1. **1.1 Load Libraries & Create Output Dirs**
2. **1.2 Stream-Load Raw CSV (No Header) & Assign Columns**
3. **1.3 Basic Sanity Checks (Row Counts, Zero/± Current)**
4. **1.4 Current/Voltage Preview (Sample Rows & Descriptives)**
5. **1.5 Discharge Segmentation (Identify Continuous Negative-Current Blocks)**
6. **1.6 Post-Process Segments (Filter by Minimal Duration/Magnitude; Index Integrity)**
7. **1.7 Save Segmentation (`/result/phase_1/data/cycles.pkl`, `summary.csv`)**
8. **1.8 Phase 1 Summary (MARKDOWN ONLY)**

### **Phase 2 — Feature Engineering**

1. **2.1 Load Segmentation (`cycles.pkl`)**
2. **2.2 Define Per-Cycle Feature Computations (capacity, energy, etc.)**
3. **2.3 Compute Features for All Discharge Cycles**
4. **2.4 Assemble Feature Table & Basic QA (NaNs, Infs, Ranges)**
5. **2.5 Save Features (`/result/phase_2/data/features.csv`)**
6. **2.6 Phase 2 Summary (MARKDOWN ONLY)**

### **Phase 3 — Model Implementation & Training**

1. **3.1 Load Features & Select Training Cycles (1..N, N=20)**
2. **3.2 Scale Features to [0, π] Using Training Min–Max (Persist Scaler)**
3. **3.3 Build Quantum Feature Map (ZZ/Pauli, Depth 1–2, 8 Qubits) & Kernel (Precomputed)**
4. **3.4 Prepare Baseline Kernels (RBF γ=median heuristic; Laplacian; Poly deg 2–3)**
5. **3.5 Train ν-OCSVM for Quantum & Baselines**
6. **3.6 Calibrate Threshold (FPR=5% on Training) & Persist**
7. **3.7 Save Models, Kernels (optional), and Params**
8. **3.8 Phase 3 Summary (MARKDOWN ONLY)**

### **Phase 4 — Analysis & Visualization**

1. **4.1 Load Artifacts (Scaler, Models, Thresholds, Features)**
2. **4.2 Score All Cycles (Quantum & RBF & Others)**
3. **4.3 Compute 80%-Capacity Cycle (C80) and First-Alarm Cycle per Model**
4. **4.4 Lead-Time & AUROC Computation**
5. **4.5 Plot Capacity vs. Cycle (`capacity_vs_cycle.png`)**
6. **4.6 Plot Anomaly Score vs. Cycle (Quantum vs. RBF)**
7. **4.7 Plot Kernel Gram Matrix Heatmaps (Quantum & RBF)**
8. **4.8 Kernel-PCA 2-D Scatter (Quantum & RBF)**
9. **4.9 Save Metrics Table (`/result/phase_4/data/metrics_summary.csv`)**
10. **4.10 Phase 4 Summary (MARKDOWN ONLY)**

---

## **Operational Definitions (for Implementation Consistency)**

* **Sampling interval:** `dt = 1 s`.
* **Discharge cycle:** a continuous block where **current < 0 A** (apply reasonable noise and duration filters).
* **capacity_Ah:** ( \frac{\sum |I| \cdot dt}{3600} ) over the discharge block.
* **energy_Wh:** ( \frac{\sum V \cdot |I| \cdot dt}{3600} ) over the discharge block.
* **duration_s:** number of samples in the discharge block.
* **i_rms:** ( \sqrt{\text{mean}(I^2)} ) over the discharge block.
* **(|dV/dt|_mean):** mean absolute first difference of voltage per second over the discharge block.
* **Scaling to ([0,\pi]):** per-feature min–max (from **training subset**) mapped linearly into ([0,\pi]); apply the same transform to all cycles.
* **C_ref:** capacity at **cycle 1**.
* **80%-capacity cycle:** first cycle where `capacity_Ah ≤ 0.8 × C_ref`.
* **First alarm (per model):** earliest **post-training** cycle whose score crosses that model’s **calibrated** threshold.
* **Lead-time:** `(80%-capacity cycle) − (first-alarm cycle)`.
* **AUROC split:** Early = cycles `≤ 80%-capacity cycle`; Late = `>` that cycle.

---

## **Deliverables Checklist**

* **Phase 1:** `/result/phase_1/data/cycles.pkl`, `summary.csv`, Phase 1 summary (markdown).
* **Phase 2:** `/result/phase_2/data/features.csv`, Phase 2 summary (markdown).
* **Phase 3:** scaler/model/kernel params & thresholds under `/result/phase_3/data/`, Phase 3 summary (markdown).
* **Phase 4:** required plots under `/result/phase_4/plot/`, metrics table `/result/phase_4/data/metrics_summary.csv`, Phase 4 summary (markdown).

---

## **Compliance Notes (Claude must adhere)**

* **One code cell per response.**
* **Wait after each cell** for the user’s outputs before proceeding.
* **All code** must appear **directly** in a markdown code block (no artifacts).
* **No filler/debug chatter.** Only the titled section and code cell.
* **All outputs** must be saved strictly under `/result/phase_#/…`.
* **Every phase ends** with a **markdown-only** summary cell enumerating saved files and next-phase inputs.

**End of Guidelines.**
