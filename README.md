# MOGA-FCM Cognitive Load Detection

A research-oriented unsupervised machine-learning project for discovering cognitive-load prototypes from multimodal physiological driving data. The project combines NSGA-III multi-objective feature selection, Bayesian-optimized Fuzzy C-Means clustering, physiological validation, distribution-shift analysis, and cross-driver transferability testing on AffectiveROAD-derived feature tables.

The repository is packaged as a GitHub-ready research project. It retains processed feature tables, generated result artifacts, Pareto/convergence outputs, validation summaries, and selected figures. Raw AffectiveROAD wearable-data archives are excluded because they are large and may be governed by separate dataset usage terms.

## Research Motivation

Driver cognitive load is a latent physiological state rather than a directly observed label. In naturalistic driving, workload-related changes can appear through heart-rate variability, electrodermal activity, motion, respiration, temperature, and behavioral covariates, but those signals are noisy, subject-dependent, and only partially aligned with subjective labels.

This project treats cognitive-load discovery as an unsupervised multi-objective optimization problem. Instead of assuming fixed labels or a single physiological marker, it searches for compact feature subsets that induce stable fuzzy clusters while preserving physiological interpretability, cross-driver transferability, and validation evidence.

The core research questions are:

1. Can multi-objective evolutionary search identify compact physiological feature subsets that support stable unsupervised workload-state discovery?
2. Can Fuzzy C-Means represent overlapping cognitive states more naturally than hard clustering assignments?
3. Do the discovered workload prototypes generalize across drivers without driver-specific retraining?
4. How do objective autonomic signatures diverge from subjective or behaviorally mediated labels?

## Methodological Pipeline

```text
AffectiveROAD-derived physiological features
        |
        v
Feature normalization and validation-label alignment
        |
        v
NSGA-III multi-objective feature subset optimization
        |
        v
Pareto-front selection and feature-importance analysis
        |
        v
Bayesian-optimized Fuzzy C-Means clustering
        |
        v
SVBI/RMSSD physiological validation
        |
        v
MRT/ACC dissociation and modality-level analysis
        |
        v
Cross-driver transferability and MMD distribution-shift analysis
        |
        v
Result tables, convergence plots, validation figures, and driver summaries
```

## Technical Contributions

- NSGA-III multi-objective feature selection over high-dimensional physiological feature spaces.
- Optimization objectives combining silhouette score, within-cluster distance, between-cluster distance, mutual information, and feature-count pressure.
- Bayesian-optimized Fuzzy C-Means for selecting cluster configuration and fuzziness parameters.
- Fuzzy prototype discovery for overlapping low, medium, and high cognitive-load states.
- SVBI/RMSSD validation to connect discovered clusters with autonomic physiology.
- MRT/ACC dissociation analysis to separate objective autonomic state structure from behaviorally mediated subjective labels.
- Maximum Mean Discrepancy analysis for cross-driver distribution shift.
- Cross-driver transferability tests to evaluate whether selected physiological subspaces generalize beyond individual drivers.
- Pareto-front, convergence, modality-importance, and transferability visualizations.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── affectiveroad_preprocess.py
├── part1_affectiveroad.py
├── part3_affectiveroad.py
├── plot_convergence_affectiveroad.py
├── affective_normalized.csv
├── affective_gt_labels.csv
├── ga_results/
├── results/
└── plots/
```

## Main Components

| File or Folder | Purpose |
|---|---|
| `affectiveroad_preprocess.py` | Optional preprocessing script for local raw AffectiveROAD data |
| `part1_affectiveroad.py` | NSGA-III feature selection, Pareto search, and per-driver optimization artifacts |
| `part3_affectiveroad.py` | Bayesian-optimized FCM validation, physiological interpretation, transferability, and aggregate analysis |
| `plot_convergence_affectiveroad.py` | Convergence, Pareto, validation, modality-importance, and transferability figure generation |
| `affective_normalized.csv` | Processed feature matrix retained for reproducible project inspection |
| `affective_gt_labels.csv` | Validation/ground-truth alignment table used by downstream analysis |
| `ga_results/` | GA summaries, Pareto fronts, selected features, convergence logs, and feature importances |
| `results/` | Final CSV tables, driver summaries, transferability outputs, and validation artifacts |
| `plots/` | Driver-level, convergence, SVBI, MMD, and transferability figures |

## Key Outputs

| Output | Description |
|---|---|
| `results/Table_A_Clustering.csv` | Main clustering metrics and raw/latent silhouette comparison |
| `results/Table_C_GTValidation.csv` | Ground-truth alignment, ARI, NMI, purity, and Hungarian accuracy |
| `results/Table_D_Transferability.csv` | Cross-driver transferability and SVBI summary |
| `results/Table_E_MRT_Validation.csv` | Modality-level MRT/ACC dissociation analysis |
| `results/final_driver_summary.csv` | Per-driver selected features, cluster metrics, and validation summary |
| `results/mmd_distribution_shift.csv` | MMD-based distribution-shift measurements |
| `ga_results/ngen90_pop150/` | Per-driver convergence traces, Pareto artifacts, and selected feature subsets |
| `plots/convergence/` | Convergence, Pareto, feature-importance, transferability, MMD, and SVBI plots |

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main stages from processed feature tables:

```bash
python part1_affectiveroad.py
python part3_affectiveroad.py
python plot_convergence_affectiveroad.py
```

If using raw AffectiveROAD data locally, keep the raw files outside the repository or in an ignored `AffectiveROAD_Data/` folder, then run:

```bash
python affectiveroad_preprocess.py
```

## Dataset Note

Raw AffectiveROAD archives are excluded. The retained processed CSV files and generated outputs allow the project to be inspected without pushing bulky wearable-signal archives to GitHub.

## Scope

This repository focuses only on the MOGA-FCM cognitive-load detection research project, processed features, generated result artifacts, and selected plots. Raw archives, caches, temporary files, and unrelated local artifacts are excluded.

## Data
The normalized driving-telemetry table (`affective_normalized.csv`) is derived from the third-party AffectiveROAD
dataset and is not redistributed here. Obtain the original data from its authors and place the normalized file
in the repository root to reproduce the results.
