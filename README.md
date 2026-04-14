# ml-diagnostics-production-drift
Production drift detection framework for ML systems. Complete code for 6 diagnostics (PSI, KS/Wasserstein, ECE, rolling performance, SHAP attribution, drift robustness) + lightweight 4-gate monitoring system. Reproducible fraud detection example with progressive drift. Part 3 of ML Diagnostics Mastery series.

# ML Diagnostics Mastery — Part 3: Production Drift

**How to Detect and Fix Production Drift in Machine Learning (Complete Guide)**

Why models that pass every diagnostic still fail three months after deployment — and how to build a lightweight monitoring system that catches distribution shift before your stakeholders do.

This repository contains the full reproducible code for the article published on EmiTechLogic (April 14, 2026).

**Part of the ML Diagnostics Mastery Series:**
- [Part 1: Overfitting, Instability & Data Leakage](https://emitechlogic.com/how-to-diagnose-overfitting-in-machine-learning-9-proven-tools/) → [companion repo](https://github.com/Emmimal/ml-diagnostics-overfitting)
- [Part 2: Class Imbalance, Threshold Tuning & Calibration](https://emitechlogic.com/how-to-diagnose-and-fix-class-imbalance-in-machine-learning-complete-guide/)
- **Part 3: Production Drift & Monitoring Systems ([You Are Here](https://emitechlogic.com/how-to-detect-and-fix-production-drift-in-machine-learning-complete-guide/))**

## What This Repository Covers

- Synthetic fraud detection dataset with progressive **covariate shift**, **label shift**, and mild **concept drift** over 6 simulated production months.
- **Diagnostic 1**: Population Stability Index (PSI) for input distributions (with standard thresholds: <0.10 stable, 0.10–0.25 warning, >0.25 action).
- **Diagnostic 2**: Score distribution monitoring using KS statistic and Wasserstein distance.
- **Diagnostic 3**: Label drift & ground truth lag using Expected Calibration Error (ECE) and chargeback proxy.
- **Diagnostic 4**: Rolling performance at the fixed deployed threshold (including flag rate constraint and re-thresholding).
- **Diagnostic 5**: SHAP drift attribution to localize which features drive behavioral change.
- **Diagnostic 6**: Drift robustness comparison across 5 models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting).
- **Lightweight 4-Gate Monitoring System** with alert levels (GREEN / AMBER / RED) and clear escalation logic.
- The "Five Decisions" checklist for production ML systems.
- All figures and the final monitoring report exactly as shown in the article.

## Quick Start

```bash
git clone https://github.com/Emmimal/ml-diagnostics-production-drift.git
cd ml-diagnostics-production-drift
```
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Run the full diagnostics (generates all figures + monitoring report)
python production_drift_diagnostics.py

# FAST_MODE=True by default → all 6 figures render in <90 seconds.
Set FAST_MODE=False for more exhaustive threshold sweeps (identical conclusions).
Outputs (including 6 high-quality figures) are saved to the outputs/ folder.

# Key Features & Implementation Notes

- Fully reproducible with random_state=42.
- Synthetic dataset generated via sklearn.datasets.make_classification (6% baseline fraud rate).
- Cost structure: $10 per false positive, $500 per missed fraud.
- SHAP uses TreeExplainer with feature_perturbation='interventional' to handle shifting backgrounds.
- Monitoring report prints a parseable summary ready for logging/ingestion.
- Eight documented implementation fixes (see script header and final output).

# Dependencies (see requirements.txt):
```Bash
pip install scikit-learn numpy pandas matplotlib scipy shap
```
# The Five Decisions (from the article)

1. Monitor input distributions (PSI) — your earliest signal.
2. Track score distributions (KS + Wasserstein) independently of labels.
3. Handle ground truth lag explicitly with a two-lane pipeline (confirmed labels + proxies like ECE).
4. Re-threshold before retraining — often recovers most performance in seconds.
5. Choose models with drift robustness in mind (Random Forest often outperforms Gradient Boosting over long horizons in drifting environments).

# Monitoring System Summary
The system uses four sequential gates:

- Gate 1: PSI on features
- Gate 2: KS on score distribution
- Gate 3: ECE (proxy when labels are delayed)
- Gate 4: Rolling recall drop at deployed threshold

# Re-threshold first. Retrain only when all gates fire (RED alert).

# Reproducibility & Citation
All results match the article when run with the default settings. Minor floating-point differences may occur across platforms due to NumPy BLAS implementations, but material conclusions remain unchanged.
Cite the article / repo if you use this in research, blogs, or internal systems.

# What's Next?
Extensions could include:

- Integration with production pipelines (e.g., Evidently, Prometheus, MLflow).
- Analyst override rate tracking as an additional concept-drift signal.
- Automated re-thresholding + targeted data collection based on SHAP attribution.

Feedback, issues, or production scenarios where this framework behaved differently are welcome — open an issue or reach out via the blog.

Author: Emmimal Alexander (EmiTechLogic)
License: MIT
Related: EmiTechLogic Blog | ML Diagnostics Series
Made with ❤️ for production-grade machine learning.
