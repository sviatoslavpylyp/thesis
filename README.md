# Suricata Alert Classification with LightGBM

This project implements a machine-learning–based alert filtering pipeline for
**Suricata IDS** logs (`eve.json`) using a **LightGBM classifier** trained on the
UNSW-NB15 dataset.

The system supports two inference modes:
- **Single read**: process all existing alerts once
- **Continuous mode**: monitor `eve.json` and classify only newly appended alerts
  at fixed intervals

If a trained model is not found locally, it is automatically created.

---

## High-Level Overview

### Pipeline

1. **Training (optional, automatic)**
   - Train a LightGBM model on UNSW-NB15 data
   - Persist the model locally

2. **Inference**
   - Parse Suricata `eve.json` (JSON Lines format)
   - Extract a reduced feature set
   - Run predictions
   - Label alerts as *Benign* or *Attack*

3. **Monitoring (optional)**
   - Continuously tail `eve.json`
   - Process only newly added alerts every 15 seconds

---

## Project Structure

```text
project/
│
├── config.py                # Centralized configuration (paths, features, timing)
├── train_model.py           # Model training & evaluation
├── infer_suricata.py        # Single-read & continuous inference
├── lightgbm_unsw_model_subset.pkl  # Trained model (auto-created if missing)
└── README.md
