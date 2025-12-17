from pathlib import Path

# =============================================================================
# Base Paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lightgbm_unsw_model_subset.pkl"

# =============================================================================
# Data Sources (⚠️ VERIFY THESE PATHS ON YOUR SYSTEM)
# =============================================================================

EVE_PATH = Path(r"C:\Program Files\Suricata\eve.json")

TRAIN_DATA_PATH = Path(r"D:\thesis\model_datasets\UNSW_NB15_training-set.csv")
TEST_DATA_PATH = Path(r"D:\thesis\model_datasets\UNSW_NB15_testing-set.csv")

# =============================================================================
# Inference Configuration
# =============================================================================

POLL_INTERVAL_SECONDS = 15
DEFAULT_THRESHOLD = 0.90

# =============================================================================
# Features
# =============================================================================

FEATURE_COLUMNS = [
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
]

CATEGORICAL_FEATURES = {"proto", "service", "state"}
