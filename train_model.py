from typing import List

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    MODEL_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
)


# =============================================================================
# Training
# =============================================================================

def train_model() -> LGBMClassifier:
    """
    Train a LightGBM model using a reduced UNSW-NB15 feature set.
    """
    print("Loading training dataset...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    print(f"Training data loaded: {df_train.shape[0]} rows × {df_train.shape[1]} columns")

    features = [f for f in FEATURE_COLUMNS if f in df_train.columns]
    X_train = df_train[features].copy()
    y_train = df_train["label"]

    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        class_weight={0: 1.0, 1: 0.5},
        n_jobs=-1,
    )

    print("Training LightGBM model...")
    model.fit(
        X_train,
        y_train,
        eval_metric="binary_logloss",
        categorical_feature=list(CATEGORICAL_FEATURES),
    )

    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to: {MODEL_PATH}")

    return model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model: LGBMClassifier) -> None:
    """
    Evaluate a trained model on the UNSW-NB15 test set.
    """
    print("Loading testing dataset...")
    df_test = pd.read_csv(TEST_DATA_PATH)
    print(f"Testing data loaded: {df_test.shape[0]} rows × {df_test.shape[1]} columns")

    features = [f for f in FEATURE_COLUMNS if f in df_test.columns]
    X_test = df_test[features].copy()
    y_test = df_test["label"]

    for col in CATEGORICAL_FEATURES:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    print("Evaluating model on test data...")
    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Attack"],
            zero_division=0,
        )
    )

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\nLabel distribution (test data):")
    print(y_test.value_counts())


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=== UNSW-NB15 LightGBM Training Pipeline ===\n")

    model = train_model()
    evaluate_model(model)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
