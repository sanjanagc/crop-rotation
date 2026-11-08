"""
Trains a RandomForestClassifier on the crop recommendation dataset,
evaluates it, and pickles the model + scaler + label encoder for use
by the Flask app.
"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y_raw = df[TARGET]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling isn't strictly required for Random Forest (it's tree-based
    # and scale-invariant), but we keep a scaler in the pipeline so the
    # same preprocessing artifact could support swapping in a
    # scale-sensitive model (e.g. SVM/logistic regression) later without
    # touching the Flask app.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Feature importance — useful to mention in an interview / README
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("Feature importances:")
    print(importances.to_string())

    with open(os.path.join(MODEL_DIR, "crop_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    print(f"\nSaved model, scaler, and label encoder to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
