import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

with open(os.path.join(MODEL_DIR, "crop_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

FIELD_META = {
    "N": {"label": "Nitrogen (N)", "unit": "kg/ha", "placeholder": "e.g. 90"},
    "P": {"label": "Phosphorus (P)", "unit": "kg/ha", "placeholder": "e.g. 42"},
    "K": {"label": "Potassium (K)", "unit": "kg/ha", "placeholder": "e.g. 43"},
    "temperature": {"label": "Temperature", "unit": "°C", "placeholder": "e.g. 25.5"},
    "humidity": {"label": "Humidity", "unit": "%", "placeholder": "e.g. 80"},
    "ph": {"label": "Soil pH", "unit": "", "placeholder": "e.g. 6.5"},
    "rainfall": {"label": "Rainfall", "unit": "mm", "placeholder": "e.g. 200"},
}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", fields=FEATURES, meta=FIELD_META, values={}, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    values = {}
    try:
        for feat in FEATURES:
            raw = request.form.get(feat, "").strip()
            if raw == "":
                raise ValueError(f"{FIELD_META[feat]['label']} is required.")
            values[feat] = float(raw)
    except ValueError as e:
        return render_template(
            "index.html", fields=FEATURES, meta=FIELD_META, values=request.form, error=str(e)
        )

    x = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    x_scaled = scaler.transform(x)

    proba = model.predict_proba(x_scaled)[0]
    top_idx = np.argsort(proba)[::-1][:3]
    top_crops = [(encoder.classes_[i], round(proba[i] * 100, 1)) for i in top_idx]

    return render_template(
        "result.html",
        crop=top_crops[0][0],
        confidence=top_crops[0][1],
        alternatives=top_crops[1:],
        values=values,
        meta=FIELD_META,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
