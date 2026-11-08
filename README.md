# Crop Recommendation System

A machine learning web app that recommends a suitable crop based on soil nutrients (N, P, K), local temperature, humidity, soil pH, and rainfall.

## How it works

1. `generate_dataset.py` builds a labeled dataset of 2,200 samples across 22 crops, using agronomically realistic ranges for each crop's ideal N/P/K, temperature, humidity, pH, and rainfall.
2. `train_model.py` trains a `RandomForestClassifier` (200 trees, max depth 12) on the data, evaluates it on a held-out 20% test split, and pickles the trained model, scaler, and label encoder.
3. `app.py` is a Flask app: a form collects the 7 input values, feeds them through the saved pipeline, and returns the top predicted crop plus two runner-up candidates with confidence scores.

**Test accuracy: 98.9%** (440 held-out samples, 22 classes). Rainfall, humidity, and potassium are the most influential features per the model's feature importances.

## Project structure

```
crop-recommendation/
├── generate_dataset.py     # builds data/Crop_recommendation.csv
├── train_model.py          # trains + pickles the model
├── app.py                  # Flask app
├── data/
│   └── Crop_recommendation.csv
├── model/
│   ├── crop_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── style.css
└── requirements.txt
```

## Setup (Windows / PowerShell)

```powershell
# from inside the crop-recommendation folder
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# (data/ and model/ are already included — skip these two if you just want to run the app)
python generate_dataset.py
python train_model.py

python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

If PowerShell blocks the venv activation script, run this once first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Setup (macOS / Linux)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Retraining on a different dataset

Swap `data/Crop_recommendation.csv` for your own (same 7 feature columns + `label`), then rerun `python train_model.py`. It will overwrite the pickled model/scaler/encoder in `model/`.

## Notes on the dataset

The dataset here is synthetically generated from published agronomic parameter ranges per crop (not scraped from a third party), so the project runs fully offline and reproducibly. If you have access to a real soil/crop dataset (e.g. from a government agriculture database or a field survey), swapping it in is a drop-in replacement — same column names, same retrain command.

## Possible extensions

- Add a fertilizer recommendation layer once a crop is selected
- Swap Random Forest for XGBoost and compare accuracy/feature importance
- Add a map-based location picker that pre-fills rainfall/humidity from a weather API
- Deploy to Render/Railway and add the live link to your resume
