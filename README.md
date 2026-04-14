# Heart (UCI) ML model — quickstart

This repo trains and evaluates a baseline ML model on the public **UCI Heart Disease** dataset and exports a `joblib` artifact for reuse.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

### Option A (recommended): fetch from UCI via `ucimlrepo`

```powershell
python .\src\train.py --out-dir .\artifacts
```

### Option B: train from a local CSV

If you already downloaded a CSV (e.g. from Kaggle/UCI mirrors), run:

```powershell
python .\src\train.py --csv-path .\data\heart.csv --out-dir .\artifacts
```

## Predict (example)

```powershell
python .\src\predict.py --model-path .\artifacts\model.joblib --input-json .\examples\one_patient.json
```

## Visualize

This writes plots to `artifacts/plots/`.

```powershell
python .\src\visualize.py --out-dir .\artifacts\plots
```

## What you get

- `artifacts/model.joblib`: fitted preprocessing + model pipeline
- `artifacts/metrics.json`: evaluation metrics
- `artifacts/feature_info.json`: basic column/label info

## Notes

- The UCI Heart dataset has multiple variants across sources. The training code standardizes common column names when possible and will print what it detected.
- This is a strong baseline; you can iterate on feature engineering, calibration, and thresholding once the pipeline runs end-to-end.

