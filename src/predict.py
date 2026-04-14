import argparse
import json
from typing import Any, Dict

import joblib
import pandas as pd


def predict_one(model_path: str, input_json_path: str) -> Dict[str, Any]:
    pipe = joblib.load(model_path)
    with open(input_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object with feature_name -> value.")

    X = pd.DataFrame([payload])
    proba = pipe.predict_proba(X)[0, 1]
    pred = int(proba >= 0.5)

    return {"pred": pred, "proba": float(proba), "threshold": 0.5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="Path to artifacts/model.joblib")
    ap.add_argument("--input-json", required=True, help="Path to JSON file with one row of features.")
    args = ap.parse_args()

    out = predict_one(args.model_path, args.input_json)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

