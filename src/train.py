import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass
class TrainOutputs:
    model_path: str
    metrics_path: str
    feature_info_path: str


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize some common UCI Heart dataset column name variants.
    """
    rename_map = {
        "cp": "chest_pain_type",
        "trestbps": "resting_bp",
        "chol": "cholesterol",
        "fbs": "fasting_blood_sugar",
        "restecg": "resting_ecg",
        "thalach": "max_heart_rate",
        "exang": "exercise_induced_angina",
        "oldpeak": "st_depression",
        "slope": "st_slope",
        "ca": "num_major_vessels",
        "thal": "thalassemia",
        "sex": "sex",
        "age": "age",
    }
    cols = {c: rename_map.get(c, c) for c in df.columns}
    return df.rename(columns=cols)


def _infer_target_column(df: pd.DataFrame) -> str:
    # Common target names across variants
    for cand in ["target", "num", "diagnosis", "heart_disease", "output"]:
        if cand in df.columns:
            return cand
    raise ValueError(
        "Could not infer target column. Expected one of: "
        "target/num/diagnosis/heart_disease/output. "
        f"Columns: {list(df.columns)}"
    )


def _load_from_ucimlrepo() -> Tuple[pd.DataFrame, pd.Series]:
    # UCI ML Repo dataset: Heart Disease (id=45)
    from ucimlrepo import fetch_ucirepo  # type: ignore

    ds = fetch_ucirepo(id=45)
    # ds.data.features and ds.data.targets are pandas objects
    X = ds.data.features.copy()
    y_df = ds.data.targets.copy()
    if isinstance(y_df, pd.DataFrame) and y_df.shape[1] == 1:
        y = y_df.iloc[:, 0]
    elif isinstance(y_df, pd.DataFrame):
        # pick the first target if multiple
        y = y_df.iloc[:, 0]
    else:
        y = pd.Series(y_df)

    X = _standardize_columns(X)
    y = y.rename("target")
    return X, y


def _load_from_csv(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    df = _standardize_columns(df)
    target_col = _infer_target_column(df)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    y = y.rename("target")
    return X, y


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def _coerce_binary_target(y: pd.Series) -> pd.Series:
    """
    UCI Heart can be binary (0/1) or ordinal (0..4). Convert to binary by:
    - if values are {0,1}: keep
    - else: map (0 -> 0, >0 -> 1)
    """
    y_clean = y.copy()
    # Try numeric coercion first
    y_num = pd.to_numeric(y_clean, errors="coerce")
    if y_num.notna().all():
        unique = set(y_num.unique().tolist())
        if unique.issubset({0, 1}):
            return y_num.astype(int)
        return (y_num > 0).astype(int)

    # Non-numeric labels: treat common strings
    y_str = y_clean.astype(str).str.strip().str.lower()
    if set(y_str.unique().tolist()).issubset({"0", "1"}):
        return y_str.astype(int)
    if "no" in set(y_str.unique().tolist()) or "yes" in set(y_str.unique().tolist()):
        return y_str.map({"no": 0, "yes": 1}).astype(int)
    raise ValueError(f"Unrecognized target labels: {sorted(set(y_str.unique().tolist()))[:20]}")


def _evaluate_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")
    preds = (proba[:, 1] >= 0.5).astype(int)

    metrics: Dict[str, Any] = {
        "cv": {"n_splits": 5, "type": "StratifiedKFold", "random_state": 42},
        "accuracy": float(accuracy_score(y, preds)),
        "roc_auc": float(roc_auc_score(y, proba[:, 1])),
        "classification_report": classification_report(y, preds, output_dict=True),
        "threshold": 0.5,
        "n_rows": int(X.shape[0]),
        "n_features_raw": int(X.shape[1]),
        "positive_rate": float(np.mean(y.values)),
    }
    return metrics


def train(
    csv_path: Optional[str],
    out_dir: str,
    model_type: str,
) -> TrainOutputs:
    _ensure_dir(out_dir)

    if csv_path:
        X, y = _load_from_csv(csv_path)
        source = {"type": "csv", "path": os.path.abspath(csv_path)}
    else:
        X, y = _load_from_ucimlrepo()
        source = {"type": "ucimlrepo", "id": 45}

    y_bin = _coerce_binary_target(y)

    pre = _make_preprocessor(X)

    if model_type == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif model_type == "hgb":
        clf = HistGradientBoostingClassifier(random_state=42)
    else:
        raise ValueError("--model-type must be one of: logreg, hgb")

    pipe = Pipeline(steps=[("pre", pre), ("model", clf)])

    metrics = _evaluate_cv(pipe, X, y_bin)
    metrics["data_source"] = source
    metrics["model_type"] = model_type

    # Fit final model on full data
    pipe.fit(X, y_bin)

    model_path = os.path.join(out_dir, "model.joblib")
    metrics_path = os.path.join(out_dir, "metrics.json")
    feature_info_path = os.path.join(out_dir, "feature_info.json")

    joblib.dump(pipe, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    feature_info = {
        "raw_columns": list(X.columns),
        "target_name": "target",
        "target_mapping": "binary: 0=no disease, 1=disease",
    }
    with open(feature_info_path, "w", encoding="utf-8") as f:
        json.dump(feature_info, f, indent=2)

    return TrainOutputs(model_path=model_path, metrics_path=metrics_path, feature_info_path=feature_info_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", type=str, default=None, help="Optional local CSV path (if not using ucimlrepo fetch).")
    ap.add_argument("--out-dir", type=str, default="artifacts", help="Output directory for model + metrics.")
    ap.add_argument(
        "--model-type",
        type=str,
        default="logreg",
        choices=["logreg", "hgb"],
        help="Model family to train.",
    )
    args = ap.parse_args()

    outputs = train(csv_path=args.csv_path, out_dir=args.out_dir, model_type=args.model_type)
    print(json.dumps(asdict(outputs), indent=2))


if __name__ == "__main__":
    main()

