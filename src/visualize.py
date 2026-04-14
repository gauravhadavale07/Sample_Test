import argparse
import os
from typing import Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    for cand in ["target", "num", "diagnosis", "heart_disease", "output"]:
        if cand in df.columns:
            return cand
    raise ValueError(
        "Could not infer target column. Expected one of: "
        "target/num/diagnosis/heart_disease/output. "
        f"Columns: {list(df.columns)}"
    )


def _load_from_ucimlrepo() -> Tuple[pd.DataFrame, pd.Series]:
    from ucimlrepo import fetch_ucirepo  # type: ignore

    ds = fetch_ucirepo(id=45)
    X = ds.data.features.copy()
    y_df = ds.data.targets.copy()
    if isinstance(y_df, pd.DataFrame) and y_df.shape[1] >= 1:
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
    y = df[target_col].rename("target")
    X = df.drop(columns=[target_col])
    return X, y


def _coerce_binary_target(y: pd.Series) -> pd.Series:
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.notna().all():
        unique = set(y_num.unique().tolist())
        if unique.issubset({0, 1}):
            return y_num.astype(int)
        return (y_num > 0).astype(int)
    y_str = y.astype(str).str.strip().str.lower()
    if set(y_str.unique().tolist()).issubset({"0", "1"}):
        return y_str.astype(int)
    if "no" in set(y_str.unique().tolist()) or "yes" in set(y_str.unique().tolist()):
        return y_str.map({"no": 0, "yes": 1}).astype(int)
    raise ValueError("Unrecognized target labels.")


def plot_target_balance(y: pd.Series, out_path: str) -> None:
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(x=y.astype(int))
    ax.set_title("Target balance")
    ax.set_xlabel("target (0=no disease, 1=disease)")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_numeric_distributions(X: pd.DataFrame, y: pd.Series, out_path: str) -> None:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if not num_cols:
        return
    df = X[num_cols].copy()
    df["target"] = y.astype(int).values
    melted = df.melt(id_vars=["target"], var_name="feature", value_name="value")
    g = sns.FacetGrid(melted, col="feature", col_wrap=4, sharex=False, sharey=False, height=2.2)
    g.map_dataframe(sns.histplot, x="value", hue="target", element="step", stat="density", common_norm=False, bins=20)
    g.add_legend(title="target")
    g.fig.suptitle("Numeric feature distributions by target", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_correlation_heatmap(X: pd.DataFrame, out_path: str) -> None:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if len(num_cols) < 2:
        return
    corr = X[num_cols].corr(numeric_only=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, cmap="vlag", center=0, square=True)
    plt.title("Numeric feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_cv_roc(model_path: str, X: pd.DataFrame, y: pd.Series, out_path: str) -> None:
    pipe = joblib.load(model_path)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y, proba, ax=ax, name="CV ROC")
    ax.set_title("Cross-validated ROC curve")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", type=str, default=None, help="Optional local CSV path (if not using ucimlrepo fetch).")
    ap.add_argument("--model-path", type=str, default="artifacts/model.joblib", help="Trained model pipeline joblib.")
    ap.add_argument("--out-dir", type=str, default="artifacts/plots", help="Directory to write plots.")
    args = ap.parse_args()

    if args.csv_path:
        X, y = _load_from_csv(args.csv_path)
    else:
        X, y = _load_from_ucimlrepo()

    y_bin = _coerce_binary_target(y)

    _ensure_dir(args.out_dir)
    plot_target_balance(y_bin, os.path.join(args.out_dir, "01_target_balance.png"))
    plot_numeric_distributions(X, y_bin, os.path.join(args.out_dir, "02_numeric_distributions.png"))
    plot_correlation_heatmap(X, os.path.join(args.out_dir, "03_corr_heatmap.png"))

    if os.path.exists(args.model_path):
        plot_cv_roc(args.model_path, X, y_bin, os.path.join(args.out_dir, "04_cv_roc.png"))

    print(f"Wrote plots to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

