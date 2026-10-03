"""
models.py — Phase 3: train Random Forest, SVM and Decision Tree under identical conditions
==========================================================================================
MSc AI dissertation — Ghanem Almarar

All three classifiers are trained on the SAME preprocessed arrays produced by
`src/preprocessing.py` (SMOTE-balanced 80 % of KDDTrain+, top-20 RF-selected features,
Min-Max scaled). Hyper-parameters are FIXED and recorded below — we are comparing
algorithms, not tuning budgets.

For every model we record:
  * 10-fold StratifiedKFold CV (shuffle=True, random_state=42) on the training set:
    per-fold accuracy, precision, recall, F1 (weighted) + fit/score time per fold
  * one final fit on the full training set, with wall-clock fit time
  * predict time on the 20 % holdout and on the official KDDTest+  (RQ2: efficiency)
  * saved model  -> models/<name>.pkl
  * saved predictions -> results/tables/predictions.npz (used by Phase 4 evaluation)

Run:  python -m src.models
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "nslkdd_processed.npz"
MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "results" / "tables"

RANDOM_STATE = 42
N_FOLDS = 10

# Fixed hyper-parameters, exactly as stated in PROJECT_PLAN.md / proposal.
# (scikit-learn defaults otherwise — recorded in the run log for the write-up.)
MODEL_SPECS = {
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "svm_rbf":       lambda: SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
    "decision_tree": lambda: DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE),
}
SCORING = {
    "accuracy": "accuracy",
    "precision_weighted": "precision_weighted",
    "recall_weighted": "recall_weighted",
    "f1_weighted": "f1_weighted",
}


def load_data() -> dict[str, np.ndarray]:
    d = np.load(PROCESSED)
    data = {k: d[k] for k in d.files}
    print(f"[data] train {data['X_train'].shape} | holdout {data['X_holdout'].shape} | official {data['X_test_official'].shape}")
    return data


def cross_validate_model(name: str, make, X, y) -> pd.DataFrame:
    """10-fold stratified CV; returns one row per fold."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    # n_jobs=-1 runs folds in parallel; RF itself is limited to 1 job inside CV to avoid oversubscription
    est = make()
    if name == "random_forest":
        est.set_params(n_jobs=1)
    res = cross_validate(est, X, y, cv=cv, scoring=SCORING, n_jobs=-1, return_train_score=False)
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(res)
    df.insert(0, "fold", range(1, N_FOLDS + 1))
    df.insert(0, "model", name)
    print(f"[cv] {name}: F1w {df['test_f1_weighted'].mean():.4f} ± {df['test_f1_weighted'].std():.4f} "
          f"| acc {df['test_accuracy'].mean():.4f} | wall {elapsed/60:.1f} min")
    return df


def final_fit_and_predict(name: str, make, data: dict) -> tuple[object, dict, dict]:
    """Fit once on full train; time fit + predictions on both test sets."""
    model = make()
    t0 = time.perf_counter(); model.fit(data["X_train"], data["y_train"]); fit_s = time.perf_counter() - t0

    t0 = time.perf_counter(); pred_ho = model.predict(data["X_holdout"]); ho_s = time.perf_counter() - t0
    t0 = time.perf_counter(); pred_of = model.predict(data["X_test_official"]); of_s = time.perf_counter() - t0

    timing = {
        "model": name,
        "fit_seconds": round(fit_s, 3),
        "predict_holdout_seconds": round(ho_s, 3),
        "predict_official_seconds": round(of_s, 3),
        "n_train": int(len(data["y_train"])),
        "n_holdout": int(len(data["y_holdout"])),
        "n_official": int(len(data["y_test_official"])),
        "predict_holdout_us_per_row": round(ho_s / len(data["y_holdout"]) * 1e6, 2),
    }
    print(f"[fit] {name}: fit {fit_s:.1f}s | predict holdout {ho_s:.2f}s | predict official {of_s:.2f}s")
    return model, timing, {"holdout": pred_ho, "official": pred_of}


def run() -> None:
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    data = load_data()
    X, y = data["X_train"], data["y_train"]

    cv_frames, timings, preds, params = [], [], {}, {}
    for name, make in MODEL_SPECS.items():
        print(f"\n===== {name} =====")
        cv_frames.append(cross_validate_model(name, make, X, y))
        model, timing, p = final_fit_and_predict(name, make, data)
        timings.append(timing)
        preds[f"{name}__holdout"] = p["holdout"]
        preds[f"{name}__official"] = p["official"]
        params[name] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                        for k, v in model.get_params().items()}
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    cv_all = pd.concat(cv_frames, ignore_index=True)
    cv_all.to_csv(TABLES_DIR / "cv_results_per_fold.csv", index=False)

    summary = (
        cv_all.groupby("model")[[f"test_{k}" for k in SCORING] + ["fit_time", "score_time"]]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.round(4).to_csv(TABLES_DIR / "cv_results_summary.csv")
    pd.DataFrame(timings).to_csv(TABLES_DIR / "timing.csv", index=False)
    np.savez_compressed(TABLES_DIR / "predictions.npz", **preds)

    with open(TABLES_DIR / "training_run_log.json", "w") as fh:
        json.dump(
            {
                "n_folds": N_FOLDS, "random_state": RANDOM_STATE,
                "sklearn_version": sklearn.__version__, "python": platform.python_version(),
                "machine": platform.machine(), "processor": platform.processor(),
                "hyperparameters": params,
            },
            fh, indent=2,
        )
    print("\n[cv summary]")
    print(summary[[c for c in summary.columns if c.startswith("test_f1") or c.startswith("test_acc")]].round(4).to_string())
    print(f"\n[done] models -> {MODELS_DIR} | tables -> {TABLES_DIR}")


if __name__ == "__main__":
    run()
