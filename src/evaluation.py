"""
evaluation.py — Phase 4: compare RF / SVM / DT on the five study metrics
=======================================================================
MSc AI dissertation — Ghanem Almarar

Inputs  (produced by src/preprocessing.py and src/models.py):
  data/processed/nslkdd_processed.npz      y_holdout, y_test_official
  results/tables/predictions.npz           <model>__holdout, <model>__official
  results/tables/timing.csv                fit / predict times
  results/tables/cv_results_per_fold.csv   10-fold CV per fold
  models/label_encoder.pkl                 class names <-> ints

Outputs (results/):
  tables/performance_comparison.csv        3 models x 2 test sets x 5 metrics  (master table, RQ1 + RQ2)
  tables/per_class_metrics.csv             precision / recall / F1 per class per model per test set
  tables/confusion_<model>_<set>.csv       raw confusion matrices
  figures/cm_<model>_<set>.png             confusion-matrix heatmaps
  figures/fig_metric_comparison.png        grouped bars: 5 metrics x 3 models (holdout vs official)
  figures/fig_cv_f1_boxplot.png            10-fold F1 distribution per model
  figures/fig_timing.png                   fit + predict time per model (log scale)
  figures/fig_per_class_recall.png         recall per attack class (fairness view)

Metric definitions (documented for Ch3):
  accuracy, precision/recall/F1 = weighted average over the 5 classes (sklearn `average='weighted'`)
  FPR (Normal) = (# truly-Normal records predicted as ANY attack) / (# truly-Normal records)
      -> "normal traffic falsely flagged as malicious" — the alert-fatigue metric in the proposal.

Run:  python -m src.evaluation
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "nslkdd_processed.npz"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
MODELS = ROOT / "models"

MODEL_LABELS = {"random_forest": "Random Forest", "svm_rbf": "SVM (RBF)", "decision_tree": "Decision Tree"}
SET_LABELS = {"holdout": "20 % holdout (KDDTrain+)", "official": "KDDTest+ (official)"}
sns.set_theme(style="whitegrid", context="paper")


def fpr_normal(y_true, y_pred, normal_idx: int) -> float:
    """False-positive rate for the Normal class: Normal records predicted as any attack."""
    is_normal = y_true == normal_idx
    if is_normal.sum() == 0:
        return float("nan")
    return float((y_pred[is_normal] != normal_idx).mean())


def five_metrics(y_true, y_pred, normal_idx) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "fpr_normal": fpr_normal(y_true, y_pred, normal_idx),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),   # extra: exposes minority classes
    }


def plot_confusion(cm: np.ndarray, classes, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues", cbar=False, xticklabels=classes, yticklabels=classes, ax=ax,
                annot_kws={"size": 8})
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=10)
    plt.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def run() -> pd.DataFrame:
    FIGURES.mkdir(parents=True, exist_ok=True)
    d = np.load(PROCESSED)
    preds = np.load(TABLES / "predictions.npz")
    le = joblib.load(MODELS / "label_encoder.pkl")
    classes = list(le.classes_)
    normal_idx = classes.index("Normal")
    truth = {"holdout": d["y_holdout"], "official": d["y_test_official"]}

    rows, per_class_rows = [], []
    for model in MODEL_LABELS:
        for tset, y_true in truth.items():
            y_pred = preds[f"{model}__{tset}"]
            m = five_metrics(y_true, y_pred, normal_idx)
            rows.append({"model": MODEL_LABELS[model], "test_set": SET_LABELS[tset], **m})

            cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(TABLES / f"confusion_{model}_{tset}.csv")
            plot_confusion(cm, classes, f"{MODEL_LABELS[model]} — {SET_LABELS[tset]}", FIGURES / f"cm_{model}_{tset}.png")

            rep = classification_report(y_true, y_pred, labels=range(len(classes)), target_names=classes,
                                        output_dict=True, zero_division=0)
            for cls in classes:
                per_class_rows.append({"model": MODEL_LABELS[model], "test_set": SET_LABELS[tset], "class": cls,
                                       "precision": rep[cls]["precision"], "recall": rep[cls]["recall"],
                                       "f1": rep[cls]["f1-score"], "support": int(rep[cls]["support"])})

    perf = pd.DataFrame(rows)
    timing = pd.read_csv(TABLES / "timing.csv")
    timing["model"] = timing["model"].map(MODEL_LABELS)
    perf = perf.merge(timing[["model", "fit_seconds", "predict_holdout_seconds", "predict_official_seconds"]], on="model")
    perf.round(4).to_csv(TABLES / "performance_comparison.csv", index=False)
    per_class = pd.DataFrame(per_class_rows)
    per_class.round(4).to_csv(TABLES / "per_class_metrics.csv", index=False)

    # --- figures -------------------------------------------------------------
    metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "fpr_normal"]
    long = perf.melt(id_vars=["model", "test_set"], value_vars=metrics, var_name="metric", value_name="value")
    g = sns.catplot(data=long, x="metric", y="value", hue="model", col="test_set", kind="bar", height=3.6, aspect=1.25,
                    palette="deep")
    g.set_xticklabels(rotation=20); g.set_axis_labels("", "score"); g.set_titles("{col_name}")
    for ax in g.axes.flat:
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=6, rotation=90, padding=2)
        ax.set_ylim(0, 1.12)
    g.figure.suptitle("Five study metrics — RF vs SVM vs DT (FPR: lower is better)", y=1.03, fontsize=10)
    g.savefig(FIGURES / "fig_metric_comparison.png", dpi=200, bbox_inches="tight"); plt.close(g.figure)

    cv = pd.read_csv(TABLES / "cv_results_per_fold.csv")
    cv["model"] = cv["model"].map(MODEL_LABELS)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    sns.boxplot(data=cv, x="model", y="test_f1_weighted", ax=ax, palette="deep", hue="model", legend=False)
    sns.stripplot(data=cv, x="model", y="test_f1_weighted", ax=ax, color="k", size=3)
    ax.set_ylabel("weighted F1 (per fold)"); ax.set_xlabel(""); ax.set_title("10-fold stratified CV on SMOTE'd training set")
    plt.tight_layout(); fig.savefig(FIGURES / "fig_cv_f1_boxplot.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    t_long = timing.melt(id_vars="model", value_vars=["fit_seconds", "predict_holdout_seconds", "predict_official_seconds"],
                         var_name="phase", value_name="seconds")
    sns.barplot(data=t_long, x="model", y="seconds", hue="phase", ax=ax, palette="muted")
    ax.set_yscale("log"); ax.set_ylabel("seconds (log)"); ax.set_xlabel(""); ax.set_title("Computational cost (RQ2)")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
    plt.tight_layout(); fig.savefig(FIGURES / "fig_timing.png", dpi=200); plt.close(fig)

    g2 = sns.catplot(data=per_class, x="class", y="recall", hue="model", col="test_set", kind="bar", height=3.6,
                     aspect=1.25, palette="deep", order=["Normal", "DoS", "Probe", "R2L", "U2R"])
    g2.set_axis_labels("", "recall"); g2.set_titles("{col_name}")
    for ax in g2.axes.flat:
        ax.set_ylim(0, 1.1)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=6, rotation=90, padding=2)
    g2.figure.suptitle("Per-class recall — where do the models fail? (fairness / RQ3)", y=1.03, fontsize=10)
    g2.savefig(FIGURES / "fig_per_class_recall.png", dpi=200, bbox_inches="tight"); plt.close(g2.figure)

    print(perf.round(4).to_string(index=False))
    print("\n[per-class recall]")
    print(per_class.pivot_table(index=["test_set", "class"], columns="model", values="recall").round(3).to_string())
    print(f"\n[done] tables -> {TABLES} | figures -> {FIGURES}")
    return perf


if __name__ == "__main__":
    run()
