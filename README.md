# Machine Learning-Based Network Intrusion Detection
## A Comparative Analysis of Supervised Algorithms Using the NSL-KDD Dataset

**Author:** Ghanem Saeed Salem Saeed Almarar
**Programme:** MSc Artificial Intelligence, University of Essex Online (CSPROJ_PCOM7E)

Comparative study of Random Forest, SVM (RBF) and Decision Tree classifiers on the
NSL-KDD benchmark under a single fixed preprocessing pipeline, so that any difference
in measured performance is attributable to the algorithm rather than to feature
engineering. Models are evaluated on five metrics (accuracy, weighted precision /
recall / F1, and false-positive rate for the Normal class) plus training and
prediction time, on two test sets: a stratified 20% holdout of KDDTrain+ and the
official KDDTest+ (which contains attack types unseen in training).

## Repository structure

| Path | Contents |
|---|---|
| `src/preprocessing.py` | 11-step fixed pipeline: load, label mapping, one-hot, align, split, scale, SMOTE, feature selection, persist |
| `src/models.py` | RF / SVM / DT training, 10-fold stratified CV, timing capture |
| `src/evaluation.py` | Metrics on both test sets, per-class results, confusion matrices, figures |
| `notebooks/` | Narrated, executed walkthroughs of each stage (02–04) |
| `results/tables/` | All result tables (CSV) and the training run log |
| `results/figures/` | All figures used in the dissertation |
| `data/raw/README.md` | Dataset provenance, download commands and MD5 checksums |

## Reproducing the experiment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
# download NSL-KDD into data/raw/ (see data/raw/README.md), then:
python -m src.preprocessing
python -m src.models
python -m src.evaluation
```

A single random seed (42) is used for every stochastic operation; the full run takes
approximately fifteen minutes on commodity hardware and regenerates every table and
figure deterministically.

## Dataset

NSL-KDD (Tavallaee et al., 2009), obtained from its public mirror; 125,973 training
and 22,544 test records. See `data/raw/README.md` for provenance and checksums.
