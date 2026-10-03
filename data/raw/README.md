# data/raw — NSL-KDD provenance

Files here are **git-ignored** (see `.gitignore`). Re-download with the commands below if the folder is empty.

| File | Rows | Fields | MD5 |
|---|---|---|---|
| `KDDTrain+.txt` | 125,973 | 43 (41 features + label + difficulty) | `f5592a95d1d1428348dfa6ca9652a800` |
| `KDDTest+.txt`  | 22,544  | 43 | `7c6d1b1af246690766394920d6b4c751` |

**Origin:** NSL-KDD (Tavallaee et al., 2009), originally hosted by the Canadian Institute for Cybersecurity,
University of New Brunswick. The UNB archive is no longer maintained, so the canonical files were obtained on
**22 August 2026** from the public GitHub mirror `defcom17/NSL_KDD` (identical canonical release also mirrored
on Kaggle — Hassan, 2019 — which is the source cited in the research proposal).

```bash
curl -sSL -o "KDDTrain+.txt" "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
curl -sSL -o "KDDTest+.txt"  "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
md5sum KDDTrain+.txt KDDTest+.txt   # must match the table above
```

Row counts match the figures stated in the research proposal (125,973 / 22,544).
