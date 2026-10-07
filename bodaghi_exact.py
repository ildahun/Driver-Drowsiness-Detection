#!/usr/bin/env python3
"""
Αναπαραγωγή του κώδικα των Bodaghi et al. (4.Multimodal_Analysis.ipynb), όπως ακριβώς τον έγραψαν,
πάνω στα χαρακτηριστικά της διπλωματικής (features.csv), σε τρία πρωτόκολλα:

  A) "Όπως στο άρθρο":   MinMax + PCA(0.90) σε ΟΛΑ τα δεδομένα, μετά StratifiedKFold(5, shuffle, 42)
  B) "5-fold χωρίς διαρροή": MinMax + PCA μαθαίνονται ΜΟΝΟ στο train κάθε fold (Pipeline)
  C) "LOSO":               νέος οδηγός (leave-one-subject-out), χωρίς διαρροή

Συνδυασμοί όπως στο notebook:  Bio+Behavioral, Bio+Facial, Bio+Behavioral+Facial
(Behavioral = λαβή + τηλεμετρία = κλάδος veh της διπλωματικής· το pose δεν υπάρχει στα χαρακτηριστικά μας)

Υπερπαράμετροι ΑΚΡΙΒΩΣ όπως στον κώδικά τους:
  SVC(kernel='rbf', C=0.1, gamma='auto', class_weight='balanced', random_state=42)
  RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)

Χρήση:  python bodaghi_exact.py --features features.csv --out results_bodaghi
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

COMBOS = {"Bio+Behavioral": ["bio", "veh"], "Bio+Facial": ["bio", "face"], "Bio+Behavioral+Facial": ["bio", "veh", "face"]}
PAPER = {  # Πίνακας 8 του άρθρου (Acc, Pre, Rec, F1 σε %)
    ("SVM", "Bio+Facial"): (73.06, 73.05, 72.84, 72.43), ("SVM", "Bio+Behavioral"): (79.87, 79.88, 80.38, 80.04),
    ("SVM", "Bio+Behavioral+Facial"): (83.75, 83.85, 84.13, 83.95),
    ("RF", "Bio+Facial"): (70.65, 75.94, 68.71, 69.77), ("RF", "Bio+Behavioral"): (74.14, 77.92, 72.53, 73.61),
    ("RF", "Bio+Behavioral+Facial"): (75.14, 76.41, 74.69, 75.21),
}


def models():
    return {"SVM": SVC(kernel="rbf", C=0.1, gamma="auto", cache_size=200, class_weight="balanced", random_state=42),
            "RF": RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)}


def scores(y, p):
    return {"Acc": 100 * accuracy_score(y, p),
            "Pre": 100 * precision_score(y, p, average="macro", zero_division=0),
            "Rec": 100 * recall_score(y, p, average="macro", zero_division=0),
            "F1": 100 * f1_score(y, p, average="macro", zero_division=0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--out", default="results_bodaghi")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.features, dtype={"subject": str, "session": str}, low_memory=False)
    y = df["y"].to_numpy(int)
    groups = df["subject"].to_numpy()
    print(f"{len(df)} παράθυρα, {df.subject.nunique()} οδηγοί")

    rows = []
    for combo, branches in COMBOS.items():
        cols = [c for c in df.columns if any(c.startswith(f"{b}__") for b in branches)]
        X = df[cols].to_numpy(float)
        X[~np.isfinite(X)] = np.nan
        X[np.abs(np.nan_to_num(X)) > 1e9] = np.nan
        # οι Bodaghi δεν έχουν ελλιπείς τιμές στα δεδομένα τους· εδώ συμπληρώνονται με τη διάμεσο
        for name, clf in models().items():
            # A) ακριβώς όπως στο notebook: scaler + PCA σε όλα τα δεδομένα πριν το CV
            Xa = SimpleImputer(strategy="median").fit_transform(X)
            Xa = PCA(n_components=0.90).fit_transform(MinMaxScaler().fit_transform(Xa))
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            pa = cross_val_predict(clf, Xa, y, cv=cv, n_jobs=-1)
            # B) 5-fold χωρίς διαρροή
            pipe = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler(), PCA(n_components=0.90), clf)
            pb = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
            # C) LOSO
            pc = cross_val_predict(pipe, X, y, cv=LeaveOneGroupOut(), groups=groups, n_jobs=-1)
            for proto, p in [("A_5fold_όπως_στο_άρθρο", pa), ("B_5fold_χωρίς_διαρροή", pb), ("C_LOSO", pc)]:
                rows.append({"model": name, "combo": combo, "protocol": proto, **scores(y, p)})
            pap = PAPER[(name, combo)]
            rows.append({"model": name, "combo": combo, "protocol": "Άρθρο_Πίνακας8",
                         "Acc": pap[0], "Pre": pap[1], "Rec": pap[2], "F1": pap[3]})
            print(f"  {name:3s} · {combo:22s} ok")

    res = pd.DataFrame(rows)
    res.to_csv(f"{a.out}/bodaghi_exact.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    for proto in ["Άρθρο_Πίνακας8", "A_5fold_όπως_στο_άρθρο", "B_5fold_χωρίς_διαρροή", "C_LOSO"]:
        print(f"\n=== {proto} ===")
        print(res[res.protocol == proto].drop(columns="protocol").round(2).to_string(index=False))
    print(f"\nΑρχείο: {a.out}/bodaghi_exact.csv")
    print("Σύγκρινε με MAIN_TABLE_kfold.csv / MAIN_TABLE_loso.csv (late_meta_reliability = η προτεινόμενη μέθοδος).")


if __name__ == "__main__":
    main()
