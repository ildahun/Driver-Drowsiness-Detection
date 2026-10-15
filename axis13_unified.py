#!/usr/bin/env python3
"""
UL-DD · Ενοποίηση Αξόνων 1, 2 και 3: η προτεινόμενη όψιμη σύντηξη (meta-classifier με αξιοπιστία)
πάνω σε ΑΤΟΜΙΚΑ ΚΑΝΟΝΙΚΟΠΟΙΗΜΕΝΑ χαρακτηριστικά, σε νέο οδηγό (LOSO).

Συγκρίνει, στα ίδια ακριβώς παράθυρα ελέγχου (χωρίς τα k λεπτά βαθμονόμησης):
  raw  : χαρακτηριστικά όπως είναι (όπως στον Άξονα 1)
  norm : κάθε χαρακτηριστικό μείον τη μέση τιμή του οδηγού στα πρώτα k λεπτά (όπως στον Άξονα 3)
για τις μεθόδους:
  late_meta_reliability (η προτεινόμενη)
  Bodaghi et al.: early_fusion_RF, early_paper_SVM, early_paper_RF, early_MLP
  Das et al.:     late_soft_voting, late_hard_voting
  Krishna et al.: late_meta_dropout (meta-classifier + modality dropout, χωρίς αξιοπιστία)
  Kontras et al.: unimodal_bio, unimodal_veh, unimodal_face (κάθε τροπικότητα μόνη της)
  Κλασικό stacking: late_stacking_LR
και για τα σενάρια all / no_bio / no_veh / no_face (Άξονας 2: λείπει ένας αισθητήρας στον έλεγχο).

Χρειάζεται στον ίδιο φάκελο: ul_dd_baseline.py, axis3_personalization.py, features.csv

Χρήση:
    python axis13_unified.py --features features.csv --out results_unified
    (προαιρετικά: --minutes 4 --baseline same --trees 150)
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

import ul_dd_baseline as ul
import axis3_personalization as a3

warnings.filterwarnings("ignore")
BR = ul.BRANCHES
NC = ul.N_CLASSES
PROPOSED = "late_meta_reliability"
WORK = {  # μέθοδος -> σχετική εργασία
    PROPOSED: "Η παρούσα εργασία",
    "early_fusion_RF": "Bodaghi et al.", "early_paper_SVM": "Bodaghi et al.", "early_paper_RF": "Bodaghi et al.",
    "early_MLP": "Bodaghi et al.", "late_soft_voting": "Das et al.", "late_hard_voting": "Das et al.",
    "late_meta_dropout": "Krishna et al.", "late_stacking_LR": "Κλασικό stacking",
    "unimodal_bio": "Kontras et al.", "unimodal_veh": "Kontras et al.", "unimodal_face": "Kontras et al.",
}


def mask(P_list, flags, Q, b):
    """Ο κλάδος b «χάνεται» στον έλεγχο."""
    P2 = [p.copy() for p in P_list]
    f2, Q2 = flags.copy(), Q.copy()
    P2[b][:] = np.nan
    f2[:, b] = 0
    Q2[:, b] = 0.0
    return P2, f2, Q2


def run(df, k_min, baseline, trees):
    y = df.y.to_numpy(int)
    groups = df.subject.to_numpy()
    feat = {b: [c for c in df.columns if c.startswith(f"{b}__")] for b in BR}
    Xraw = {b: df[feat[b]].to_numpy(float) for b in BR}
    Xnorm = {b: a3.baseline_normalize(Xraw[b], df, k_min, baseline) for b in BR}
    avail = {b: ~np.isnan(Xraw[b]).all(axis=1) for b in BR}
    qcols = [f"q__{b}" for b in BR]
    Qall = (df[qcols].fillna(0.0).to_numpy(float) if all(c in df for c in qcols)
            else np.stack([avail[b] for b in BR], 1).astype(float))
    cal = a3.calib_mask(df, k_min)
    rng = np.random.default_rng(ul.SEED)
    rec = []

    subjects = sorted(np.unique(groups))
    for si, s in enumerate(subjects, 1):
        tr = np.where(groups != s)[0]
        te = np.where((groups == s) & ~cal)[0]          # έλεγχος ΜΟΝΟ μετά τη βαθμονόμηση
        if len(te) == 0:
            continue
        t0 = time.time()
        for space, X in (("raw", Xraw), ("norm", Xnorm)):
            # --- κλάδοι (Random Forest): OOF στο train, μοντέλο στο train
            Ptr, Pte = [], []
            for b in BR:
                Ptr.append(ul.branch_oof(X[b][tr], y[tr], groups[tr], avail[b][tr], trees, cv="loso"))
                p = np.full((len(te), NC), np.nan)
                ok, okt = avail[b][te], avail[b][tr]
                if okt.sum() >= 10 and ok.any():
                    m = ul.branch_model(trees).fit(X[b][tr][okt], y[tr][okt])
                    p[ok] = ul.full_proba(m, X[b][te][ok])
                Pte.append(p)
            ftr = np.stack([~np.isnan(P[:, 0]) for P in Ptr], 1).astype(int)
            fte = np.stack([~np.isnan(P[:, 0]) for P in Pte], 1).astype(int)
            Qtr, Qte = Qall[tr] * ftr, Qall[te] * fte

            # --- η προτεινόμενη: meta-classifier + modality dropout + αξιοπιστία
            copies = ul.dropout_copies(Ptr, ftr, Qtr, rng)
            yd = np.concatenate([y[tr]] * len(copies))
            meta = ul.fit_meta(np.vstack([ul.reliability_features(P, f, Q) for P, f, Q in copies]), yd)
            # Krishna et al.: meta-classifier + modality dropout, χωρίς ποιότητα σήματος / βεβαιότητα
            meta_drop = ul.fit_meta(np.vstack([ul.meta_features(P, f) for P, f, _ in copies]), yd)
            # κλασικό stacking: logistic regression στις πιθανότητες των κλάδων
            stack_lr = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000,
                                 class_weight="balanced"))]).fit(
                np.nan_to_num(ul.meta_features(Ptr, ftr), nan=1.0 / NC), y[tr])

            # --- πρώιμη σύντηξη (Bodaghi): RF και SVM με MinMax + PCA
            Xall = np.hstack([X[b] for b in BR])
            early_rf = Pipeline([("imp", SimpleImputer(strategy="median")),
                                 ("rf", RandomForestClassifier(n_estimators=trees, max_depth=8, min_samples_leaf=5,
                                                               class_weight="balanced", n_jobs=-1,
                                                               random_state=ul.SEED))]).fit(Xall[tr], y[tr])
            early_svm = Pipeline([("imp", SimpleImputer(strategy="median")), ("mm", MinMaxScaler()),
                                  ("pca", PCA(0.90, random_state=ul.SEED)),
                                  ("svm", SVC(kernel="rbf", C=0.1, gamma="auto", class_weight="balanced"))]
                                 ).fit(Xall[tr], y[tr])
            early_prf = Pipeline([("imp", SimpleImputer(strategy="median")), ("mm", MinMaxScaler()),
                                  ("pca", PCA(0.90, random_state=ul.SEED)),
                                  ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10,
                                                                n_jobs=-1, random_state=ul.SEED))]).fit(Xall[tr], y[tr])
            early_mlp = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                  ("mlp", MLPClassifier((64, 32), alpha=1e-3, max_iter=300, early_stopping=True,
                                                        random_state=ul.SEED))]).fit(Xall[tr], y[tr])
            cols = np.cumsum([0] + [X[b].shape[1] for b in BR])
            # Kontras et al.: κάθε τροπικότητα μόνη της (μόνο στο σενάριο all, στα διαθέσιμα παράθυρα)
            for bi, b in enumerate(BR):
                ok = ~np.isnan(Pte[bi][:, 0])
                for i, p in zip(te[ok], Pte[bi][ok].argmax(1)):
                    rec.append((space, f"unimodal_{b}", "all", s, int(y[i]), int(p)))

            for scen in ["all"] + [f"no_{b}" for b in BR]:
                if scen == "all":
                    P_, f_, Q_ = Pte, fte, Qte
                    Xe = Xall[te]
                else:
                    bi = BR.index(scen[3:])
                    P_, f_, Q_ = mask(Pte, fte, Qte, bi)
                    Xe = Xall[te].copy()
                    Xe[:, cols[bi]:cols[bi + 1]] = np.nan
                preds = {
                    PROPOSED: meta.predict(ul.reliability_features(P_, f_, Q_)),
                    "late_soft_voting": ul.soft_vote(P_),
                    "early_fusion_RF": early_rf.predict(Xe),
                    "early_paper_SVM": early_svm.predict(Xe),
                    "early_paper_RF": early_prf.predict(Xe),
                    "early_MLP": early_mlp.predict(Xe),
                    "late_hard_voting": ul.hard_vote(P_),
                    "late_meta_dropout": meta_drop.predict(ul.meta_features(P_, f_)),
                    "late_stacking_LR": stack_lr.predict(np.nan_to_num(ul.meta_features(P_, f_), nan=1.0 / NC)),
                }
                for mth, pr in preds.items():
                    for i, p in zip(te, pr):
                        rec.append((space, mth, scen, s, int(y[i]), int(p)))
        print(f"LOSO {si}/{len(subjects)}: νέος οδηγός {s}  ({time.time() - t0:.0f} s)")
    return pd.DataFrame(rec, columns=["space", "method", "scenario", "subject", "y_true", "y_pred"])


def metrics(g):
    yt, yp = g.y_true.to_numpy(), g.y_pred.to_numpy()
    alert = yt == 0
    return pd.Series({
        "accuracy": accuracy_score(yt, yp),
        "precision_macro": precision_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
        "recall_macro": recall_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
        "macro_F1": f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
        "recall_High": recall_score(yt, yp, labels=[2], average="macro", zero_division=0),
        "kappa_quadratic": cohen_kappa_score(yt, yp, weights="quadratic"),
        "false_alarm_rate": float((yp[alert] == 2).mean()) if alert.any() else np.nan,
    })


def per_subject(res):
    return (res.groupby(["space", "method", "scenario", "subject"])
            .apply(lambda g: f1_score(g.y_true, g.y_pred, average="macro", labels=[0, 1, 2], zero_division=0))
            .rename("F1").reset_index())


def compare(ps, a, b, scen="all"):
    """Wilcoxon ανά οδηγό: a=(space, method) έναντι b=(space, method)."""
    s = ps[ps.scenario == scen].pivot_table(index="subject", columns=["space", "method"], values="F1")
    d = (s[a] - s[b]).dropna()
    try:
        p = wilcoxon(d).pvalue if (d != 0).any() else np.nan
    except ValueError:
        p = np.nan
    return {"A": f"{a[1]} ({a[0]})", "B": f"{b[1]} ({b[0]})", "mean_F1_A": s[a].mean(), "mean_F1_B": s[b].mean(),
            "mean_diff": d.mean(), "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
            "losses": int((d < 0).sum()), "wilcoxon_p": p, "significant": bool(p < 0.05) if p == p else False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--out", default="results_unified")
    ap.add_argument("--minutes", type=int, default=4)
    ap.add_argument("--baseline", choices=["sessionA", "same"], default="sessionA")
    ap.add_argument("--trees", type=int, default=150)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = a3.load(a.features)
    print(f"{len(df)} παράθυρα, {df.subject.nunique()} οδηγοί · k = {a.minutes} λεπτά · baseline = {a.baseline}")
    res = run(df, a.minutes, a.baseline, a.trees)
    res.to_csv(f"{a.out}/unified_predictions.csv", index=False)

    summ = res.groupby(["scenario", "space", "method"]).apply(metrics).reset_index()
    summ.insert(3, "related_work", summ.method.map(WORK))
    summ.to_csv(f"{a.out}/unified_summary.csv", index=False, encoding="utf-8-sig")
    ps = per_subject(res)
    ps.to_csv(f"{a.out}/unified_per_subject.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== ΕΝΟΠΟΙΗΣΗ · νέος οδηγός (LOSO) · όλοι οι αισθητήρες ===")
    print(summ[summ.scenario == "all"].drop(columns="scenario").sort_values("macro_F1", ascending=False)
          .round(3).to_string(index=False))

    # η προτεινόμενη με εξατομίκευση έναντι: (α) του εαυτού της χωρίς, (β) κάθε άλλης μεθόδου με και χωρίς
    rows = [compare(ps, ("norm", PROPOSED), ("raw", PROPOSED))]
    for m in WORK:
        if m == PROPOSED:
            continue
        for sp in ("norm", "raw"):
            r = compare(ps, ("norm", PROPOSED), (sp, m))
            r["related_work"] = WORK[m]
            rows.append(r)
    st = pd.DataFrame(rows)
    st.to_csv(f"{a.out}/unified_stats.csv", index=False, encoding="utf-8-sig")
    print("\n=== Στατιστικός έλεγχος ανά οδηγό (Wilcoxon) ===")
    print(st.round(4).to_string(index=False))

    print("\n=== Άξονας 2 μετά την εξατομίκευση: λείπει ένας αισθητήρας (macro F1) ===")
    t = summ[~summ.method.str.startswith("unimodal")].pivot_table(index=["space", "method"], columns="scenario",
                                                                    values="macro_F1")
    t["F1_missing_mean"] = t[[f"no_{b}" for b in BR]].mean(1)
    t["retention_%"] = 100 * t.F1_missing_mean / t["all"]
    t.to_csv(f"{a.out}/unified_missing.csv", encoding="utf-8-sig")
    print(t.round(3).to_string())

    print(f"\nΑρχεία στον φάκελο '{a.out}': unified_summary.csv, unified_stats.csv, unified_missing.csv, "
          "unified_per_subject.csv, unified_predictions.csv")
    print("space: raw = χωρίς εξατομίκευση · norm = ατομική κανονικοποίηση (χωρίς ετικέτες)")
    print("false_alarm_rate = ποσοστό παραθύρων Alert που χαρακτηρίστηκαν High.")


if __name__ == "__main__":
    main()
