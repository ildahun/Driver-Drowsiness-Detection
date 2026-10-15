#!/usr/bin/env python3
"""
UL-DD · Άξονας 2: ελλιπείς τροπικότητες.

(α) Καμπύλη ανθεκτικότητας: στον έλεγχο, κάθε κλάδος (βιομετρικά / όχημα / πρόσωπο) κάθε παραθύρου
    «χάνεται» ανεξάρτητα με πιθανότητα p = 0, 10, 20, 30, 40, 50, 60%. Τα μοντέλα εκπαιδεύονται μία φορά
    ανά fold με πλήρη δεδομένα· το p αλλάζει μόνο στον έλεγχο (3 επαναλήψεις ανά p).
(α2) Βλάβη αισθητήρα: σε κάθε οδήγηση ένας τυχαίος κλάδος χάνεται για ΣΥΝΕΧΟΜΕΝΟ διάστημα
    4, 8, 16 λεπτών ή όλη την οδήγηση (40 λεπτά) — ρεαλιστικό σενάριο, όπου η τελευταία γνωστή τιμή «παλιώνει».
(β) Πραγματικά κενά: απόδοση στα παράθυρα όπου κάποιος αισθητήρας πραγματικά έλειπε ή είχε κακό σήμα
    (ποιότητα < 0.5) έναντι των πλήρων παραθύρων, και ξεχωριστά στις οδηγήσεις A_D, N_D, O_D.

Μέθοδοι:
  late_meta_reliability (η προτεινόμενη) · late_meta_dropout (Krishna) · late_soft_voting / late_hard_voting (Das)
  late_stacking_LR · early_fusion_RF / early_paper_SVM (Bodaghi, ό,τι λείπει -> διάμεσος)
  early_fusion_RF_LOCF (Bodaghi + LOCF: ό,τι λείπει -> η τελευταία διαθέσιμη τιμή της ίδιας οδήγησης)
  Επιπλέον κλασικές μέθοδοι για ελλιπή δεδομένα:
    early_RF_zero (συμπλήρωση με 0) · early_RF_kNN (συμπλήρωση kNN, k=5) · early_MLP (νευρωνικό, διάμεσος)
    late_weighted_vote (ψηφοφορία σταθμισμένη με την ακρίβεια κάθε κλάδου)
    late_max_confidence (δυναμική επιλογή: αποφασίζει ο πιο σίγουρος διαθέσιμος κλάδος)
    late_meta_zero_dropout (meta-classifier, ο κλάδος που λείπει = 0, με dropout· αρχικό σχέδιο)
  late_meta_reliability_LOCF (η προτεινόμενη + LOCF): κενό έως --max-age λεπτά -> τελευταία τιμή του κλάδου
      (με μειωμένη ποιότητα σήματος)· μεγαλύτερο κενό -> ο κλάδος δηλώνεται ελλιπής και αποφασίζει ο meta-classifier

Χρειάζεται στον ίδιο φάκελο: ul_dd_baseline.py, features.csv
Χρήση:
    python axis2_missing.py --features features.csv --out results_axis2
    νέος οδηγός με εξατομίκευση (Άξονες 1+2+3):
    python axis2_missing.py --features features.csv --out results_axis2_loso_norm --cv loso --normalize
    (προαιρετικά: --trees 150, --repeats 3, --minutes 4, --baseline same)
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
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

import ul_dd_baseline as ul

warnings.filterwarnings("ignore")
BR = ul.BRANCHES
NC = ul.N_CLASSES
PROPOSED = "late_meta_reliability"
RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
REAL_GAP_SESSIONS = [("A", "D"), ("N", "D"), ("O", "D")]
OUTAGES = [4, 8, 16, 40]            # λεπτά συνεχόμενης βλάβης ενός αισθητήρα
HYBRID = PROPOSED + "_LOCF"
WORK = {PROPOSED: "Η παρούσα εργασία", HYBRID: "Η παρούσα εργασία + LOCF", "late_meta_dropout": "Krishna et al.",
        "late_soft_voting": "Das et al.", "late_hard_voting": "Das et al.", "late_stacking_LR": "Κλασικό stacking",
        "early_fusion_RF": "Bodaghi et al.", "early_paper_SVM": "Bodaghi et al.",
        "early_fusion_RF_LOCF": "Bodaghi et al. + LOCF", "early_MLP": "Bodaghi et al. (MLP)",
        "early_RF_zero": "Κλασική συμπλήρωση με 0", "early_RF_kNN": "Κλασική συμπλήρωση kNN",
        "late_weighted_vote": "Σταθμισμένη ψηφοφορία", "late_max_confidence": "Δυναμική επιλογή κλάδου",
        "late_meta_zero_dropout": "Zero imputation + dropout"}


def locf(Xfull, df):
    """Last observation carried forward μέσα σε κάθε οδήγηση (με χρονική σειρά)· ό,τι μένει κενό μένει NaN."""
    out = pd.DataFrame(Xfull, index=df.index)
    order = df.sort_values(["subject", "session", "interval", "window"]).index
    key = df.loc[order, ["subject", "session"]].astype(str).agg("_".join, axis=1)
    filled = out.loc[order].groupby(key.values).ffill()
    return filled.loc[df.index].to_numpy()


def locf_age(Xb, df):
    """LOCF για έναν κλάδο: συμπληρωμένος πίνακας + «ηλικία» (σε παράθυρα) της τιμής που αντιγράφηκε."""
    order = df.sort_values(["subject", "session", "interval", "window"]).index.to_numpy()
    key = (df.subject + "_" + df.session).to_numpy()
    valid = ~np.isnan(Xb).all(axis=1)
    filled = np.full_like(Xb, np.nan)
    age = np.full(len(Xb), np.inf)
    last_row, last_key, last_pos = None, None, None
    for pos, i in enumerate(order):
        if key[i] != last_key:
            last_row, last_key = None, key[i]
        if valid[i]:
            last_row, last_pos = i, pos
            filled[i], age[i] = Xb[i], 0
        elif last_row is not None:
            filled[i], age[i] = Xb[last_row], pos - last_pos
    return filled, age


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--out", default="results_axis2")
    ap.add_argument("--cv", choices=["kfold", "loso"], default="kfold")
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-age", type=float, default=2.0, help="μέγιστη ηλικία τιμής LOCF για την υβριδική (λεπτά)")
    ap.add_argument("--normalize", action="store_true",
                    help="ατομική κανονικοποίηση (Άξονας 3) πριν από όλα· τα λεπτά βαθμονόμησης εξαιρούνται από τον έλεγχο")
    ap.add_argument("--minutes", type=int, default=4, help="λεπτά βαθμονόμησης για το --normalize")
    ap.add_argument("--baseline", choices=["sessionA", "same"], default="sessionA")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.features, dtype={"subject": str, "session": str}, low_memory=False)
    num = df.select_dtypes("number").columns
    v = df[num].to_numpy(float)
    v[~np.isfinite(v) | (np.abs(np.nan_to_num(v, posinf=0, neginf=0)) > 1e9)] = np.nan
    df[num] = v
    y = df.y.to_numpy(int)
    groups = df.subject.to_numpy()
    feat = {b: [c for c in df.columns if c.startswith(f"{b}__")] for b in BR}
    X = {b: df[feat[b]].to_numpy(float) for b in BR}
    cal = np.zeros(len(df), bool)
    if a.normalize:                                          # Άξονας 3: ατομική κανονικοποίηση
        import axis3_personalization as a3
        n_w = int(df.window.max()) + 1
        df["t_s"] = df.interval * a3.INTERVAL_S + df.window * (a3.INTERVAL_S // n_w)
        X = {b: a3.baseline_normalize(X[b], df, a.minutes, a.baseline) for b in BR}
        cal = a3.calib_mask(df, a.minutes)
    Xall = np.hstack([X[b] for b in BR])
    cols = np.cumsum([0] + [X[b].shape[1] for b in BR])
    avail = {b: ~np.isnan(X[b]).all(axis=1) for b in BR}
    qcols = [f"q__{b}" for b in BR]
    Qall = (df[qcols].fillna(0.0).to_numpy(float) if all(c in df for c in qcols)
            else np.stack([avail[b] for b in BR], 1).astype(float))

    # πραγματικά κενά: κλάδος που λείπει ή με ποιότητα < 0.5
    real_gap = np.zeros(len(df), bool)
    for bi, b in enumerate(BR):
        real_gap |= ~avail[b] | (Qall[:, bi] < 0.5)
    sess_tag = (df.subject + "_" + df.session).to_numpy()
    print(f"{len(df)} παράθυρα, {df.subject.nunique()} οδηγοί · πρωτόκολλο = {a.cv}" +
          (f" · κανονικοποίηση: {a.minutes} λεπτά, baseline = {a.baseline} (εξαιρούνται {cal.sum()} παράθυρα "
           "βαθμονόμησης από τον έλεγχο)" if a.normalize else ""))
    print(f"Παράθυρα με πραγματικό κενό ή κακό σήμα: {real_gap.sum()} ({real_gap.mean():.1%})")
    for s, ss in REAL_GAP_SESSIONS:
        m = sess_tag == f"{s}_{ss}"
        if m.any():
            print(f"  {s}_{ss}: {m.sum()} παράθυρα, με κενό: {real_gap[m].sum()} · διαθεσιμότητα " +
                  ", ".join(f"{b} {avail[b][m].mean():.0%}" for b in BR))

    if a.cv == "loso":
        folds = [(np.where(groups != s)[0], np.where(groups == s)[0]) for s in sorted(np.unique(groups))]
    else:
        folds = list(StratifiedKFold(5, shuffle=True, random_state=ul.SEED).split(Xall, y))

    # μάσκες (ίδιες για όλες τις μεθόδους → δίκαιη σύγκριση)
    n_win = int(df.window.max()) + 1
    win_min = 4.0 / n_win                                    # διάρκεια παραθύρου σε λεπτά
    tw = (df.interval * n_win + df.window).to_numpy()        # αύξων αριθμός παραθύρου στην οδήγηση
    rng_m = np.random.default_rng(123)
    masks = {("random", p, r): (rng_m.random((len(df), len(BR))) < p) for p in RATES for r in range(a.repeats)}
    for L in OUTAGES:
        Lw = int(round(L / win_min))
        for r in range(a.repeats):
            M = np.zeros((len(df), len(BR)), bool)
            for key in np.unique(sess_tag):
                rows = sess_tag == key
                n = int(tw[rows].max()) + 1
                bi = rng_m.integers(len(BR))
                start = rng_m.integers(0, max(1, n - Lw + 1))
                M[rows & (tw >= start) & (tw < start + Lw), bi] = True
            masks[("outage", L, r)] = M
    max_age_w = a.max_age / win_min
    # LOCF για κάθε μάσκα: (α) για την πρώιμη σύντηξη, όλος ο πίνακας· (β) ανά κλάδο με ηλικία, για την υβριδική
    locf_cache = {}
    for key, M in masks.items():
        Xm = Xall.copy()
        per_branch = []
        for bi, b in enumerate(BR):
            Xm[M[:, bi], cols[bi]:cols[bi + 1]] = np.nan
            Xb = X[b].copy()
            Xb[M[:, bi]] = np.nan
            per_branch.append(locf_age(Xb, df))
        locf_cache[key] = (locf(Xm, df), per_branch)
    print(f"Μάσκες: {len(masks)} (τυχαίες + βλάβες {OUTAGES} λεπτών) · LOCF υβριδικής έως {a.max_age} λεπτά")

    rec_curve, rec_real = [], []
    rng = np.random.default_rng(ul.SEED)
    folds = [(tr, te[~cal[te]]) for tr, te in folds]          # έλεγχος μόνο μετά τη βαθμονόμηση
    for k, (tr, te) in enumerate(folds, 1):
        t0 = time.time()
        Ptr, Pte, bmodels = [], [], []
        for b in BR:
            Ptr.append(ul.branch_oof(X[b][tr], y[tr], groups[tr], avail[b][tr], a.trees, cv=a.cv))
            p = np.full((len(te), NC), np.nan)
            ok, okt = avail[b][te], avail[b][tr]
            m = ul.branch_model(a.trees).fit(X[b][tr][okt], y[tr][okt]) if okt.sum() >= 10 else None
            if m is not None and ok.any():
                p[ok] = ul.full_proba(m, X[b][te][ok])
            Pte.append(p)
            bmodels.append(m)
        ftr = np.stack([~np.isnan(P[:, 0]) for P in Ptr], 1).astype(int)
        fte = np.stack([~np.isnan(P[:, 0]) for P in Pte], 1).astype(int)
        Qtr, Qte = Qall[tr] * ftr, Qall[te] * fte

        copies = ul.dropout_copies(Ptr, ftr, Qtr, rng)
        yd = np.concatenate([y[tr]] * len(copies))
        meta_rel = ul.fit_meta(np.vstack([ul.reliability_features(P, f, Q) for P, f, Q in copies]), yd)
        meta_drop = ul.fit_meta(np.vstack([ul.meta_features(P, f) for P, f, _ in copies]), yd)
        stack_lr = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000,
                             class_weight="balanced"))]).fit(np.nan_to_num(ul.meta_features(Ptr, ftr), nan=1 / NC), y[tr])
        early_rf = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("rf", RandomForestClassifier(n_estimators=a.trees, max_depth=8, min_samples_leaf=5,
                                                           class_weight="balanced", n_jobs=-1,
                                                           random_state=ul.SEED))]).fit(Xall[tr], y[tr])
        early_svm = Pipeline([("imp", SimpleImputer(strategy="median")), ("mm", MinMaxScaler()),
                              ("pca", PCA(0.90, random_state=ul.SEED)),
                              ("svm", SVC(kernel="rbf", C=0.1, gamma="auto", class_weight="balanced"))]
                             ).fit(Xall[tr], y[tr])

        rf_kw = dict(n_estimators=a.trees, max_depth=8, min_samples_leaf=5, class_weight="balanced", n_jobs=-1,
                     random_state=ul.SEED)
        early_zero = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
                               ("rf", RandomForestClassifier(**rf_kw))]).fit(Xall[tr], y[tr])
        early_knn = Pipeline([("sc", StandardScaler()), ("imp", KNNImputer(n_neighbors=5)),
                              ("rf", RandomForestClassifier(**rf_kw))]).fit(Xall[tr], y[tr])
        early_mlp = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                              ("mlp", MLPClassifier((64, 32), alpha=1e-3, max_iter=300, early_stopping=True,
                                                    random_state=ul.SEED))]).fit(Xall[tr], y[tr])
        meta_zero = ul.fit_meta(ul.to_zero_input(np.vstack([ul.meta_features(P, f) for P, f, _ in copies])), yd)
        # βάρη σταθμισμένης ψηφοφορίας: ακρίβεια κάθε κλάδου στις OOF προβλέψεις του train
        wts = np.array([accuracy_score(y[tr][~np.isnan(P[:, 0])], P[~np.isnan(P[:, 0])].argmax(1))
                        if (~np.isnan(P[:, 0])).any() else 0.0 for P in Ptr])

        def weighted_vote(P_):
            S = np.zeros((len(P_[0]), NC))
            for w, P in zip(wts, P_):
                S += w * np.nan_to_num(P, nan=0.0)
            S[(S == 0).all(1)] = 1.0 / NC
            return S.argmax(1)

        def max_confidence(P_):
            st = np.stack([np.nan_to_num(P, nan=-1.0) for P in P_])        # (κλάδοι, n, C)
            best = st.max(2).argmax(0)                                    # ο πιο σίγουρος διαθέσιμος κλάδος
            out = st[best, np.arange(st.shape[1])].argmax(1)
            out[(st.max(2) < 0).all(0)] = 1
            return out

        def predict_all(P_, f_, Q_, Xe, Xe_locf):
            return {PROPOSED: meta_rel.predict(ul.reliability_features(P_, f_, Q_)),
                    "late_meta_dropout": meta_drop.predict(ul.meta_features(P_, f_)),
                    "late_soft_voting": ul.soft_vote(P_), "late_hard_voting": ul.hard_vote(P_),
                    "late_stacking_LR": stack_lr.predict(np.nan_to_num(ul.meta_features(P_, f_), nan=1 / NC)),
                    "early_fusion_RF": early_rf.predict(Xe), "early_paper_SVM": early_svm.predict(Xe),
                    "early_fusion_RF_LOCF": early_rf.predict(Xe_locf),
                    "early_MLP": early_mlp.predict(Xe), "early_RF_zero": early_zero.predict(Xe),
                    "early_RF_kNN": early_knn.predict(Xe), "late_weighted_vote": weighted_vote(P_),
                    "late_max_confidence": max_confidence(P_),
                    "late_meta_zero_dropout": meta_zero.predict(ul.to_zero_input(ul.meta_features(P_, f_)))}

        for key, M in masks.items():
            kind, lvl, r = key
            Mt = M[te]
            Xlocf_all, per_branch = locf_cache[key]
            P_ = [P.copy() for P in Pte]
            f_, Q_ = fte.copy(), Qte.copy()
            Xe = Xall[te].copy()
            for bi in range(len(BR)):
                P_[bi][Mt[:, bi]] = np.nan
                f_[Mt[:, bi], bi] = 0
                Q_[Mt[:, bi], bi] = 0.0
                Xe[Mt[:, bi], cols[bi]:cols[bi + 1]] = np.nan
            preds = predict_all(P_, f_, Q_, Xe, Xlocf_all[te])
            # υβριδική: σύντομα κενά (τεχνητά ή πραγματικά) -> LOCF στον κλάδο, μεγάλα -> ελλιπής κλάδος
            Ph = [P.copy() for P in P_]
            fh, Qh = f_.copy(), Q_.copy()
            for bi, b in enumerate(BR):
                filled, age = per_branch[bi]
                sel = (Mt[:, bi] | ~avail[b][te]) & (age[te] <= max_age_w) & ~np.isnan(filled[te]).all(1)
                if bmodels[bi] is not None and sel.any():
                    Ph[bi][sel] = ul.full_proba(bmodels[bi], filled[te][sel])
                    fh[sel, bi] = 1
                    Qh[sel, bi] = 0.5 * Qall[te][sel, bi]
            preds[HYBRID] = meta_rel.predict(ul.reliability_features(Ph, fh, Qh))
            for mth, pr in preds.items():
                for i, pp in zip(te, pr):
                    rec_curve.append((kind, lvl, r, mth, groups[i], int(y[i]), int(pp)))
                if kind == "random" and lvl == 0.0 and r == 0:   # πραγματικά δεδομένα, χωρίς τεχνητά κενά
                    for i, pp in zip(te, pr):
                        rec_real.append((mth, groups[i], sess_tag[i], bool(real_gap[i]), int(y[i]), int(pp)))
        print(f"{a.cv.upper()} fold {k}/{len(folds)} ({time.time() - t0:.0f} s)")

    f1 = lambda g: f1_score(g.y_true, g.y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    pd.set_option("display.width", 220)

    # (α) καμπύλη
    allrec = pd.DataFrame(rec_curve, columns=["kind", "p", "rep", "method", "subject", "y_true", "y_pred"])
    cur = allrec[allrec.kind == "random"]
    out_ = allrec[allrec.kind == "outage"]
    curve = (cur.groupby(["method", "p", "rep"]).apply(f1).rename("F1").reset_index()
             .groupby(["method", "p"]).F1.agg(["mean", "std"]).reset_index())
    tab = curve.pivot(index="method", columns="p", values="mean")
    tab.columns = [f"{int(100 * c)}%" for c in tab.columns]
    tab["retention_at_60%"] = 100 * tab["60%"] / tab["0%"]
    tab.insert(0, "related_work", tab.index.map(WORK))
    tab = tab.sort_values("60%", ascending=False)
    tab.to_csv(f"{a.out}/axis2_curve.csv", encoding="utf-8-sig")
    curve.to_csv(f"{a.out}/axis2_curve_long.csv", index=False)
    print("\n=== Άξονας 2 · macro F1 όταν λείπει τυχαία p% των κλάδων στον έλεγχο ===")
    print(tab.round(3).to_string())

    # (α2) βλάβη αισθητήρα για συνεχόμενο διάστημα
    oc = (out_.groupby(["method", "p", "rep"]).apply(f1).rename("F1").reset_index()
          .groupby(["method", "p"]).F1.mean().reset_index())
    otab = oc.pivot(index="method", columns="p", values="F1")
    otab.columns = [f"{int(c)} λεπτά" for c in otab.columns]
    otab.insert(0, "χωρίς_κενά", tab["0%"].reindex(otab.index))
    otab.insert(0, "related_work", otab.index.map(WORK))
    otab = otab.sort_values(otab.columns[-1], ascending=False)
    otab.to_csv(f"{a.out}/axis2_outage.csv", encoding="utf-8-sig")
    print("\n=== Άξονας 2 · βλάβη ενός αισθητήρα για συνεχόμενο διάστημα (macro F1) ===")
    print(otab.round(3).to_string())

    # Wilcoxon ανά οδηγό: τυχαία 30% / 60% και βλάβη 16 / 40 λεπτών
    rows = []
    for kind, lvl, label, src in [("random", 0.3, "τυχαία 30%", cur), ("random", 0.6, "τυχαία 60%", cur),
                                  ("outage", 16, "βλάβη 16 λεπτών", out_), ("outage", 40, "βλάβη 40 λεπτών", out_)]:
        ps = (src[src.p == lvl].groupby(["method", "subject"]).apply(f1).rename("F1").reset_index()
              .pivot(index="subject", columns="method", values="F1"))
        for ref in (PROPOSED, HYBRID):
            for m in ps.columns:
                if m in (PROPOSED, HYBRID) and m == ref or (ref == PROPOSED and m == HYBRID):
                    continue
                d = (ps[ref] - ps[m]).dropna()
                try:
                    pv = wilcoxon(d).pvalue if (d != 0).any() else np.nan
                except ValueError:
                    pv = np.nan
                rows.append({"scenario": label, "method": ref, "baseline": m, "related_work": WORK[m],
                             "F1_method": ps[ref].mean(), "F1_baseline": ps[m].mean(), "mean_diff": d.mean(),
                             "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
                             "wilcoxon_p": pv, "significant": bool(pv < 0.05) if pv == pv else False})
    st = pd.DataFrame(rows)
    st.to_csv(f"{a.out}/axis2_stats.csv", index=False, encoding="utf-8-sig")
    print("\n=== Στατιστικός έλεγχος ανά οδηγό (Wilcoxon) ===")
    print(st.round(4).to_string(index=False))

    # (β) πραγματικά κενά
    real = pd.DataFrame(rec_real, columns=["method", "subject", "session", "gap", "y_true", "y_pred"])
    out = []
    for mth, g in real.groupby("method"):
        row = {"method": mth, "related_work": WORK[mth],
               "F1_complete": f1(g[~g.gap]) if (~g.gap).any() else np.nan,
               "F1_real_gap": f1(g[g.gap]) if g.gap.any() else np.nan, "n_gap_windows": int(g.gap.sum())}
        for s, ss in REAL_GAP_SESSIONS:
            gs = g[g.session == f"{s}_{ss}"]
            row[f"acc_{s}_{ss}"] = accuracy_score(gs.y_true, gs.y_pred) if len(gs) else np.nan
        out.append(row)
    rt = pd.DataFrame(out).sort_values("F1_real_gap", ascending=False)
    rt.to_csv(f"{a.out}/axis2_real_gaps.csv", index=False, encoding="utf-8-sig")
    print("\n=== Πραγματικά κενά στα δεδομένα (χωρίς τεχνητή αφαίρεση) ===")
    print(rt.round(3).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5.5))
        order = tab.index.tolist()
        for m in order:
            c = curve[curve.method == m]
            lw, z = (3, 3) if m == PROPOSED else (1.6, 2)
            ax.errorbar(100 * c.p, c["mean"], yerr=c["std"], marker="o", lw=lw, zorder=z, capsize=3,
                        label=f"{m} ({WORK[m]})", color="#B4531A" if m == PROPOSED else None)
        ax.axhline(1 / 3, color="grey", ls=":", lw=1)
        ax.set_xlabel("Κλάδοι που λείπουν στον έλεγχο (%)")
        ax.set_ylabel("Macro F1")
        ax.set_title(f"Άξονας 2 · ανθεκτικότητα σε ελλιπείς αισθητήρες ({a.cv})")
        ax.grid(alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        fig.tight_layout()
        fig.savefig(f"{a.out}/axis2_curve.png", dpi=200)
    except Exception as e:  # το γράφημα είναι προαιρετικό
        print("[!] Δεν φτιάχτηκε γράφημα:", e)

    print(f"\nΑρχεία στον φάκελο '{a.out}': axis2_curve.csv, axis2_curve.png, axis2_outage.csv, "
          "axis2_stats.csv, axis2_real_gaps.csv")
    print("retention_at_60% = πόσο % της απόδοσης (χωρίς κενά) διατηρείται όταν λείπει το 60% των κλάδων.")


if __name__ == "__main__":
    main()
