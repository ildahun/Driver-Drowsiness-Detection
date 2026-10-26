#!/usr/bin/env python3
"""
UL-DD · Επόμενο βήμα 1: ΧΡΟΝΙΚΟ ΠΛΑΙΣΙΟ στον meta-classifier, σε νέο οδηγό (LOSO + εξατομίκευση).

Η κόπωση αλλάζει αργά (το KSS μετριέται κάθε 4 λεπτά), αλλά μέχρι τώρα κάθε παράθυρο 30 s κρινόταν μόνο του.
Εδώ ο meta-classifier βλέπει ΚΑΙ τις προβλέψεις των κλάδων στα k προηγούμενα παράθυρα της ίδιας οδήγησης
(μόνο προηγούμενα: έτσι θα δούλευε και σε πραγματικό χρόνο στο αυτοκίνητο).

Μέθοδοι (όλες με την ίδια εξατομίκευση, όπως στην ενοποίηση):
  late_meta_reliability        η προτεινόμενη (χωρίς χρόνο)
  late_meta_temporal_k<k>      η προτεινόμενη + μέσος όρος των πιθανοτήτων κάθε κλάδου στα k προηγούμενα παράθυρα
                               + πόσα από αυτά είχαν διαθέσιμο τον κλάδο           <-- ΝΕΟ
  late_meta_reliability_smooth η προτεινόμενη + εξομάλυνση της εξόδου στα k τελευταία παράθυρα
  late_meta_dropout            Krishna et al.
  late_meta_dropout_smooth     Krishna et al. + την ίδια εξομάλυνση (για δίκαιη σύγκριση)
  late_soft_voting             Das et al.
  late_stacking_LR             κλασικό stacking
Σενάρια ελέγχου: all · no_bio / no_veh / no_face (βλάβη σε όλη την οδήγηση) · random30 / random60 (τυχαία κενά).

Χρειάζεται στον ίδιο φάκελο: ul_dd_baseline.py, axis3_personalization.py, features.csv
Χρήση:
    python axis4_temporal.py --features features.csv --out results_temporal
    (προαιρετικά: --context 2,4,8  --minutes 4  --baseline sessionA  --trees 150)
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import ul_dd_baseline as ul
import axis3_personalization as a3

warnings.filterwarnings("ignore")
BR, NC = ul.BRANCHES, ul.N_CLASSES
PROPOSED = "late_meta_reliability"
WORK = {PROPOSED: "Η παρούσα εργασία", "late_meta_reliability_smooth": "Η παρούσα εργασία + εξομάλυνση",
        "late_meta_dropout": "Krishna et al.", "late_meta_dropout_smooth": "Krishna et al. + εξομάλυνση",
        "late_soft_voting": "Das et al.", "late_stacking_LR": "Κλασικό stacking"}


# ----------------------------------------------------------------------------- χρόνος
def segments(df):
    """Σειρά των παραθύρων στον χρόνο και ταυτότητα οδήγησης (οδηγός + session)."""
    seg = pd.factorize(df.subject.astype(str) + "_" + df.session.astype(str))[0]
    order = np.lexsort((df.t_s.to_numpy(), seg))
    return seg, order


def past_mean(M, seg, order, k):
    """Για κάθε γραμμή: μέσος όρος (αγνοώντας NaN) των k ΠΡΟΗΓΟΥΜΕΝΩΝ γραμμών της ίδιας οδήγησης,
    και πόσες από αυτές ήταν διαθέσιμες. M: (n, c) με NaN όπου λείπει."""
    n, c = M.shape
    d = pd.DataFrame(M[order], columns=range(c))
    g = d.groupby(seg[order], sort=False)
    mean = g.transform(lambda s: s.shift(1).rolling(k, min_periods=1).mean()).to_numpy()
    cnt = g[0].transform(lambda s: s.shift(1).notna().rolling(k, min_periods=1).sum()).fillna(0).to_numpy()
    out_m, out_c = np.full((n, c), np.nan), np.zeros(n)
    out_m[order], out_c[order] = mean, cnt
    return out_m, out_c


def smooth_with_past(Pcur, seg, order, k):
    """Εξομάλυνση της εξόδου: μέσος όρος της τρέχουσας και των k-1 προηγούμενων πιθανοτήτων."""
    d = pd.DataFrame(Pcur[order])
    sm = d.groupby(seg[order], sort=False).transform(lambda s: s.rolling(k, min_periods=1).mean()).to_numpy()
    out = np.empty_like(Pcur)
    out[order] = sm
    return out


def temporal_block(Pfull, seg, order, k):
    """Χρονικά χαρακτηριστικά: για κάθε κλάδο, οι μέσες πιθανότητες στα k προηγούμενα παράθυρα + το πλήθος τους."""
    cols = []
    for P in Pfull:
        m, c = past_mean(P, seg, order, k)
        cols += [np.nan_to_num(m, nan=1.0 / NC), (c / k)[:, None]]
    return np.hstack(cols)


# ----------------------------------------------------------------------------- κενά
def apply_mask(P_list, flags, Q, drop):
    """drop: (n, 3) boolean · ο κλάδος b «χάνεται» όπου drop[:, b]."""
    P2 = [p.copy() for p in P_list]
    f2, Q2 = flags.copy(), Q.copy()
    for b in range(len(BR)):
        P2[b][drop[:, b]] = np.nan
        f2[drop[:, b], b] = 0
        Q2[drop[:, b], b] = 0.0
    return P2, f2, Q2


def scenario_masks(n, rng):
    m = {"all": np.zeros((n, len(BR)), bool)}
    for bi, b in enumerate(BR):
        d = np.zeros((n, len(BR)), bool); d[:, bi] = True
        m[f"no_{b}"] = d
    for p in (30, 60):
        m[f"random{p}"] = rng.random((n, len(BR))) < p / 100
    return m


# ----------------------------------------------------------------------------- πείραμα
def run(df, k_min, baseline, trees, ctx):
    y = df.y.to_numpy(int)
    groups = df.subject.to_numpy()
    seg, order = segments(df)
    feat = {b: [c for c in df.columns if c.startswith(f"{b}__")] for b in BR}
    X = {b: a3.baseline_normalize(df[feat[b]].to_numpy(float), df, k_min, baseline) for b in BR}
    avail = {b: ~np.isnan(df[feat[b]].to_numpy(float)).all(axis=1) for b in BR}
    qcols = [f"q__{b}" for b in BR]
    Qall = (df[qcols].fillna(0.0).to_numpy(float) if all(c in df for c in qcols)
            else np.stack([avail[b] for b in BR], 1).astype(float))
    cal = a3.calib_mask(df, k_min)
    rng = np.random.default_rng(ul.SEED)
    masks_all = scenario_masks(len(df), np.random.default_rng(ul.SEED + 1))   # ίδια κενά για όλες τις μεθόδους
    n = len(df)
    rec = []

    for si, s in enumerate(sorted(np.unique(groups)), 1):
        t0 = time.time()
        tr = np.where(groups != s)[0]
        sub = np.where(groups == s)[0]                    # όλη η οδήγηση του νέου οδηγού (και τα λεπτά βαθμονόμησης,
        te = np.where((groups == s) & ~cal)[0]            # χωρίς ετικέτες, ως παρελθόν) · αξιολόγηση ΜΟΝΟ μετά
        if len(te) == 0:
            continue
        # --- κλάδοι: OOF στο train, μοντέλο για τον νέο οδηγό
        Pfull = [np.full((n, NC), np.nan) for _ in BR]
        for bi, b in enumerate(BR):
            Pfull[bi][tr] = ul.branch_oof(X[b][tr], y[tr], groups[tr], avail[b][tr], trees, cv="loso")
            ok, okt = avail[b][sub], avail[b][tr]
            if okt.sum() >= 10 and ok.any():
                m = ul.branch_model(trees).fit(X[b][tr][okt], y[tr][okt])
                Pfull[bi][sub[ok]] = ul.full_proba(m, X[b][sub][ok])
        flags = np.stack([~np.isnan(P[:, 0]) for P in Pfull], 1).astype(int)
        Q = Qall * flags

        # --- εκπαίδευση: modality dropout · τα χρονικά χαρακτηριστικά από το ίδιο «χαλασμένο» αντίγραφο
        copies = [(Pfull, flags, Q)]
        for _ in range(2):
            copies.append(apply_mask(Pfull, flags, Q, rng.random((n, len(BR))) < 0.2))
        yd = np.concatenate([y[tr]] * len(copies))
        meta = ul.fit_meta(np.vstack([ul.reliability_features([p[tr] for p in P], f[tr], q[tr])
                                      for P, f, q in copies]), yd)
        meta_drop = ul.fit_meta(np.vstack([ul.meta_features([p[tr] for p in P], f[tr]) for P, f, _ in copies]), yd)
        meta_t = {}
        for k in ctx:
            Xt = [np.hstack([ul.reliability_features([p[tr] for p in P], f[tr], q[tr]),
                             temporal_block(P, seg, order, k)[tr]]) for P, f, q in copies]
            meta_t[k] = ul.fit_meta(np.vstack(Xt), yd)
        stack_lr = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000,
                             class_weight="balanced"))]).fit(
            np.nan_to_num(ul.meta_features([p[tr] for p in Pfull], flags[tr]), nan=1.0 / NC), y[tr])

        # --- έλεγχος σε κάθε σενάριο (τα κενά ισχύουν σε όλη την οδήγηση του νέου οδηγού)
        for scen, M in masks_all.items():
            drop = np.zeros((n, len(BR)), bool); drop[sub] = M[sub]
            P_, f_, Q_ = apply_mask(Pfull, flags, Q, drop)
            Rs = ul.reliability_features([p[sub] for p in P_], f_[sub], Q_[sub])
            Ms = ul.meta_features([p[sub] for p in P_], f_[sub])
            pos = {i: j for j, i in enumerate(sub)}
            ite = np.array([pos[i] for i in te])
            pr_meta = np.zeros((n, NC)); pr_meta[sub] = meta.predict_proba(Rs)
            pr_drop = np.zeros((n, NC)); pr_drop[sub] = meta_drop.predict_proba(Ms)
            k0 = max(ctx) if 4 not in ctx else 4
            preds = {
                PROPOSED: pr_meta[te].argmax(1),
                "late_meta_reliability_smooth": smooth_with_past(pr_meta, seg, order, k0)[te].argmax(1),
                "late_meta_dropout": pr_drop[te].argmax(1),
                "late_meta_dropout_smooth": smooth_with_past(pr_drop, seg, order, k0)[te].argmax(1),
                "late_soft_voting": ul.soft_vote([p[te] for p in P_]),
                "late_stacking_LR": stack_lr.predict(np.nan_to_num(Ms[ite], nan=1.0 / NC)),
            }
            for k in ctx:
                Tb = temporal_block(P_, seg, order, k)[sub]
                preds[f"late_meta_temporal_k{k}"] = meta_t[k].predict(np.hstack([Rs, Tb]))[ite]
            for mth, pr in preds.items():
                for i, p in zip(te, pr):
                    rec.append((mth, scen, s, int(y[i]), int(p)))
        print(f"LOSO {si}/{len(np.unique(groups))}: νέος οδηγός {s}  ({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(rec, columns=["method", "scenario", "subject", "y_true", "y_pred"])


def metrics(g):
    yt, yp = g.y_true.to_numpy(), g.y_pred.to_numpy()
    alert = yt == 0
    return pd.Series({"accuracy": accuracy_score(yt, yp),
                      "macro_F1": f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
                      "recall_High": recall_score(yt, yp, labels=[2], average="macro", zero_division=0),
                      "kappa_quadratic": cohen_kappa_score(yt, yp, weights="quadratic"),
                      "false_alarm_rate": float((yp[alert] == 2).mean()) if alert.any() else np.nan})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--out", default="results_temporal")
    ap.add_argument("--context", default="2,4,8", help="πόσα προηγούμενα παράθυρα των 30 s (4 = 2 λεπτά)")
    ap.add_argument("--minutes", type=int, default=4)
    ap.add_argument("--baseline", choices=["sessionA", "same"], default="sessionA")
    ap.add_argument("--trees", type=int, default=150)
    a = ap.parse_args()
    ctx = [int(x) for x in a.context.split(",")]
    os.makedirs(a.out, exist_ok=True)
    for k in ctx:
        WORK[f"late_meta_temporal_k{k}"] = f"Η παρούσα εργασία + χρόνος ({k} παράθυρα = {k * 0.5:g} λεπ.)"

    df = a3.load(a.features)
    print(f"{len(df)} παράθυρα, {df.subject.nunique()} οδηγοί · LOSO + εξατομίκευση ({a.minutes} λεπτά, "
          f"baseline = {a.baseline}) · χρονικό πλαίσιο: {ctx} προηγούμενα παράθυρα")
    res = run(df, a.minutes, a.baseline, a.trees, ctx)
    res.to_csv(f"{a.out}/temporal_predictions.csv", index=False)

    summ = res.groupby(["scenario", "method"]).apply(metrics).reset_index()
    summ.insert(2, "related_work", summ.method.map(WORK))
    summ.to_csv(f"{a.out}/temporal_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n=== Χρονικό πλαίσιο · νέος οδηγός · όλοι οι αισθητήρες ===")
    print(summ[summ.scenario == "all"].drop(columns="scenario").sort_values("macro_F1", ascending=False)
          .round(3).to_string(index=False))

    piv = summ.pivot_table(index="method", columns="scenario", values="macro_F1")
    piv["no_mean"] = piv[[f"no_{b}" for b in BR]].mean(1)
    piv = piv[["all", "no_mean", "random30", "random60"] + [f"no_{b}" for b in BR]].sort_values("all", ascending=False)
    piv.to_csv(f"{a.out}/temporal_scenarios.csv", encoding="utf-8-sig")
    print("\n=== macro F1 ανά σενάριο (no_mean = μέσος όρος όταν λείπει ένας αισθητήρας) ===")
    print(piv.round(3).to_string())

    # Wilcoxon ανά οδηγό: η καλύτερη χρονική εκδοχή έναντι όλων
    ps = (res.groupby(["method", "scenario", "subject"])
          .apply(lambda g: f1_score(g.y_true, g.y_pred, average="macro", labels=[0, 1, 2], zero_division=0))
          .rename("F1").reset_index())
    ps.to_csv(f"{a.out}/temporal_per_subject.csv", index=False)
    best = piv.loc[[f"late_meta_temporal_k{k}" for k in ctx], "all"].idxmax()
    rows = []
    for scen in ["all", "no_bio", "no_veh", "no_face", "random60"]:
        t = ps[ps.scenario == scen].pivot_table(index="subject", columns="method", values="F1")
        for m in t.columns:
            if m == best:
                continue
            d = (t[best] - t[m]).dropna()
            try:
                p = wilcoxon(d).pvalue if (d != 0).any() else np.nan
            except ValueError:
                p = np.nan
            rows.append({"scenario": scen, "method": best, "baseline": m, "related_work": WORK.get(m, ""),
                         "F1_method": t[best].mean(), "F1_baseline": t[m].mean(), "wins": int((d > 0).sum()),
                         "ties": int((d == 0).sum()), "losses": int((d < 0).sum()), "wilcoxon_p": p,
                         "significant": bool(p < 0.05) if p == p else False})
    st = pd.DataFrame(rows)
    st.to_csv(f"{a.out}/temporal_stats.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== Wilcoxon ανά οδηγό · {best} έναντι των άλλων ===")
    print(st.drop(columns=["method"]).round(4).to_string(index=False))
    print(f"\nΑρχεία στον φάκελο '{a.out}': temporal_summary.csv, temporal_scenarios.csv, temporal_stats.csv, "
          "temporal_per_subject.csv, temporal_predictions.csv")


if __name__ == "__main__":
    main()
