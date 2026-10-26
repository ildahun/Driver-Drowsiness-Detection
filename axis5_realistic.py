#!/usr/bin/env python3
"""
UL-DD · Επόμενο βήμα 2: ΕΚΠΑΙΔΕΥΣΗ ΜΕ ΡΕΑΛΙΣΤΙΚΕΣ ΒΛΑΒΕΣ, σε νέο οδηγό (LOSO + εξατομίκευση).

Μέχρι τώρα ο meta-classifier εκπαιδευόταν μόνο με «λείπει τυχαία ένα παράθυρο» (modality dropout).
Γι' αυτό (α) οι δείκτες αξιοπιστίας δεν είχαν ρόλο και (β) το χρονικό πλαίσιο κατέρρεε όταν ένας αισθητήρας
έλειπε σε ΟΛΗ την οδήγηση. Εδώ τα αντίγραφα εκπαίδευσης περιέχουν και:
  1. βλάβη σε συνεχόμενο διάστημα (4–40 λεπτά, ή όλη την οδήγηση)
  2. κακό σήμα: ο κλάδος δίνει θορυβώδεις πιθανότητες και ο δείκτης ποιότητας πέφτει
  3. LOCF με ΗΛΙΚΙΑ: στα μικρά κενά κρατάμε την τελευταία πρόβλεψη του κλάδου και λέμε στον κριτή πόσο παλιά είναι
     (αντί για τον σταθερό κανόνα «ποιότητα × 0,5»)

Μέθοδοι:
  late_meta_reliability          η προτεινόμενη, όπως πριν (χωρίς LOCF)
  late_meta_reliability_LOCF     η υβριδική, όπως πριν (LOCF έως 2 λεπτά, ποιότητα × 0,5)
  late_meta_realistic            ΝΕΟ: εκπαίδευση με ρεαλιστικές βλάβες + ηλικία LOCF
  late_meta_realistic_temporal   ΝΕΟ: το ίδιο + χρονικό πλαίσιο 2 παραθύρων (1 λεπτό)
  late_meta_dropout              Krishna et al., όπως πριν
  late_meta_dropout_realistic    Krishna et al. με τις ίδιες ρεαλιστικές βλάβες (δίκαιη σύγκριση: χωρίς ποιότητα/ηλικία)
  late_soft_voting               Das et al.
Σενάρια ελέγχου: all · random30/60 · outage4/16/40 (ένας αισθητήρας χαλάει συνεχόμενα) ·
                 noise16/40 (ένας αισθητήρας δίνει θορυβώδες σήμα· ο θόρυβος μπαίνει στα ΧΑΡΑΚΤΗΡΙΣΤΙΚΑ, όχι στις
                 πιθανότητες, για να μη «μαντεύει» ο κριτής τον τρόπο που τον εκπαιδεύσαμε)

Χρειάζεται στον ίδιο φάκελο: ul_dd_baseline.py, axis3_personalization.py, axis4_temporal.py, features.csv
Χρήση:
    python axis5_realistic.py --features features.csv --out results_realistic
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, recall_score

import ul_dd_baseline as ul
import axis3_personalization as a3
from axis4_temporal import segments, temporal_block, apply_mask

warnings.filterwarnings("ignore")
BR, NC = ul.BRANCHES, ul.N_CLASSES
WIN_MIN = 0.5                    # ένα παράθυρο = 30 s
PROPOSED = "late_meta_reliability"
NEW = "late_meta_realistic"
WORK = {PROPOSED: "Η παρούσα εργασία (όπως πριν)", "late_meta_reliability_LOCF": "Υβριδική (όπως πριν)",
        NEW: "ΝΕΟ: ρεαλιστικές βλάβες + ηλικία LOCF", "late_meta_realistic_temporal": "ΝΕΟ: + χρονικό πλαίσιο 1 λεπ.",
        "late_meta_dropout": "Krishna et al.", "late_meta_dropout_realistic": "Krishna et al. + ρεαλιστικές βλάβες",
        "late_soft_voting": "Das et al."}


# ----------------------------------------------------------------------------- βλάβες
def block_mask(df, seg, order, minutes, rng, rows=None):
    """Ένας τυχαίος αισθητήρας ανά οδήγηση χάνεται για `minutes` συνεχόμενα λεπτά (ή όλη την οδήγηση)."""
    n = len(df)
    drop = np.zeros((n, len(BR)), bool)
    L = int(round(minutes / WIN_MIN))
    sorted_idx = order
    for sg in np.unique(seg):
        idx = sorted_idx[seg[sorted_idx] == sg]
        if rows is not None and not np.isin(idx, rows).any():
            continue
        b = rng.integers(len(BR))
        if L >= len(idx):
            drop[idx, b] = True
        else:
            st = rng.integers(0, len(idx) - L + 1)
            drop[idx[st:st + L], b] = True
    return drop


def locf_with_age(P_list, flags, Q, seg, order, max_age):
    """Στα κενά: η τελευταία πρόβλεψη του κλάδου στην ίδια οδήγηση, αν είναι έως max_age λεπτά παλιά.
    Επιστρέφει P, flags, Q και την ηλικία (λεπτά / max_age, 0 = φρέσκια μέτρηση, 1 = δεν καλύφθηκε)."""
    P2 = [p.copy() for p in P_list]
    f2, Q2 = flags.copy(), Q.copy()
    age = np.zeros((len(flags), len(BR)))
    o = order
    for b in range(len(BR)):
        last, last_seg, last_t = None, None, 0
        for pos, i in enumerate(o):
            if last_seg != seg[i]:
                last, last_seg, cnt = None, seg[i], 0
            if f2[i, b]:
                last, cnt = P2[b][i].copy(), 0
                continue
            cnt += 1
            a = cnt * WIN_MIN
            if last is not None and a <= max_age:
                P2[b][i] = last
                f2[i, b] = 1
                Q2[i, b] = Q[i, b] if Q[i, b] > 0 else 0.0
                age[i, b] = a / max_age
            else:
                age[i, b] = 1.0
    return P2, f2, Q2, age


def degrade(P_list, Q, rows_b, rng):
    """Κακό σήμα σε επίπεδο πιθανοτήτων (μόνο για την εκπαίδευση): ανάμειξη με ομοιόμορφη + χαμηλή ποιότητα."""
    P2 = [p.copy() for p in P_list]
    Q2 = Q.copy()
    for b, rows in rows_b.items():
        if not rows.any():
            continue
        alpha = rng.uniform(0.4, 0.9, rows.sum())[:, None]
        P2[b][rows] = (1 - alpha) * P2[b][rows] + alpha / NC
        Q2[rows, b] = Q2[rows, b] * (1 - alpha[:, 0])
    return P2, Q2


def realistic_copies(Pfull, flags, Q, df, seg, order, rng, max_age):
    """Αντίγραφα εκπαίδευσης: τυχαία κενά, συνεχόμενες βλάβες διαφόρων διαρκειών, κακό σήμα, LOCF με ηλικία."""
    n = len(df)
    out = []
    # (α) πλήρη + τυχαία κενά (όπως πριν)
    out.append((Pfull, flags, Q, np.zeros((n, len(BR)))))
    out.append(locf_with_age(*apply_mask(Pfull, flags, Q, rng.random((n, len(BR))) < 0.3), seg, order, max_age))
    # (β) συνεχόμενες βλάβες
    for mins in (4, 16, 1000):
        out.append(locf_with_age(*apply_mask(Pfull, flags, Q, block_mask(df, seg, order, mins, rng)),
                                 seg, order, max_age))
    # (γ) κακό σήμα σε συνεχόμενα διαστήματα
    bm = block_mask(df, seg, order, 16, rng)
    P2, Q2 = degrade(Pfull, Q, {b: bm[:, b] for b in range(len(BR))}, rng)
    out.append((P2, flags, Q2, np.zeros((n, len(BR)))))
    return out


# ----------------------------------------------------------------------------- πείραμα
def run(df, k_min, baseline, trees, max_age, noise_sd):
    y = df.y.to_numpy(int)
    groups = df.subject.to_numpy()
    seg, order = segments(df)
    n = len(df)
    feat = {b: [c for c in df.columns if c.startswith(f"{b}__")] for b in BR}
    X = {b: a3.baseline_normalize(df[feat[b]].to_numpy(float), df, k_min, baseline) for b in BR}
    avail = {b: ~np.isnan(df[feat[b]].to_numpy(float)).all(axis=1) for b in BR}
    sd = {b: np.nanstd(X[b], axis=0) for b in BR}
    qcols = [f"q__{b}" for b in BR]
    Qall = (df[qcols].fillna(0.0).to_numpy(float) if all(c in df for c in qcols)
            else np.stack([avail[b] for b in BR], 1).astype(float))
    cal = a3.calib_mask(df, k_min)
    rng = np.random.default_rng(ul.SEED)
    srng = np.random.default_rng(ul.SEED + 7)          # ίδια σενάρια για όλες τις μεθόδους
    scen = {"all": np.zeros((n, len(BR)), bool)}
    for p in (30, 60):
        scen[f"random{p}"] = srng.random((n, len(BR))) < p / 100
    for m in (4, 16, 40):
        scen[f"outage{m}"] = block_mask(df, seg, order, m if m < 40 else 1000, srng)
    for m in (16, 40):
        scen[f"noise{m}"] = block_mask(df, seg, order, m if m < 40 else 1000, srng)
    noise = {b: srng.normal(0, 1, X[b].shape) for b in BR}
    rec = []

    for si, s in enumerate(sorted(np.unique(groups)), 1):
        t0 = time.time()
        tr = np.where(groups != s)[0]
        sub = np.where(groups == s)[0]
        te = np.where((groups == s) & ~cal)[0]
        if len(te) == 0:
            continue
        Pfull = [np.full((n, NC), np.nan) for _ in BR]
        models = {}
        for bi, b in enumerate(BR):
            Pfull[bi][tr] = ul.branch_oof(X[b][tr], y[tr], groups[tr], avail[b][tr], trees, cv="loso")
            ok, okt = avail[b][sub], avail[b][tr]
            if okt.sum() >= 10 and ok.any():
                models[b] = ul.branch_model(trees).fit(X[b][tr][okt], y[tr][okt])
                Pfull[bi][sub[ok]] = ul.full_proba(models[b], X[b][sub][ok])
        flags = np.stack([~np.isnan(P[:, 0]) for P in Pfull], 1).astype(int)
        Q = Qall * flags

        # --- εκπαίδευση
        base = [(Pfull, flags, Q)] + [apply_mask(Pfull, flags, Q, rng.random((n, len(BR))) < 0.2) for _ in range(2)]
        yb = np.concatenate([y[tr]] * len(base))
        meta = ul.fit_meta(np.vstack([ul.reliability_features([p[tr] for p in P], f[tr], q[tr]) for P, f, q in base]), yb)
        meta_drop = ul.fit_meta(np.vstack([ul.meta_features([p[tr] for p in P], f[tr]) for P, f, _ in base]), yb)
        real = realistic_copies(Pfull, flags, Q, df, seg, order, rng, max_age)
        yr = np.concatenate([y[tr]] * len(real))
        feats_r = lambda P, f, q, a: np.hstack([ul.reliability_features([p[tr] for p in P], f[tr], q[tr]), a[tr]])
        meta_r = ul.fit_meta(np.vstack([feats_r(*c) for c in real]), yr)
        meta_rt = ul.fit_meta(np.vstack([np.hstack([feats_r(*c), temporal_block(c[0], seg, order, 2)[tr]])
                                         for c in real]), yr)
        meta_dr = ul.fit_meta(np.vstack([ul.meta_features([p[tr] for p in P], f[tr]) for P, f, _, _ in real]), yr)

        # --- έλεγχος
        pos = {i: j for j, i in enumerate(sub)}
        ite = np.array([pos[i] for i in te])
        for sc, M in scen.items():
            P_t = [p.copy() for p in Pfull]
            Q_t = Q.copy()
            drop = np.zeros((n, len(BR)), bool)
            if sc.startswith("noise"):
                # θόρυβος στα χαρακτηριστικά του κλάδου και νέα πρόβλεψη από το μοντέλο του κλάδου
                for bi, b in enumerate(BR):
                    rows = sub[M[sub, bi] & avail[b][sub]]
                    if len(rows) and b in models:
                        Xn = X[b][rows] + noise_sd * sd[b] * noise[b][rows]
                        P_t[bi][rows] = ul.full_proba(models[b], Xn)
                        Q_t[rows, bi] = 0.3
            else:
                drop[sub] = M[sub]
            P_, f_, Q_ = apply_mask(P_t, flags, Q_t, drop)
            PL, fL, QL, aL = locf_with_age(P_, f_, Q_, seg, order, max_age)
            QLh = np.where((fL == 1) & (f_ == 0), 0.5 * Qall, QL)         # ο κανόνας της υβριδικής: ποιότητα × 0,5
            R = lambda P, f, q: ul.reliability_features([p[sub] for p in P], f[sub], q[sub])
            preds = {
                PROPOSED: meta.predict(R(P_, f_, Q_))[ite],
                "late_meta_reliability_LOCF": meta.predict(R(PL, fL, QLh))[ite],
                NEW: meta_r.predict(np.hstack([R(PL, fL, QL), aL[sub]]))[ite],
                "late_meta_realistic_temporal": meta_rt.predict(np.hstack([R(PL, fL, QL), aL[sub],
                                                                          temporal_block(PL, seg, order, 2)[sub]]))[ite],
                "late_meta_dropout": meta_drop.predict(ul.meta_features([p[sub] for p in P_], f_[sub]))[ite],
                "late_meta_dropout_realistic": meta_dr.predict(ul.meta_features([p[sub] for p in PL], fL[sub]))[ite],
                "late_soft_voting": ul.soft_vote([p[te] for p in P_]),
            }
            for mth, pr in preds.items():
                for i, p in zip(te, pr):
                    rec.append((mth, sc, s, int(y[i]), int(p)))
        print(f"LOSO {si}/{len(np.unique(groups))}: νέος οδηγός {s}  ({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(rec, columns=["method", "scenario", "subject", "y_true", "y_pred"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--out", default="results_realistic")
    ap.add_argument("--minutes", type=int, default=4)
    ap.add_argument("--baseline", choices=["sessionA", "same"], default="sessionA")
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--max-age", type=float, default=2.0, help="μέγιστη ηλικία LOCF σε λεπτά")
    ap.add_argument("--noise", type=float, default=1.5, help="θόρυβος σε τυπικές αποκλίσεις του χαρακτηριστικού")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = a3.load(a.features)
    print(f"{len(df)} παράθυρα, {df.subject.nunique()} οδηγοί · LOSO + εξατομίκευση · LOCF έως {a.max_age} λεπ. · "
          f"θόρυβος {a.noise} SD")
    res = run(df, a.minutes, a.baseline, a.trees, a.max_age, a.noise)
    res.to_csv(f"{a.out}/realistic_predictions.csv", index=False)

    def met(g):
        yt, yp = g.y_true.to_numpy(), g.y_pred.to_numpy()
        return pd.Series({"macro_F1": f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
                          "kappa": cohen_kappa_score(yt, yp, weights="quadratic"),
                          "false_alarm": float((yp[yt == 0] == 2).mean()) if (yt == 0).any() else np.nan})
    summ = res.groupby(["scenario", "method"]).apply(met).reset_index()
    summ.to_csv(f"{a.out}/realistic_summary.csv", index=False, encoding="utf-8-sig")
    order_sc = ["all", "random30", "random60", "outage4", "outage16", "outage40", "noise16", "noise40"]
    piv = summ.pivot_table(index="method", columns="scenario", values="macro_F1")[order_sc]
    piv.insert(0, "related_work", piv.index.map(WORK))
    piv = piv.sort_values("all", ascending=False)
    piv.to_csv(f"{a.out}/realistic_scenarios.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n=== Ρεαλιστικές βλάβες · νέος οδηγός · macro F1 ανά σενάριο ===")
    print(piv.round(3).to_string())

    ps = (res.groupby(["method", "scenario", "subject"])
          .apply(lambda g: f1_score(g.y_true, g.y_pred, average="macro", labels=[0, 1, 2], zero_division=0))
          .rename("F1").reset_index())
    ps.to_csv(f"{a.out}/realistic_per_subject.csv", index=False)
    rows = []
    for sc in order_sc:
        t = ps[ps.scenario == sc].pivot_table(index="subject", columns="method", values="F1")
        for m in t.columns:
            if m == NEW:
                continue
            d = (t[NEW] - t[m]).dropna()
            try:
                p = wilcoxon(d).pvalue if (d != 0).any() else np.nan
            except ValueError:
                p = np.nan
            rows.append({"scenario": sc, "baseline": m, "related_work": WORK.get(m, ""), "F1_new": t[NEW].mean(),
                         "F1_baseline": t[m].mean(), "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                         "losses": int((d < 0).sum()), "wilcoxon_p": p,
                         "significant": bool(p < 0.05) if p == p else False})
    st = pd.DataFrame(rows)
    st.to_csv(f"{a.out}/realistic_stats.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== Wilcoxon ανά οδηγό · {NEW} έναντι των άλλων ===")
    print(st.round(4).to_string(index=False))
    print(f"\nΑρχεία στον φάκελο '{a.out}': realistic_scenarios.csv, realistic_stats.csv, realistic_summary.csv, "
          "realistic_per_subject.csv, realistic_predictions.csv")


if __name__ == "__main__":
    main()
