#!/usr/bin/env python3
"""
UL-DD · Βήμα 4: ΔΕΙΚΤΕΣ ΑΞΙΟΠΙΣΤΙΑΣ ΠΟΥ ΚΑΤΑΛΑΒΑΙΝΟΥΝ ΤΟ ΧΑΛΑΣΜΕΝΟ ΣΗΜΑ, σε νέο οδηγό (LOSO + εξατομίκευση).

Πρόβλημα: ο δείκτης ποιότητας q__* μετράει μόνο πόσα δείγματα ΛΕΙΠΟΥΝ. Όταν ο αισθητήρας δίνει θόρυβο, τα δείγματα
υπάρχουν όλα και ο δείκτης λέει «όλα καλά» · ο meta-classifier εμπιστεύεται έναν κλάδο σίγουρο αλλά λάθος.
Γι' αυτό μέχρι τώρα η μέθοδος δεν ξεχώριζε από τον Krishna.

Νέοι δείκτες αξιοπιστίας (ανά κλάδο, ανά παράθυρο):
  1. ανωμαλία: πόσο έξω από τα φυσιολογικά όρια είναι τα (εξατομικευμένα) χαρακτηριστικά του κλάδου
     (z ως προς διάμεσο / IQR των οδηγών εκπαίδευσης · ποσοστό |z| > 3 και μέσο |z|)
  2. συμφωνία: πόσο συμφωνεί η πρόβλεψη του κλάδου με τους άλλους διαθέσιμους κλάδους
Στην εκπαίδευση προστίθεται ΕΝΑ είδος βλάβης σήματος: γκαουσιανός θόρυβος στα χαρακτηριστικά, χωρίς να πέφτει ο
δείκτης ποιότητας. Ο έλεγχος γίνεται ΚΑΙ σε είδη βλάβης που δεν είδε ποτέ (κορυφές, ολίσθηση, πάγωμα).

Μέθοδοι:
  late_meta_anomaly             ΝΕΟ: ρεαλιστικές βλάβες + θόρυβος χαρακτηριστικών + ανωμαλία + συμφωνία + χρόνος 1′
  late_reliability_vote         ΝΕΟ: ψηφοφορία με βάρος κάθε κλάδου (1 − ανωμαλία)² · χωρίς εκπαίδευση
  late_meta_anomaly_blend       ΝΕΟ: μέσος όρος των δύο παραπάνω (σταθερό 0,5 · όχι ρύθμιση)
  late_meta_realistic_temporal  το καλύτερο του βήματος 2 (axis5)
  late_meta_dropout_realistic   Krishna et al. + ρεαλιστικές βλάβες (όπως στο axis5)
  late_meta_dropout_robust      Krishna et al. + ΙΔΙΑ εκπαίδευση με τη νέα (και θόρυβο χαρακτηριστικών) · χωρίς δείκτες
  late_soft_voting              Das et al.
Σενάρια ελέγχου (40 = όλη η οδήγηση του νέου οδηγού, σε έναν τυχαίο αισθητήρα):
  all · random60 · outage40 · noise16 · noise40  (ίδια με το axis5: θόρυβος 1,5 SD, ποιότητα = 0,3)
  noise40_blind  ο ίδιος θόρυβος αλλά ο δείκτης ποιότητας ΔΕΝ πέφτει (όπως στην πράξη)
  spike40        τυχαίες κορυφές ±5 SD στο 20% των χαρακτηριστικών          (είδος που δεν είδε στην εκπαίδευση)
  drift40        ολίσθηση που φτάνει σταδιακά τα 3 SD                         (είδος που δεν είδε στην εκπαίδευση)
  frozen40       ο αισθητήρας «παγώνει» στην πρώτη τιμή                       (είδος που δεν είδε στην εκπαίδευση)
Στα blind / spike / drift / frozen ο δείκτης ποιότητας μένει ως έχει.

Χρειάζεται στον ίδιο φάκελο: ul_dd_baseline.py, axis3_personalization.py, axis4_temporal.py, axis5_realistic.py,
features.csv
Χρήση:
    python axis6_reliability.py --features features.csv --out results_reliability
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold

import ul_dd_baseline as ul
import axis3_personalization as a3
from axis4_temporal import segments, temporal_block, apply_mask
from axis5_realistic import block_mask, degrade

warnings.filterwarnings("ignore")
BR, NC = ul.BRANCHES, ul.N_CLASSES
NB = len(BR)
WIN_MIN = 0.5
NEW = "late_meta_anomaly"
NEWS = [NEW, "late_reliability_vote", "late_meta_anomaly_blend"]
REFS = ["late_meta_dropout_robust", "late_meta_dropout_realistic", "late_soft_voting", "late_meta_realistic_temporal"]
WORK = {NEW: "ΝΕΟ: δείκτες ανωμαλίας + συμφωνίας", "late_reliability_vote": "ΝΕΟ: ψηφοφορία με βάρη αξιοπιστίας",
        "late_meta_anomaly_blend": "ΝΕΟ: meta + ψηφοφορία αξιοπιστίας",
        "late_meta_realistic_temporal": "Βήμα 2: ρεαλιστικές + χρόνος",
        "late_meta_dropout_realistic": "Krishna et al. + ρεαλιστικές βλάβες",
        "late_meta_dropout_robust": "Krishna et al. + ίδια εκπαίδευση με τη νέα",
        "late_soft_voting": "Das et al."}
ORDER_SC = ["all", "random60", "outage40", "noise16", "noise40", "noise40_blind", "spike40", "drift40", "frozen40"]


# ----------------------------------------------------------------------------- δείκτες αξιοπιστίας
def robust_stats(Xtr):
    med = np.nanmedian(Xtr, axis=0)
    q1, q3 = np.nanpercentile(Xtr, 25, axis=0), np.nanpercentile(Xtr, 75, axis=0)
    scale = (q3 - q1) / 1.349
    sd = np.nanstd(Xtr, axis=0)
    scale = np.where(scale > 1e-9, scale, np.where(sd > 1e-9, sd, 1.0))
    return np.nan_to_num(med), scale


def anomaly(Xb, med, scale):
    """(n, 2): ποσοστό χαρακτηριστικών με |z| > 3 και μέσο |z| / 10 (με όριο 10). 0 όπου λείπει ο κλάδος."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = np.abs((Xb - med) / scale)
        frac = np.nanmean(z > 3, axis=1)
        frac = np.where(np.isnan(Xb).all(1), np.nan, frac)
        mz = np.nanmean(np.minimum(z, 10.0), axis=1) / 10.0
    return np.nan_to_num(np.stack([frac, mz], 1), nan=0.0)


def agreement(P_list, flags):
    """(n, 3): 1 − ½·L1 ανάμεσα στον κλάδο και τον μέσο όρο των άλλων διαθέσιμων · 0,5 αν δεν υπάρχει σύγκριση."""
    n = len(flags)
    out = np.full((n, NB), 0.5)
    S = np.stack([np.nan_to_num(P, nan=0.0) for P in P_list])           # (3, n, C)
    F = flags.astype(float)
    for b in range(NB):
        others = [o for o in range(NB) if o != b]
        cnt = F[:, others].sum(1)
        m = sum(S[o] * F[:, [o]] for o in others) / np.maximum(cnt, 1)[:, None]
        ok = (F[:, b] == 1) & (cnt > 0)
        out[ok, b] = 1 - 0.5 * np.abs(S[b][ok] - m[ok]).sum(1)
    return out


def locf_ext(P_list, flags, Q, A, seg, order, max_age):
    """Όπως το locf_with_age του axis5, αλλά μεταφέρει και τους δείκτες ανωμαλίας A (n, 3, 2)."""
    P2 = [p.copy() for p in P_list]
    f2, Q2, A2 = flags.copy(), Q.copy(), A.copy()
    age = np.zeros((len(flags), NB))
    for b in range(NB):
        last, lastA, last_seg, cnt = None, None, None, 0
        for i in order:
            if last_seg != seg[i]:
                last, lastA, last_seg, cnt = None, None, seg[i], 0
            if f2[i, b]:
                last, lastA, cnt = P2[b][i].copy(), A2[i, b].copy(), 0
                continue
            cnt += 1
            a = cnt * WIN_MIN
            if last is not None and a <= max_age:
                P2[b][i], A2[i, b] = last, lastA
                f2[i, b] = 1
                age[i, b] = a / max_age
            else:
                age[i, b] = 1.0
    return P2, f2, Q2, A2, age


def mask_A(A, drop):
    A2 = A.copy()
    A2[drop] = 0.0
    return A2


def rel_weights(A, flags):
    return flags * (1 - np.clip(A[:, :, 0], 0, 1)) ** 2


def rel_vote_proba(P_list, A, flags):
    w = rel_weights(A, flags)
    S = sum(np.nan_to_num(P, nan=0.0) * w[:, [b]] for b, P in enumerate(P_list))
    tot = w.sum(1)
    S = S / np.maximum(tot, 1e-9)[:, None]
    S[tot <= 1e-9] = 1.0 / NC
    return S


def soft_vote_proba(P_list):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        avg = np.nanmean(np.stack(P_list), axis=0)
    avg[np.isnan(avg).all(1)] = 1.0 / NC
    return avg


# ----------------------------------------------------------------------------- βλάβες στα χαρακτηριστικά
def corrupt(Xb, rows, kind, sd, rng, seg, order, noise=None, noise_sd=1.5):
    """Επιστρέφει αντίγραφο του Xb με βλάβη στις γραμμές rows (boolean, μήκους n)."""
    Xc = Xb.copy()
    if not rows.any():
        return Xc
    if kind == "noise":
        Xc[rows] = Xb[rows] + noise_sd * sd * noise[rows]
    elif kind == "train_noise":                       # μόνο για την εκπαίδευση: ένταση 0,5–2,5 SD
        k = rng.uniform(0.5, 2.5, rows.sum())[:, None]
        Xc[rows] = Xb[rows] + k * sd * rng.normal(0, 1, (rows.sum(), Xb.shape[1]))
    elif kind == "spike":
        hit = rng.random((rows.sum(), Xb.shape[1])) < 0.2
        sign = np.where(rng.random(hit.shape) < 0.5, -1.0, 1.0)
        Xc[rows] = Xb[rows] + hit * sign * 5.0 * sd
    elif kind in ("drift", "frozen"):
        for sg in np.unique(seg[rows]):
            idx = order[(seg[order] == sg) & rows[order]]          # οι γραμμές της βλάβης, με χρονική σειρά
            if kind == "drift":
                ramp = np.linspace(0, 1, len(idx))[:, None]
                Xc[idx] = Xb[idx] + 3.0 * sd * ramp
            else:
                Xc[idx] = Xb[idx[0]]
    return Xc


def oof_multi(Xs, y, groups, avail, trees):
    """OOF προβλέψεις κλάδου για το καθαρό Xs[0] ΚΑΙ για αλλοιωμένες εκδοχές Xs[1:], με τα ίδια μοντέλα των folds."""
    out = [np.full((len(y), NC), np.nan) for _ in Xs]
    splits = GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(Xs[0], y, groups)
    for tr, va in splits:
        tr, va = tr[avail[tr]], va[avail[va]]
        if len(tr) == 0 or len(va) == 0:
            continue
        m = ul.branch_model(trees).fit(Xs[0][tr], y[tr])
        for P, Xv in zip(out, Xs):
            P[va] = ul.full_proba(m, Xv[va])
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
    # σενάρια: ίδια σειρά κλήσεων με το axis5 → ίδιες μάσκες και ίδιος θόρυβος στα κοινά σενάρια
    srng = np.random.default_rng(ul.SEED + 7)
    s5 = {"all": np.zeros((n, NB), bool)}
    for p in (30, 60):
        s5[f"random{p}"] = srng.random((n, NB)) < p / 100
    for m in (4, 16, 40):
        s5[f"outage{m}"] = block_mask(df, seg, order, m if m < 40 else 1000, srng)
    for m in (16, 40):
        s5[f"noise{m}"] = block_mask(df, seg, order, m if m < 40 else 1000, srng)
    noise = {b: srng.normal(0, 1, X[b].shape) for b in BR}
    xrng = np.random.default_rng(ul.SEED + 11)
    scen = {k: s5[k] for k in ("all", "random60", "outage40", "noise16", "noise40")}
    scen["noise40_blind"] = s5["noise40"]
    for kind in ("spike", "drift", "frozen"):
        scen[f"{kind}40"] = block_mask(df, seg, order, 1000, xrng)
    test_rng = {k: np.random.default_rng(ul.SEED + 100 + i) for i, k in enumerate(scen)}
    rec = []

    for si, s in enumerate(sorted(np.unique(groups)), 1):
        t0 = time.time()
        tr = np.where(groups != s)[0]
        sub = np.where(groups == s)[0]
        te = np.where((groups == s) & ~cal)[0]
        if len(te) == 0:
            continue
        trm = np.zeros(n, bool); trm[tr] = True
        # --- βλάβες σήματος για την εκπαίδευση (μόνο στους οδηγούς εκπαίδευσης)
        fb = [block_mask(df, seg, order, 16, rng), block_mask(df, seg, order, 1000, rng)]
        stats = {b: robust_stats(X[b][tr]) for b in BR}
        Pfull = [np.full((n, NC), np.nan) for _ in BR]
        Pnoisy = [[np.full((n, NC), np.nan) for _ in BR] for _ in fb]
        A = np.zeros((n, NB, 2))
        An = [np.zeros((n, NB, 2)) for _ in fb]
        models = {}
        for bi, b in enumerate(BR):
            Xn = [corrupt(X[b], trm & fbk[:, bi] & avail[b], "train_noise", sd[b], rng, seg, order) for fbk in fb]
            outs = oof_multi([X[b][tr]] + [x[tr] for x in Xn], y[tr], groups[tr], avail[b][tr], trees)
            Pfull[bi][tr] = outs[0]
            for k in range(len(fb)):
                Pnoisy[k][bi][tr] = outs[k + 1]
                An[k][:, bi] = anomaly(Xn[k], *stats[b])
            A[:, bi] = anomaly(X[b], *stats[b])
            ok, okt = avail[b][sub], avail[b][tr]
            if okt.sum() >= 10 and ok.any():
                models[b] = ul.branch_model(trees).fit(X[b][tr][okt], y[tr][okt])
                Pfull[bi][sub[ok]] = ul.full_proba(models[b], X[b][sub][ok])
        flags = np.stack([~np.isnan(P[:, 0]) for P in Pfull], 1).astype(int)
        Q = Qall * flags
        A = A * flags[:, :, None]

        # --- αντίγραφα εκπαίδευσης: ίδια με το axis5 (κενά, βλάβες, κακό σήμα, LOCF) + θόρυβος χαρακτηριστικών
        copies = []                                           # (P, f, Q, A, age)
        copies.append((Pfull, flags, Q, A, np.zeros((n, NB))))
        drops = [rng.random((n, NB)) < 0.3] + [block_mask(df, seg, order, m, rng) for m in (4, 16, 1000)]
        for d in drops:
            P_, f_, Q_ = apply_mask(Pfull, flags, Q, d)
            copies.append(locf_ext(P_, f_, Q_, mask_A(A, d), seg, order, max_age))
        bm = block_mask(df, seg, order, 16, rng)
        P2, Q2 = degrade(Pfull, Q, {b: bm[:, b] for b in range(NB)}, rng)
        copies.append((P2, flags, Q2, A, np.zeros((n, NB))))
        for k in range(len(fb)):
            Pk = [np.where(np.isnan(Pnoisy[k][b]), Pfull[b], Pnoisy[k][b]) for b in range(NB)]
            copies.append((Pk, flags, Q, An[k] * flags[:, :, None], np.zeros((n, NB))))
        yc = np.concatenate([y[tr]] * len(copies))

        def feats_new(P, f, q, Aa, age, rows):
            Pr = [p[rows] for p in P]
            return np.hstack([ul.reliability_features(Pr, f[rows], q[rows]), age[rows],
                              Aa[rows].reshape(len(rows), -1), agreement(Pr, f[rows]),
                              temporal_block(P, seg, order, 2)[rows]])

        meta_new = ul.fit_meta(np.vstack([feats_new(*c, tr) for c in copies]), yc)
        meta_kr = ul.fit_meta(np.vstack([ul.meta_features([p[tr] for p in c[0]], c[1][tr]) for c in copies]), yc)
        # τα δύο μοντέλα αναφοράς του axis5 (χωρίς θόρυβο χαρακτηριστικών στην εκπαίδευση)
        real = copies[:-len(fb)]
        yr = np.concatenate([y[tr]] * len(real))
        meta_rt = ul.fit_meta(np.vstack([np.hstack([ul.reliability_features([p[tr] for p in c[0]], c[1][tr], c[2][tr]),
                                                    c[4][tr], temporal_block(c[0], seg, order, 2)[tr]])
                                         for c in real]), yr)
        meta_dr = ul.fit_meta(np.vstack([ul.meta_features([p[tr] for p in c[0]], c[1][tr]) for c in real]), yr)

        # --- έλεγχος
        pos = {i: j for j, i in enumerate(sub)}
        ite = np.array([pos[i] for i in te])
        subm = np.zeros(n, bool); subm[sub] = True
        for sc, M in scen.items():
            P_t = [p.copy() for p in Pfull]
            Q_t, A_t = Q.copy(), A.copy()
            drop = np.zeros((n, NB), bool)
            kind = ("noise" if sc.startswith("noise") else sc.rstrip("0123456789")
                    if sc[:5] in ("spike", "drift", "froze") else None)
            if kind:
                for bi, b in enumerate(BR):
                    rows = subm & M[:, bi] & avail[b]
                    if rows.any() and b in models:
                        Xc = corrupt(X[b], rows, kind, sd[b], test_rng[sc], seg, order, noise[b], noise_sd)
                        P_t[bi][rows] = ul.full_proba(models[b], Xc[rows])
                        A_t[rows, bi] = anomaly(Xc[rows], *stats[b])
                        if sc in ("noise16", "noise40"):
                            Q_t[rows, bi] = 0.3                     # όπως στο axis5
            else:
                drop[sub] = M[sub]
            P_, f_, Q_ = apply_mask(P_t, flags, Q_t, drop)
            PL, fL, QL, AL, aL = locf_ext(P_, f_, Q_, mask_A(A_t, drop), seg, order, max_age)
            pm = meta_new.predict_proba(feats_new(PL, fL, QL, AL, aL, sub))
            pv = rel_vote_proba([p[sub] for p in PL], AL[sub], fL[sub])
            preds = {
                NEW: pm.argmax(1)[ite],
                "late_reliability_vote": pv.argmax(1)[ite],
                "late_meta_anomaly_blend": (0.5 * pm + 0.5 * pv).argmax(1)[ite],
                "late_meta_realistic_temporal": meta_rt.predict(np.hstack([
                    ul.reliability_features([p[sub] for p in PL], fL[sub], QL[sub]), aL[sub],
                    temporal_block(PL, seg, order, 2)[sub]]))[ite],
                "late_meta_dropout_realistic": meta_dr.predict(ul.meta_features([p[sub] for p in PL], fL[sub]))[ite],
                "late_meta_dropout_robust": meta_kr.predict(ul.meta_features([p[sub] for p in PL], fL[sub]))[ite],
                "late_soft_voting": soft_vote_proba([p[te] for p in P_]).argmax(1),
            }
            for mth, pr in preds.items():
                for i, p in zip(te, pr):
                    rec.append((mth, sc, s, int(y[i]), int(p)))
        print(f"LOSO {si}/{len(np.unique(groups))}: νέος οδηγός {s}  ({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(rec, columns=["method", "scenario", "subject", "y_true", "y_pred"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--out", default="results_reliability")
    ap.add_argument("--minutes", type=int, default=4)
    ap.add_argument("--baseline", choices=["sessionA", "same"], default="sessionA")
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--max-age", type=float, default=2.0)
    ap.add_argument("--noise", type=float, default=1.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = a3.load(a.features)
    print(f"{len(df)} παράθυρα, {df.subject.nunique()} οδηγοί · LOSO + εξατομίκευση · δείκτες ανωμαλίας και συμφωνίας")
    res = run(df, a.minutes, a.baseline, a.trees, a.max_age, a.noise)
    res.to_csv(f"{a.out}/reliability_predictions.csv", index=False)

    def met(g):
        yt, yp = g.y_true.to_numpy(), g.y_pred.to_numpy()
        return pd.Series({"macro_F1": f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
                          "kappa": cohen_kappa_score(yt, yp, weights="quadratic"),
                          "false_alarm": float((yp[yt == 0] == 2).mean()) if (yt == 0).any() else np.nan})
    summ = res.groupby(["scenario", "method"]).apply(met).reset_index()
    summ.to_csv(f"{a.out}/reliability_summary.csv", index=False, encoding="utf-8-sig")
    piv = summ.pivot_table(index="method", columns="scenario", values="macro_F1")[ORDER_SC]
    piv = piv.sort_values("all", ascending=False)
    piv.to_csv(f"{a.out}/reliability_scenarios.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 240)
    print("\n=== Δείκτες αξιοπιστίας · νέος οδηγός · macro F1 ανά σενάριο ===")
    print(piv.round(3).to_string())
    print("\n" + "\n".join(f"  {m:30s} {WORK[m]}" for m in piv.index))

    ps = (res.groupby(["method", "scenario", "subject"])
          .apply(lambda g: f1_score(g.y_true, g.y_pred, average="macro", labels=[0, 1, 2], zero_division=0))
          .rename("F1").reset_index())
    ps.to_csv(f"{a.out}/reliability_per_subject.csv", index=False)
    rows = []
    for sc in ORDER_SC:
        t = ps[ps.scenario == sc].pivot_table(index="subject", columns="method", values="F1")
        for mn in NEWS:
            for m in REFS:
                d = (t[mn] - t[m]).dropna()
                try:
                    p = wilcoxon(d).pvalue if (d != 0).any() else np.nan
                except ValueError:
                    p = np.nan
                rows.append({"scenario": sc, "new": mn, "baseline": m, "F1_new": t[mn].mean(),
                             "F1_baseline": t[m].mean(), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                             "wilcoxon_p": p, "significant": bool(p < 0.05) if p == p else False})
    st = pd.DataFrame(rows)
    st.to_csv(f"{a.out}/reliability_stats.csv", index=False, encoding="utf-8-sig")
    short = {"late_meta_dropout_robust": "Krishna_ίδια_εκπ", "late_meta_dropout_realistic": "Krishna_ρεαλ",
             "late_soft_voting": "Das", "late_meta_realistic_temporal": "Βήμα2"}
    st["cell"] = st.apply(lambda r: f"{r.wins}/{r.losses} p={r.wilcoxon_p:.3f}{'*' if r.significant else ''}", axis=1)
    for mn in NEWS:
        tab = st[st.new == mn].pivot_table(index="scenario", columns="baseline", values="cell", aggfunc="first")
        tab = tab.loc[ORDER_SC, REFS].rename(columns=short)
        print(f"\n=== Wilcoxon ανά οδηγό · {mn} έναντι · νίκες/ήττες · * = p < 0,05 ===")
        print(tab.to_string())
    print(f"\nΑρχεία στον φάκελο '{a.out}': reliability_scenarios.csv, reliability_stats.csv, reliability_summary.csv, "
          "reliability_per_subject.csv, reliability_predictions.csv")


if __name__ == "__main__":
    main()
