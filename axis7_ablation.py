#!/usr/bin/env python3
"""
UL-DD · Επόμενα βήματα 1–3: ABLATION + ΕΠΙΒΕΒΑΙΩΣΗ ΜΕ 5 ΣΠΟΡΟΥΣ + ΠΡΑΓΜΑΤΙΚΕΣ ΒΛΑΒΕΣ (νέος οδηγός, LOSO).

1. Ablation: η τελική μέθοδος του βήματος 4 (συνδυασμός meta-classifier με δείκτες ανωμαλίας/συμφωνίας
   + ψηφοφορία αξιοπιστίας) και η ίδια με ΕΝΑ κομμάτι αφαιρεμένο κάθε φορά:
     FULL              η τελική μέθοδος
     -ψηφοφορία        μόνο ο meta-classifier
     -meta             μόνο η ψηφοφορία αξιοπιστίας
     -ανωμαλία         χωρίς δείκτη ανωμαλίας (και απλή ψηφοφορία αντί για ψηφοφορία αξιοπιστίας)
     -συμφωνία         χωρίς δείκτη συμφωνίας
     -χρόνος           χωρίς χρονικό πλαίσιο
     -ρεαλιστικές      εκπαίδευση μόνο με τυχαία κενά (dropout) και θόρυβο χαρακτηριστικών
     -εξατομίκευση     χωρίς κανονικοποίηση με την ξεκούραστη οδήγηση (μόνο στους πρώτους --raw-seeds σπόρους)
   Σύγκριση και με: Krishna et al. με ΙΔΙΑ εκπαίδευση, Das et al.
2. Επιβεβαίωση: όλο το πείραμα επαναλαμβάνεται με 5 διαφορετικούς σπόρους τυχαιότητας (μοντέλα, βλάβες
   εκπαίδευσης, σενάρια ελέγχου). Ο Wilcoxon γίνεται στο F1 κάθε οδηγού (μέσος όρος των σπόρων) και
   διορθώνεται με Holm για τους πολλαπλούς ελέγχους.
3. Πραγματικές βλάβες: τα παράθυρα του UL-DD όπου κάποιος αισθητήρας ΠΡΑΓΜΑΤΙΚΑ λείπει ή έχει ποιότητα < 0,9
   (σενάριο «real_bad», χωρίς καμία τεχνητή βλάβη).

Αποθηκεύει μετά από κάθε σπόρο· αν διακοπεί, την επόμενη φορά συνεχίζει από τον επόμενο σπόρο.

Χρειάζεται στον ίδιο φάκελο: ul_dd_baseline.py, axis3_personalization.py, axis4_temporal.py, axis5_realistic.py,
axis6_reliability.py, features.csv
Χρήση:
    python axis7_ablation.py --features features.csv --out results_ablation
    (προαιρετικά: --seeds 5  --raw-seeds 1  --trees 150)
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

import ul_dd_baseline as ul
import axis3_personalization as a3
from axis4_temporal import segments, temporal_block, apply_mask
from axis5_realistic import block_mask, degrade
from axis6_reliability import (robust_stats, anomaly, agreement, locf_ext, mask_A, rel_vote_proba,
                               soft_vote_proba, corrupt, oof_multi)

warnings.filterwarnings("ignore")
BR, NC = ul.BRANCHES, ul.N_CLASSES
NB = len(BR)
FULL = "FULL"
ABL = ["-ψηφοφορία", "-meta", "-ανωμαλία", "-συμφωνία", "-χρόνος", "-ρεαλιστικές", "-εξατομίκευση"]
REFS = ["Krishna_ίδια_εκπ", "Das"]
ORDER_M = [FULL] + ABL + REFS
SCEN = ["all", "random60", "outage40", "noise16", "noise40_blind", "spike40", "frozen40"]
ORDER_SC = SCEN + ["real_bad"]
BAD_Q = 0.9


# ----------------------------------------------------------------------------- ένα fold
def build(Xd, avail, sdd, y, groups, tr, sub, n, trees, rng, df, seg, order, Qall):
    """Κλάδοι (OOF + μοντέλο για τον νέο οδηγό), δείκτες ανωμαλίας, και αντίγραφα εκπαίδευσης."""
    trm = np.zeros(n, bool); trm[tr] = True
    fb = [block_mask(df, seg, order, 16, rng), block_mask(df, seg, order, 1000, rng)]
    stats = {b: robust_stats(Xd[b][tr]) for b in BR}
    Pfull = [np.full((n, NC), np.nan) for _ in BR]
    Pnoisy = [[np.full((n, NC), np.nan) for _ in BR] for _ in fb]
    A = np.zeros((n, NB, 2)); An = [np.zeros((n, NB, 2)) for _ in fb]
    models = {}
    for bi, b in enumerate(BR):
        Xn = [corrupt(Xd[b], trm & m[:, bi] & avail[b], "train_noise", sdd[b], rng, seg, order) for m in fb]
        outs = oof_multi([Xd[b][tr]] + [x[tr] for x in Xn], y[tr], groups[tr], avail[b][tr], trees)
        Pfull[bi][tr] = outs[0]
        for k in range(len(fb)):
            Pnoisy[k][bi][tr] = outs[k + 1]
            An[k][:, bi] = anomaly(Xn[k], *stats[b])
        A[:, bi] = anomaly(Xd[b], *stats[b])
        ok, okt = avail[b][sub], avail[b][tr]
        if okt.sum() >= 10 and ok.any():
            models[b] = ul.branch_model(trees).fit(Xd[b][tr][okt], y[tr][okt])
            Pfull[bi][sub[ok]] = ul.full_proba(models[b], Xd[b][sub][ok])
    flags = np.stack([~np.isnan(P[:, 0]) for P in Pfull], 1).astype(int)
    Q = Qall * flags
    A = A * flags[:, :, None]
    z = np.zeros((n, NB))
    real = [(Pfull, flags, Q, A, z)]                                     # ρεαλιστικές βλάβες (όπως axis5/6)
    for d in [rng.random((n, NB)) < 0.3] + [block_mask(df, seg, order, m, rng) for m in (4, 16, 1000)]:
        P_, f_, Q_ = apply_mask(Pfull, flags, Q, d)
        real.append(locf_ext(P_, f_, Q_, mask_A(A, d), seg, order, 2.0))
    bm = block_mask(df, seg, order, 16, rng)
    P2, Q2 = degrade(Pfull, Q, {b: bm[:, b] for b in range(NB)}, rng)
    real.append((P2, flags, Q2, A, z))
    noisec = []
    for k in range(len(fb)):
        Pk = [np.where(np.isnan(Pnoisy[k][b]), Pfull[b], Pnoisy[k][b]) for b in range(NB)]
        noisec.append((Pk, flags, Q, An[k] * flags[:, :, None], z))
    base = [(Pfull, flags, Q, A, z)]                                     # μόνο τυχαία κενά (dropout 20%)
    for _ in range(2):
        d = rng.random((n, NB)) < 0.2
        P_, f_, Q_ = apply_mask(Pfull, flags, Q, d)
        base.append((P_, f_, Q_, mask_A(A, d), z))
    return dict(Pfull=Pfull, flags=flags, Q=Q, A=A, models=models, stats=stats,
                real=real, noisec=noisec, base=base)


def feats(P, f, q, Aa, age, rows, seg, order, anom=True, agree=True, temp=True):
    Pr = [p[rows] for p in P]
    parts = [ul.reliability_features(Pr, f[rows], q[rows]), age[rows]]
    if anom:
        parts.append(Aa[rows].reshape(len(rows), -1))
    if agree:
        parts.append(agreement(Pr, f[rows]))
    if temp:
        parts.append(temporal_block(P, seg, order, 2)[rows])
    return np.hstack(parts)


def fit_metas(B, y, tr, seg, order, which):
    copies = B["real"] + B["noisec"]
    yc = np.concatenate([y[tr]] * len(copies))
    cfg = {FULL: (copies, dict()), "-ανωμαλία": (copies, dict(anom=False)),
           "-συμφωνία": (copies, dict(agree=False)), "-χρόνος": (copies, dict(temp=False)),
           "-ρεαλιστικές": (B["base"] + B["noisec"], dict())}
    out = {}
    for name in which:
        cp, kw = cfg[name]
        yy = np.concatenate([y[tr]] * len(cp))
        out[name] = (ul.fit_meta(np.vstack([feats(*c, tr, seg, order, **kw) for c in cp]), yy), kw)
    if "krishna" in which or True:
        out["krishna"] = ul.fit_meta(np.vstack([ul.meta_features([p[tr] for p in c[0]], c[1][tr]) for c in copies]), yc)
    return out


def test_state(B, Xd, sdd, avail, sub, n, sc, M, trng, seg, order, noise, noise_sd):
    """Εφαρμόζει το σενάριο στον νέο οδηγό και επιστρέφει τις εισόδους μετά το LOCF."""
    P_t = [p.copy() for p in B["Pfull"]]
    Q_t, A_t = B["Q"].copy(), B["A"].copy()
    drop = np.zeros((n, NB), bool)
    subm = np.zeros(n, bool); subm[sub] = True
    kind = ("noise" if sc.startswith("noise") else sc.rstrip("0123456789")
            if sc[:5] in ("spike", "drift", "froze") else None)
    if kind:
        for bi, b in enumerate(BR):
            rows = subm & M[:, bi] & avail[b]
            if rows.any() and b in B["models"]:
                Xc = corrupt(Xd[b], rows, kind, sdd[b], trng, seg, order, noise[b], noise_sd)
                P_t[bi][rows] = ul.full_proba(B["models"][b], Xc[rows])
                A_t[rows, bi] = anomaly(Xc[rows], *B["stats"][b])
    elif sc != "all":
        drop[sub] = M[sub]
    P_, f_, Q_ = apply_mask(P_t, B["flags"], Q_t, drop)
    PL, fL, QL, AL, aL = locf_ext(P_, f_, Q_, mask_A(A_t, drop), seg, order, 2.0)
    return P_, (PL, fL, QL, AL, aL)


def run_seed(df, X, Xraw, avail, Qall, seed, trees, with_raw, noise_sd):
    ul.SEED = seed
    y = df.y.to_numpy(int)
    groups = df.subject.to_numpy()
    seg, order = segments(df)
    n = len(df)
    sd = {b: np.nanstd(X[b], axis=0) for b in BR}
    sdr = {b: np.nanstd(Xraw[b], axis=0) for b in BR}
    cal = a3.calib_mask(df, 4)
    rng = np.random.default_rng(seed)
    srng = np.random.default_rng(seed + 7)
    masks = {"all": np.zeros((n, NB), bool), "random60": srng.random((n, NB)) < 0.6}
    for sc in SCEN:
        if sc not in masks:
            masks[sc] = block_mask(df, seg, order, 16 if sc == "noise16" else 1000, srng)
    noise = {b: srng.normal(0, 1, X[b].shape) for b in BR}
    noise_r = {b: srng.normal(0, 1, Xraw[b].shape) for b in BR}
    bad = (np.stack([~avail[b] for b in BR], 1).any(1)) | (Qall < BAD_Q).any(1)
    rec = []
    for si, s in enumerate(sorted(np.unique(groups)), 1):
        t0 = time.time()
        tr = np.where(groups != s)[0]
        sub = np.where(groups == s)[0]
        te = np.where((groups == s) & ~cal)[0]
        if len(te) == 0:
            continue
        pos = {i: j for j, i in enumerate(sub)}
        ite = np.array([pos[i] for i in te])
        B = build(X, avail, sd, y, groups, tr, sub, n, trees, rng, df, seg, order, Qall)
        mt = fit_metas(B, y, tr, seg, order, [FULL, "-ανωμαλία", "-συμφωνία", "-χρόνος", "-ρεαλιστικές"])
        if with_raw:
            Br = build(Xraw, avail, sdr, y, groups, tr, sub, n, trees, rng, df, seg, order, Qall)
            mr = fit_metas(Br, y, tr, seg, order, [FULL])
        for sc in SCEN:
            trng = np.random.default_rng(seed + 1000 + SCEN.index(sc))
            P_, st = test_state(B, X, sd, avail, sub, n, sc, masks[sc], trng, seg, order, noise, noise_sd)
            PL, fL, QL, AL, aL = st
            mp = lambda name: mt[name][0].predict_proba(feats(PL, fL, QL, AL, aL, sub, seg, order, **mt[name][1]))
            pv = rel_vote_proba([p[sub] for p in PL], AL[sub], fL[sub])
            psv = soft_vote_proba([p[sub] for p in PL])
            pf = mp(FULL)
            preds = {
                FULL: 0.5 * pf + 0.5 * pv,
                "-ψηφοφορία": pf,
                "-meta": pv,
                "-ανωμαλία": 0.5 * mp("-ανωμαλία") + 0.5 * psv,
                "-συμφωνία": 0.5 * mp("-συμφωνία") + 0.5 * pv,
                "-χρόνος": 0.5 * mp("-χρόνος") + 0.5 * pv,
                "-ρεαλιστικές": 0.5 * mp("-ρεαλιστικές") + 0.5 * pv,
                "Krishna_ίδια_εκπ": mt["krishna"].predict_proba(ul.meta_features([p[sub] for p in PL], fL[sub])),
                "Das": soft_vote_proba([p[sub] for p in P_]),
            }
            if with_raw:
                trng = np.random.default_rng(seed + 1000 + SCEN.index(sc))
                _, str_ = test_state(Br, Xraw, sdr, avail, sub, n, sc, masks[sc], trng, seg, order, noise_r, noise_sd)
                PLr, fLr, QLr, ALr, aLr = str_
                preds["-εξατομίκευση"] = (0.5 * mr[FULL][0].predict_proba(feats(PLr, fLr, QLr, ALr, aLr, sub, seg, order))
                                          + 0.5 * rel_vote_proba([p[sub] for p in PLr], ALr[sub], fLr[sub]))
            for mth, pr in preds.items():
                lab = pr.argmax(1)[ite]
                for i, p in zip(te, lab):
                    rec.append((seed, mth, sc, s, int(y[i]), int(p), bool(bad[i])))
        print(f"  σπόρος {seed} · LOSO {si}/{len(np.unique(groups))}: νέος οδηγός {s}  ({time.time() - t0:.0f} s)",
              flush=True)
    return pd.DataFrame(rec, columns=["seed", "method", "scenario", "subject", "y_true", "y_pred", "bad"])


# ----------------------------------------------------------------------------- στατιστικά
def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok]); pv = p[ok][idx]; m = len(pv)
    adj = np.maximum.accumulate((m - np.arange(m)) * pv).clip(max=1.0)
    tmp = np.empty(m); tmp[idx] = adj
    out[ok] = tmp
    return out


def f1m(g):
    return f1_score(g.y_true, g.y_pred, average="macro", labels=[0, 1, 2], zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--out", default="results_ablation")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--raw-seeds", type=int, default=1, help="σε πόσους σπόρους τρέχει και το -εξατομίκευση")
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--noise", type=float, default=1.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = a3.load(a.features)
    feat = {b: [c for c in df.columns if c.startswith(f"{b}__")] for b in BR}
    X = {b: a3.baseline_normalize(df[feat[b]].to_numpy(float), df, 4, "sessionA") for b in BR}
    Xraw = {b: df[feat[b]].to_numpy(float) for b in BR}
    avail = {b: ~np.isnan(df[feat[b]].to_numpy(float)).all(axis=1) for b in BR}
    qcols = [f"q__{b}" for b in BR]
    Qall = (df[qcols].fillna(0.0).to_numpy(float) if all(c in df for c in qcols)
            else np.stack([avail[b] for b in BR], 1).astype(float))
    seeds = [42 + k for k in range(a.seeds)]
    print(f"{len(df)} παράθυρα, {df.subject.nunique()} οδηγοί · ablation · σπόροι {seeds} · "
          f"-εξατομίκευση στους πρώτους {a.raw_seeds}")
    parts = []
    for k, sd in enumerate(seeds):
        f = f"{a.out}/ablation_predictions_seed{sd}.csv"
        if os.path.exists(f):
            print(f"σπόρος {sd}: υπάρχει ήδη, τον φορτώνω"); parts.append(pd.read_csv(f)); continue
        t0 = time.time()
        r = run_seed(df, X, Xraw, avail, Qall, sd, a.trees, k < a.raw_seeds, a.noise)
        r.to_csv(f, index=False); parts.append(r)
        print(f"σπόρος {sd}: έτοιμος ({(time.time() - t0) / 60:.1f} λεπτά) · αποθηκεύτηκε", flush=True)
    res = pd.concat(parts)
    rb = res[(res.scenario == "all") & res.bad].copy(); rb["scenario"] = "real_bad"
    res = pd.concat([res, rb])
    pd.set_option("display.width", 260)
    global ORDER_SC
    ORDER_SC = [x for x in ORDER_SC if x in set(res.scenario)]
    if "real_bad" not in ORDER_SC:
        print(f"  (δεν βρέθηκαν παράθυρα με πραγματικά λείπων αισθητήρα ή ποιότητα < {BAD_Q})")

    # 1. μέσος όρος ± SD μεταξύ σπόρων
    per_seed = res.groupby(["seed", "method", "scenario"]).apply(f1m).rename("F1").reset_index()
    agg = per_seed.groupby(["method", "scenario"]).F1.agg(["mean", "std"]).reset_index()
    agg.to_csv(f"{a.out}/ablation_summary.csv", index=False, encoding="utf-8-sig")
    meth = [m for m in ORDER_M if m in agg.method.unique()]
    cell = agg.assign(c=agg["mean"].map("{:.3f}".format) + "±" + agg["std"].fillna(0).map("{:.3f}".format))
    t1 = cell.pivot_table(index="method", columns="scenario", values="c", aggfunc="first").loc[meth, ORDER_SC]
    print(f"\n=== Ablation · macro F1 · μέσος όρος {len(seeds)} σπόρων ± SD ===")
    print(t1.to_string())
    nbad = int(rb[(rb.method == FULL)].groupby("seed").size().mean()) if len(rb) else 0
    if nbad: print(f"  real_bad = {nbad} παράθυρα με πραγματικά λείπων αισθητήρα ή ποιότητα < {BAD_Q} (χωρίς τεχνητή βλάβη)")

    # 2. διαφορά από την πλήρη μέθοδο
    mean = agg.pivot_table(index="method", columns="scenario", values="mean").loc[meth, ORDER_SC]
    delta = ((mean - mean.loc[FULL]) * 100).round(1)
    delta.to_csv(f"{a.out}/ablation_delta.csv", encoding="utf-8-sig")
    print("\n=== Διαφορά από την πλήρη μέθοδο · μονάδες F1 × 100 (αρνητικό = το κομμάτι βοηθάει) ===")
    print(delta.drop(index=FULL).to_string())

    # 3. Wilcoxon στο F1 ανά οδηγό (μέσος όρος σπόρων) + Holm
    ps = (res.groupby(["seed", "method", "scenario", "subject"]).apply(f1m).rename("F1").reset_index()
          .groupby(["method", "scenario", "subject"]).F1.mean().reset_index())
    ps.to_csv(f"{a.out}/ablation_per_subject.csv", index=False)
    rows = []
    for sc in ORDER_SC:
        t = ps[ps.scenario == sc].pivot_table(index="subject", columns="method", values="F1")
        for m in meth:
            if m == FULL or m not in t:
                continue
            d = (t[FULL] - t[m]).dropna()
            try:
                p = wilcoxon(d).pvalue if (d != 0).any() and len(d) >= 5 else np.nan
            except ValueError:
                p = np.nan
            rows.append(dict(scenario=sc, method=m, wins=int((d > 0).sum()), losses=int((d < 0).sum()), p=p))
    st = pd.DataFrame(rows)
    st["p_holm"] = np.nan
    for fam in (ABL, REFS):                      # δύο οικογένειες ελέγχων: ablation και σύγκριση με άλλες εργασίες
        m_ = st.method.isin(fam)
        st.loc[m_, "p_holm"] = holm(st.loc[m_, "p"].to_numpy())
    st["sig"] = st.p_holm < 0.05
    st.to_csv(f"{a.out}/ablation_stats.csv", index=False, encoding="utf-8-sig")
    st["c"] = st.apply(lambda r: f"{r.wins}/{r.losses} H={r.p_holm:.3f}{'*' if r.sig else ''}", axis=1)
    t3 = st.pivot_table(index="scenario", columns="method", values="c", aggfunc="first")
    t3 = t3.loc[[s for s in ORDER_SC if s in t3.index]]
    print("\n=== Wilcoxon ανά οδηγό · FULL έναντι άλλων εργασιών · νίκες/ήττες · H = p μετά Holm · * = σημαντικό ===")
    print(t3[[m for m in REFS if m in t3.columns]].to_string())
    print("\n=== Wilcoxon ανά οδηγό · FULL έναντι ablation · νίκες/ήττες · H = p μετά Holm · * = σημαντικό ===")
    print(t3[[m for m in ABL if m in t3.columns]].to_string())

    # 4. σταθερότητα μεταξύ σπόρων
    w = per_seed.pivot_table(index=["seed", "scenario"], columns="method", values="F1")
    cons = ((w[FULL] > w["Krishna_ίδια_εκπ"]).groupby("scenario").sum().astype(int).astype(str)
            + f"/{len(seeds)}").reindex(ORDER_SC)
    print(f"\n=== Σταθερότητα: σε πόσους σπόρους η FULL είναι πάνω από τον Krishna με ίδια εκπαίδευση ===")
    print(cons.to_string())
    print(f"\nΑρχεία στον φάκελο '{a.out}': ablation_summary.csv, ablation_delta.csv, ablation_stats.csv, "
          "ablation_per_subject.csv, ablation_predictions_seed*.csv")


if __name__ == "__main__":
    main()
