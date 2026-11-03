#!/usr/bin/env python3
"""
UL-DD · Γρήγορος έλεγχος: μετράει η θερμοκρασία δέρματος (TEMP) την κόπωση ή απλώς τη ΔΙΑΡΚΕΙΑ της οδήγησης;

Αφορμή: στο SHAP (axis8) η θερμοκρασία είναι το 58% του κλάδου των βιομετρικών. Η θερμοκρασία όμως ανεβαίνει
και με τον χρόνο (ζεσταίνεται ο αισθητήρας/το χέρι), ενώ και η κόπωση ανεβαίνει με τον χρόνο.

1. Συσχετίσεις (Spearman, μέσα σε κάθε οδήγηση, μέσος όρος των οδηγήσεων):
     TEMP ~ χρόνος · TEMP ~ κόπωση · χρόνος ~ κόπωση
     και TEMP ΧΩΡΙΣ τον χρόνο (υπόλοιπο γραμμικής προσαρμογής στον χρόνο, ανά οδήγηση) ~ κόπωση
2. Νέος οδηγός (LOSO), Random Forest, macro F1:
     μόνο χρόνος · μόνο TEMP · μόνο TEMP χωρίς τον χρόνο · βιομετρικά όλα · βιομετρικά χωρίς TEMP
   Wilcoxon ανά οδηγό + Holm.
Ίδια κανονικοποίηση (ξεκούραστη αναφορά 4 λεπτών) και ίδια παράθυρα ελέγχου με τα axis7/axis8.

Χρειάζεται στον ίδιο φάκελο: ul_dd_baseline.py, axis3_personalization.py, features.csv
Χρήση:
    python axis9_temp_check.py --features features.csv --out results_temp
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

import ul_dd_baseline as ul
import axis3_personalization as a3

warnings.filterwarnings("ignore")


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok]); pv = p[ok][idx]; m = len(pv)
    adj = np.maximum.accumulate((m - np.arange(m)) * pv).clip(max=1.0)
    tmp = np.empty(m); tmp[idx] = adj
    out[ok] = tmp
    return out


def residual_on_time(x, t, units):
    """Για κάθε οδήγηση: x − (a + b·t), δηλαδή ό,τι μένει από το x όταν αφαιρεθεί η γραμμική τάση με τον χρόνο.
    Δεν χρησιμοποιεί ετικέτες, άρα επιτρέπεται και στον νέο οδηγό."""
    r = np.full(len(x), np.nan)
    for u in np.unique(units):
        m = (units == u) & ~np.isnan(x)
        if m.sum() >= 3:
            b, a = np.polyfit(t[m], x[m], 1)
            r[m] = x[m] - (a + b * t[m])
    return r


def model(trees, seed):
    return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("rf", RandomForestClassifier(n_estimators=trees, max_depth=8, min_samples_leaf=5,
                                                   class_weight="balanced", n_jobs=-1, random_state=seed))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--out", default="results_temp")
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 220)

    df = a3.load(a.features)
    bio = [c for c in df.columns if c.startswith("bio__")]
    temp = [c for c in bio if c.startswith("bio__TEMP__")]
    if not temp:
        print("Δεν βρέθηκαν στήλες bio__TEMP__* στο features.csv."); return
    Xb = a3.baseline_normalize(df[bio].to_numpy(float), df, 4, "sessionA")
    ti = [bio.index(c) for c in temp]
    t = df.t_s.to_numpy(float) / 60.0                                  # λεπτά από την αρχή της οδήγησης
    y = df.y.to_numpy(int)
    groups = df.subject.astype(str).to_numpy()
    units = (df.subject.astype(str) + "_" + df.session.astype(str)).to_numpy()
    tmean = Xb[:, bio.index("bio__TEMP__mean")] if "bio__TEMP__mean" in bio else np.nanmean(Xb[:, ti], 1)
    tres = residual_on_time(tmean, t, units)
    print(f"{len(df)} παράθυρα, {df.subject.nunique()} οδηγοί, {len(np.unique(units))} οδηγήσεις · "
          f"στήλες θερμοκρασίας: {', '.join(c.replace('bio__', '') for c in temp)}")

    # 1. συσχετίσεις μέσα σε κάθε οδήγηση
    rows = []
    for u in np.unique(units):
        m = units == u
        def rho(a_, b_):
            k = m & ~np.isnan(a_) & ~np.isnan(b_)
            if k.sum() < 8 or np.nanstd(a_[k]) == 0 or np.nanstd(b_[k]) == 0:
                return np.nan
            return spearmanr(a_[k], b_[k]).correlation
        rows.append(dict(unit=u, temp_time=rho(tmean, t), temp_y=rho(tmean, y.astype(float)),
                         time_y=rho(t, y.astype(float)), tempres_y=rho(tres, y.astype(float))))
    cr = pd.DataFrame(rows)
    cr.to_csv(f"{a.out}/temp_correlations_per_drive.csv", index=False, encoding="utf-8-sig")
    lab = {"temp_time": "θερμοκρασία ~ χρόνος", "temp_y": "θερμοκρασία ~ κόπωση", "time_y": "χρόνος ~ κόπωση",
           "tempres_y": "θερμοκρασία χωρίς τον χρόνο ~ κόπωση"}
    print("\n=== Συσχέτιση Spearman μέσα σε κάθε οδήγηση (−1 έως +1) · διάμεσος και πόσες οδηγήσεις είναι θετικές ===")
    for k, nm in lab.items():
        v = cr[k].dropna()
        print(f"  {nm:<38} διάμεσος {v.median():+.2f} · μέσος {v.mean():+.2f} · θετική σε {(v > 0).sum()}/{len(v)}")

    # 2. LOSO
    cal = a3.calib_mask(df, 4)
    feats = {
        "μόνο χρόνος": t[:, None],
        "μόνο θερμοκρασία": Xb[:, ti],
        "μόνο θερμοκρασία (μέση τιμή)": tmean[:, None],
        "μόνο θερμοκρασία χωρίς χρόνο": tres[:, None],
        "βιομετρικά όλα": Xb,
        "βιομετρικά χωρίς θερμοκρασία": np.delete(Xb, ti, axis=1),
    }
    rec = []
    subjects = sorted(np.unique(groups))
    for s in subjects:
        tr = np.where(groups != s)[0]
        te = np.where((groups == s) & ~cal)[0]
        if len(te) == 0:
            continue
        for name, F in feats.items():
            p = model(a.trees, a.seed).fit(F[tr], y[tr]).predict(F[te])
            rec.append(dict(subject=s, model=name,
                            F1=f1_score(y[te], p, average="macro", labels=[0, 1, 2], zero_division=0)))
        print(f"  LOSO: νέος οδηγός {s}", flush=True)
    res = pd.DataFrame(rec)
    res.to_csv(f"{a.out}/temp_loso_per_subject.csv", index=False, encoding="utf-8-sig")
    piv = res.pivot_table(index="subject", columns="model", values="F1")
    print("\n=== Νέος οδηγός (LOSO) · macro F1 ανά οδηγό, μέσος όρος ± SD ===")
    for name in feats:
        print(f"  {name:<34} {piv[name].mean():.3f} ± {piv[name].std():.3f}")

    comp = [("βιομετρικά όλα", "βιομετρικά χωρίς θερμοκρασία"),
            ("μόνο θερμοκρασία", "μόνο χρόνος"),
            ("μόνο θερμοκρασία (μέση τιμή)", "μόνο θερμοκρασία χωρίς χρόνο")]
    st = []
    for a_, b_ in comp:
        d = (piv[a_] - piv[b_]).dropna()
        p = wilcoxon(d).pvalue if (d != 0).any() and len(d) >= 5 else np.nan
        st.append(dict(σύγκριση=f"{a_} έναντι {b_}", νίκες=int((d > 0).sum()), ήττες=int((d < 0).sum()),
                       μέση_διαφορά=round(100 * d.mean(), 1), p=p))
    st = pd.DataFrame(st)
    st["H"] = holm(st.p.to_numpy())
    st["σημαντικό"] = np.where(st.H < 0.05, "*", "")
    st.to_csv(f"{a.out}/temp_stats.csv", index=False, encoding="utf-8-sig")
    show = st.assign(p=st.p.map(lambda v: f"{v:.3f}"), H=st.H.map(lambda v: f"{v:.3f}"))
    print("\n=== Wilcoxon ανά οδηγό · μέση διαφορά σε μονάδες F1 × 100 · H = p μετά Holm · * = σημαντικό ===")
    print(show.to_string(index=False))

    # 3. τι σημαίνουν
    m = piv.mean()
    print("\n=== Πώς διαβάζονται ===")
    print(f"  Αν η «μόνο θερμοκρασία» ({m['μόνο θερμοκρασία']:.3f}) είναι κοντά στον «μόνο χρόνο» "
          f"({m['μόνο χρόνος']:.3f}) και η «θερμοκρασία χωρίς χρόνο» ({m['μόνο θερμοκρασία χωρίς χρόνο']:.3f}, έναντι "
          f"{m['μόνο θερμοκρασία (μέση τιμή)']:.3f} με τον χρόνο) πέφτει κοντά στην τύχη (~0,33),\n  τότε η θερμοκρασία μετράει κυρίως τη διάρκεια της οδήγησης.")
    print(f"  Αν τα βιομετρικά χωρίς θερμοκρασία ({m['βιομετρικά χωρίς θερμοκρασία']:.3f}) πέφτουν λίγο σε σχέση με "
          f"όλα ({m['βιομετρικά όλα']:.3f}), η θερμοκρασία δεν είναι απαραίτητη.")
    print(f"\nΑρχεία στον φάκελο '{a.out}': temp_correlations_per_drive.csv, temp_loso_per_subject.csv, temp_stats.csv")


if __name__ == "__main__":
    main()
