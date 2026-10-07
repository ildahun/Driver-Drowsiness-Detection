#!/usr/bin/env python3
"""
UL-DD · Άξονας 3: Εξατομίκευση (PyTorch).

Διαβάζει το features.csv που έφτιαξε το ul_dd_baseline.py (δεν ξαναδιαβάζει τα αρχεία του UL-DD)
και συγκρίνει, σε νέους οδηγούς (leave-one-subject-out):

  global        : γενικό νευρωνικό μοντέλο, χωρίς εξατομίκευση
  norm          : ατομική κανονικοποίηση: κάθε χαρακτηριστικό μείον τη μέση τιμή του ίδιου οδηγού
                  στα πρώτα k λεπτά της οδήγησης σε εγρήγορση (baseline, χωρίς ετικέτες)
  finetune      : fine-tuning μόνο του τελευταίου επιπέδου (frozen encoder) με τα πρώτα k λεπτά
                  κάθε οδήγησης του νέου οδηγού και τις ετικέτες τους (λίγες ετικέτες)
  norm+finetune : και τα δύο

για τον κλάδο των βιομετρικών (όπως στη διπλωματική) και για τη σύντηξη των τριών κλάδων
(soft voting νευρωνικών κλάδων, με εξατομίκευση μόνο στα βιομετρικά ή σε όλους).
Τα παράθυρα βαθμονόμησης ΔΕΝ χρησιμοποιούνται στον έλεγχο, για καμία μέθοδο.

Χρήση:
    python -m pip install torch pandas numpy scikit-learn scipy matplotlib
    python axis3_personalization.py --features features.csv --out results_axis3
    (αν δεν εγκαθίσταται το torch: --backend sklearn)
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
N_CLASSES = 3
BRANCHES = ["bio", "veh", "face"]
INTERVAL_S = 240


# ----------------------------------------------------------------------------- μοντέλα
class TorchMLP:
    """Νευρωνικό δίκτυο ενός κλάδου: encoder (2 κρυφά επίπεδα) + head (γραμμικό επίπεδο εξόδου)."""

    def __init__(self, n_in, seed=SEED, hidden=(64, 32), dropout=0.2, epochs=60, lr=1e-3, batch=64):
        import torch
        import torch.nn as nn
        self.torch, self.nn = torch, nn
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(d, N_CLASSES)
        self.epochs, self.lr, self.batch = epochs, lr, batch

    def _train(self, X, y, params, epochs, lr, class_w):
        torch = self.torch
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        w = torch.tensor(class_w, dtype=torch.float32)
        loss_fn = self.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
        n = len(X)
        for _ in range(epochs):
            perm = self.rng.permutation(n)
            for i in range(0, n, self.batch):
                b = torch.as_tensor(perm[i:i + self.batch], dtype=torch.long)
                opt.zero_grad()
                loss = loss_fn(self.head(self.encoder(Xt[b])), yt[b])
                loss.backward()
                opt.step()

    def fit(self, X, y):
        cw = class_weights(y)
        self.encoder.train(); self.head.train()
        self._train(X, y, list(self.encoder.parameters()) + list(self.head.parameters()), self.epochs, self.lr, cw)
        return self

    def finetune(self, X, y, epochs=20, lr=5e-4):
        """Transfer learning: ο encoder «παγώνει», εκπαιδεύεται μόνο το head με τα δεδομένα βαθμονόμησης
        (μαζί με ένα μικρό δείγμα «υπενθύμισης» από τους άλλους οδηγούς, βλ. mix_replay)."""
        import copy
        m = copy.copy(self)                       # ρηχό αντίγραφο (τα modules torch/nn δεν αντιγράφονται)
        m.encoder = copy.deepcopy(self.encoder)   # πλήρες αντίγραφο μόνο των βαρών του δικτύου
        m.head = copy.deepcopy(self.head)
        m.rng = np.random.default_rng(self.rng.integers(1 << 31))
        for p in m.encoder.parameters():
            p.requires_grad = False
        m.encoder.eval(); m.head.train()
        m._train(X, y, list(m.head.parameters()), epochs, lr, class_weights(y))
        return m

    def predict_proba(self, X):
        torch = self.torch
        self.encoder.eval(); self.head.eval()
        with torch.no_grad():
            return torch.softmax(self.head(self.encoder(torch.tensor(X, dtype=torch.float32))), dim=1).numpy()


class SklearnMLP:
    """Εναλλακτική χωρίς PyTorch (ίδια αρχιτεκτονική). Το fine-tuning ενημερώνει όλο το δίκτυο με partial_fit."""

    def __init__(self, n_in, seed=SEED, hidden=(64, 32), epochs=60, **_):
        from sklearn.neural_network import MLPClassifier
        self.m = MLPClassifier(hidden, alpha=1e-4, max_iter=epochs, random_state=seed)

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m.fit(X, y)
        return self

    def finetune(self, X, y, epochs=20, **_):
        import copy
        m = copy.deepcopy(self)
        for _ in range(epochs):
            m.m.partial_fit(X, y)
        return m

    def predict_proba(self, X):
        p = np.zeros((len(X), N_CLASSES))
        p[:, self.m.classes_.astype(int)] = self.m.predict_proba(X)
        return p


def mix_replay(Xca, yca, Xtr, ytr, rng, ratio=3):
    """Τα πρώτα k λεπτά μιας οδήγησης έχουν συνήθως 1 μόνο τάξη. Αν το head εκπαιδευτεί μόνο σε αυτά,
    «ξεχνά» τις άλλες τάξεις. Γι' αυτό προσθέτουμε ratio x n τυχαία παράθυρα από τους άλλους οδηγούς
    (experience replay) και επαναλαμβάνουμε τα παράθυρα του νέου οδηγού ώστε να είναι το 50%."""
    n = len(Xca)
    r = rng.choice(len(Xtr), size=min(len(Xtr), ratio * n), replace=False)
    Xc, yc = np.repeat(Xca, ratio, axis=0), np.repeat(yca, ratio)
    return np.vstack([Xc, Xtr[r]]), np.concatenate([yc, ytr[r]])


def class_weights(y):
    present = np.unique(y)
    w = np.ones(N_CLASSES)
    w[present] = compute_class_weight("balanced", classes=present, y=y)
    return w


# ----------------------------------------------------------------------------- δεδομένα
def load(path):
    df = pd.read_csv(path, dtype={"subject": str, "session": str})
    num = df.select_dtypes("number").columns.drop(["interval", "window", "y"], errors="ignore")
    v = df[num].to_numpy(float)
    v[~np.isfinite(v) | (np.abs(np.nan_to_num(v, posinf=0, neginf=0)) > 1e9)] = np.nan
    df[num] = v
    n_win = df.window.max() + 1
    df = df.copy()
    df["t_s"] = df.interval * INTERVAL_S + df.window * (INTERVAL_S // n_win)   # χρόνος από την αρχή της οδήγησης
    return df


def calib_mask(df, minutes):
    """Παράθυρα βαθμονόμησης: τα πρώτα k λεπτά κάθε οδήγησης."""
    return (df.t_s < minutes * 60).to_numpy()


def baseline_normalize(X, df, minutes):
    """Ατομική κανονικοποίηση: αφαιρείται η μέση τιμή κάθε οδηγού στα πρώτα k λεπτά της οδήγησης σε εγρήγορση.
    Δεν χρησιμοποιεί ετικέτες. Αν λείπει baseline, χρησιμοποιείται η μέση τιμή όλων των παραθύρων του οδηγού."""
    Xn = X.copy()
    for s in df.subject.unique():
        rows = (df.subject == s).to_numpy()
        base = rows & (df.session == "A").to_numpy() & (df.t_s < minutes * 60).to_numpy()
        ref = X[base] if base.sum() >= 2 else X[rows]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu = np.nanmean(ref, axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        Xn[rows] = X[rows] - mu
    return Xn


def prep(Xtr, Xte_list):
    """Συμπλήρωση με διάμεσο και τυποποίηση με στατιστικά ΜΟΝΟ του train."""
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    f = lambda Z: np.nan_to_num(sc.transform(imp.transform(Z)))
    return f(Xtr), [f(Z) for Z in Xte_list]


# ----------------------------------------------------------------------------- πείραμα
def run(df, minutes_list, Model, seeds):
    y = df.y.to_numpy(int)
    groups = df.subject.to_numpy()
    feats = {b: [c for c in df.columns if c.startswith(f"{b}__")] for b in BRANCHES}
    Xraw = {b: df[feats[b]].to_numpy(float) for b in BRANCHES}
    avail = {b: ~np.isnan(Xraw[b]).all(axis=1) for b in BRANCHES}
    records = []
    subjects = sorted(np.unique(groups))

    for k_min in minutes_list:
        cal = calib_mask(df, k_min)
        Xnorm = {b: baseline_normalize(Xraw[b], df, k_min) for b in BRANCHES}
        for si, s in enumerate(subjects, 1):
            print(f"k = {k_min} λεπτά · LOSO {si}/{len(subjects)}: νέος οδηγός {s}")
            tr = groups != s
            te_all = groups == s
            te = te_all & ~cal                 # έλεγχος: ΜΟΝΟ μετά τη βαθμονόμηση
            ca = te_all & cal                  # βαθμονόμηση του νέου οδηγού
            if te.sum() == 0:
                continue
            probs = {}                          # (κλάδος, μέθοδος) -> πιθανότητες στο te
            for b in BRANCHES:
                for kind, X in (("raw", Xraw[b]), ("norm", Xnorm[b])):
                    trb = tr & avail[b]
                    Xtr, (Xte, Xca) = prep(X[trb], [X[te], X[ca]])
                    P_g, P_f = [], []
                    for seed in seeds:
                        m = Model(Xtr.shape[1], seed=seed).fit(Xtr, y[trb])
                        P_g.append(m.predict_proba(Xte))
                        ok = avail[b][ca]
                        if ok.sum() >= 2:
                            Xf, yf = mix_replay(Xca[ok], y[ca][ok], Xtr, y[trb], np.random.default_rng(seed))
                            P_f.append(m.finetune(Xf, yf).predict_proba(Xte))
                        else:
                            P_f.append(P_g[-1])
                    Pg, Pf = np.mean(P_g, 0), np.mean(P_f, 0)
                    Pg[~avail[b][te]] = np.nan; Pf[~avail[b][te]] = np.nan
                    tag = "global" if kind == "raw" else "norm"
                    probs[(b, tag)] = Pg
                    probs[(b, tag + "+finetune" if kind == "norm" else "finetune")] = Pf

            idx = np.where(te)[0]

            def log(method, P):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = np.where(np.isnan(P).all(1), 1, np.nan_to_num(P, nan=-1).argmax(1))
                for i, p in zip(idx, pred):
                    records.append((k_min, method, s, int(y[i]), int(p)))

            for tag in ["global", "norm", "finetune", "norm+finetune"]:
                ok = avail["bio"][te]
                Pb = probs[("bio", tag)].copy()
                Pb[~ok] = np.nan
                log(f"bio_{tag}", Pb)
                # σύντηξη: εξατομίκευση ΜΟΝΟ στον κλάδο βιομετρικών (όπως στη διπλωματική)
                stack = np.stack([probs[("bio", tag)], probs[("veh", "global")], probs[("face", "global")]])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    log(f"fusion_bioOnly_{tag}", np.nanmean(stack, axis=0))
                    # σύντηξη: εξατομίκευση σε ΟΛΟΥΣ τους κλάδους
                    stack = np.stack([probs[(b, tag)] for b in BRANCHES])
                    log(f"fusion_all_{tag}", np.nanmean(stack, axis=0))
    return pd.DataFrame(records, columns=["k_min", "method", "subject", "y_true", "y_pred"])


def metrics(yt, yp):
    alert = yt == 0
    return {"macro_F1": f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
            "precision_macro": precision_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
            "recall_macro": recall_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
            "recall_High": recall_score(yt, yp, labels=[2], average="macro", zero_division=0),
            "kappa_quadratic": cohen_kappa_score(yt, yp, weights="quadratic"),
            "balanced_acc": balanced_accuracy_score(yt, yp), "accuracy": accuracy_score(yt, yp),
            "false_alarm_rate": float(np.mean(yp[alert] == 2)) if alert.any() else np.nan}


def summarize(res):
    from scipy.stats import wilcoxon
    rows, per = [], []
    for (k, m), g in res.groupby(["k_min", "method"]):
        r = {"k_min": k, "method": m, "n_windows": len(g), **metrics(g.y_true.to_numpy(), g.y_pred.to_numpy())}
        f1s = {}
        for s, gs in g.groupby("subject"):
            f1s[s] = f1_score(gs.y_true, gs.y_pred, labels=np.unique(gs.y_true), average="macro", zero_division=0)
            per.append({"k_min": k, "method": m, "subject": s, "macro_F1": f1s[s]})
        r["subj_F1_mean"], r["subj_F1_std"] = np.mean(list(f1s.values())), np.std(list(f1s.values()))
        rows.append(r)
    summ = pd.DataFrame(rows)
    per = pd.DataFrame(per)
    # κάθε εξατομικευμένη μέθοδος έναντι της αντίστοιχης χωρίς εξατομίκευση (ανά οδηγό)
    tests = []
    for k in per.k_min.unique():
        pk = per[per.k_min == k].pivot(index="subject", columns="method", values="macro_F1")
        for prefix in ["bio_", "fusion_bioOnly_", "fusion_all_"]:
            base = prefix + "global"
            for tag in ["norm", "finetune", "norm+finetune"]:
                m = prefix + tag
                if base not in pk or m not in pk:
                    continue
                d = (pk[m] - pk[base]).dropna()
                try:
                    p = wilcoxon(d).pvalue if (d != 0).any() else np.nan
                except ValueError:
                    p = np.nan
                tests.append({"k_min": k, "method": m, "vs": base, "mean_gain_F1": d.mean(),
                              "improved_subjects": int((d > 0).sum()), "worse_subjects": int((d < 0).sum()),
                              "wilcoxon_p": p, "significant": bool(p < 0.05) if not np.isnan(p) else False})
    return summ, per, pd.DataFrame(tests)


def plot(summ, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    styles = {"global": ("#8FA3BF", "o"), "norm": ("#2F6690", "s"),
              "finetune": ("#E8A33D", "^"), "norm+finetune": ("#B4531A", "D")}
    for ax, prefix, title in [(axes[0], "bio_", "Κλάδος βιομετρικών"),
                              (axes[1], "fusion_bioOnly_", "Σύντηξη (εξατομίκευση στα βιομετρικά)")]:
        for tag, (c, mk) in styles.items():
            s = summ[summ.method == prefix + tag].sort_values("k_min")
            ax.plot(s.k_min, s.macro_F1, marker=mk, color=c, label=tag)
        ax.axhline(1 / 3, ls="--", color="#777777", lw=1)
        ax.set_title(title); ax.set_xlabel("Λεπτά βαθμονόμησης (k)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Macro F1 σε νέο οδηγό (LOSO)")
    axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--out", default="results_axis3")
    ap.add_argument("--minutes", default="2,4,8", help="λεπτά βαθμονόμησης, π.χ. 2,4,8")
    ap.add_argument("--seeds", type=int, default=3, help="επαναλήψεις με διαφορετικό seed (μέσος όρος)")
    ap.add_argument("--backend", choices=["torch", "sklearn"], default="torch")
    a = ap.parse_args()
    if a.backend == "torch":
        try:
            import torch  # noqa: F401
        except ImportError:
            raise SystemExit("Δεν βρέθηκε το PyTorch. Εγκατάσταση: python -m pip install torch  "
                             "(ή τρέξτε με --backend sklearn)")
    Model = TorchMLP if a.backend == "torch" else SklearnMLP
    df = load(a.features)
    print(f"{len(df)} παράθυρα, {df.subject.nunique()} οδηγοί, backend = {a.backend}")
    mins = [int(x) for x in a.minutes.split(",")]
    res = run(df, mins, Model, seeds=list(range(SEED, SEED + a.seeds)))
    summ, per, tests = summarize(res)
    Path(a.out).mkdir(exist_ok=True)
    summ.to_csv(f"{a.out}/axis3_summary.csv", index=False, encoding="utf-8-sig")
    per.to_csv(f"{a.out}/axis3_per_subject.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(f"{a.out}/axis3_stats.csv", index=False, encoding="utf-8-sig")
    plot(summ, f"{a.out}/axis3_calibration_curve.png")
    pd.set_option("display.width", 220)
    cols = ["k_min", "method", "macro_F1", "precision_macro", "recall_macro", "recall_High",
            "kappa_quadratic", "accuracy", "false_alarm_rate", "subj_F1_mean"]
    print("\n=== Άξονας 3 · νέος οδηγός (LOSO), έλεγχος μόνο μετά τη βαθμονόμηση ===")
    print(summ[cols].round(3).to_string(index=False))
    print("\n=== Κέρδος εξατομίκευσης ανά οδηγό (έναντι της ίδιας μεθόδου χωρίς εξατομίκευση) ===")
    print(tests.round(4).to_string(index=False))
    print(f"""
Μέθοδοι:  bio_*            = μόνο ο κλάδος βιομετρικών
          fusion_bioOnly_* = σύντηξη 3 κλάδων (soft voting), εξατομίκευση μόνο στα βιομετρικά
          fusion_all_*     = σύντηξη 3 κλάδων, εξατομίκευση σε όλους
  global = χωρίς εξατομίκευση · norm = ατομική κανονικοποίηση (χωρίς ετικέτες)
  finetune = fine-tuning του τελευταίου επιπέδου με τα πρώτα k λεπτά κάθε οδήγησης (λίγες ετικέτες)
false_alarm_rate = ποσοστό παραθύρων σε εγρήγορση (Alert) που χαρακτηρίστηκαν High.
Αρχεία στον φάκελο '{a.out}': axis3_summary.csv, axis3_stats.csv, axis3_per_subject.csv, axis3_calibration_curve.png""")


if __name__ == "__main__":
    main()
