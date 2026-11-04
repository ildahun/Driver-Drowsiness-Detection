#!/usr/bin/env python3
"""
UL-DD · Τα δύο τελευταία απαραίτητα βήματα (νέος οδηγός, LOSO).

Α. ΠΡΑΓΜΑΤΙΚΟ ΚΑΚΟ ΣΗΜΑ, ΑΝΑΛΥΤΙΚΑ (χωρίς νέα εκπαίδευση)
   Παίρνει τις προβλέψεις που έχει ήδη αποθηκεύσει το axis7 (results_ablation/ablation_predictions_seed*.csv,
   σενάριο «all», δηλαδή χωρίς καμία τεχνητή βλάβη) και χωρίζει τα κακά παράθυρα σε:
     λείπει             κάποιος αισθητήρας λείπει εντελώς (όλα τα χαρακτηριστικά του είναι κενά)
     χαμηλή_ποιότητα    όλοι οι αισθητήρες υπάρχουν, αλλά κάποιος έχει ποιότητα < 0,9
   και επιπλέον ανά αισθητήρα. Ελέγχει την υπόθεση της προηγούμενης παρουσίασης: ο δείκτης ανωμαλίας
   χάνει στο πραγματικό κακό σήμα επειδή εκεί το σήμα κυρίως ΛΕΙΠΕΙ (και όχι επειδή έχει θόρυβο).

Β. SHAP · ΠΟΙΟΙ ΑΙΣΘΗΤΗΡΕΣ ΚΑΙ ΠΟΙΑ ΣΗΜΑΤΑ ΚΡΙΝΟΥΝ ΤΗΝ ΑΠΟΦΑΣΗ
   Τιμές Shapley ανά ΟΜΑΔΑ χαρακτηριστικών, υπολογισμένες ΑΚΡΙΒΩΣ (όλοι οι συνδυασμοί ομάδων), με δείγμα
   αναφοράς από τα δεδομένα εκπαίδευσης (interventional SHAP, ο ίδιος ορισμός με τη βιβλιοθήκη shap).
   Δεν χρειάζεται η βιβλιοθήκη shap. Ελέγχεται και η αθροιστικότητα: Σ φ + βάση = πρόβλεψη του μοντέλου.
     Β1. Κριτής (meta-classifier της τελικής μεθόδου, όπως στο axis7):
           ανά αισθητήρα (βιομετρικά / όχημα / πρόσωπο) και ανά είδος εισόδου
           (πιθανότητες, διαθεσιμότητα, ποιότητα, βεβαιότητα, ηλικία LOCF, ανωμαλία, συμφωνία, χρόνος),
           σε δύο σενάρια: all (όλοι οι αισθητήρες) και random60 (60% τυχαία κενά).
     Β2. Μέσα σε κάθε κλάδο (Random Forest): ανά σήμα (π.χ. HR, EDA, PERCLOS, ταχύτητα, πίεση λαβής).
   Η εξήγηση γίνεται στον ΝΕΟ οδηγό: για κάθε οδηγό τα μοντέλα εκπαιδεύονται στους άλλους 18 (LOSO).
   Σημείωση: η τελική πρόβλεψη είναι 0,5 · κριτής + 0,5 · ψηφοφορία αξιοπιστίας· το SHAP εξηγεί τον κριτή.

Χρειάζεται στον ίδιο φάκελο: ul_dd_baseline.py, axis3_personalization.py, axis4_temporal.py, axis5_realistic.py,
axis6_reliability.py, axis7_ablation.py, features.csv και (για το Α) τον φάκελο results_ablation του axis7.
Χρήση:
    python axis8_shap.py --features features.csv --ablation results_ablation --out results_shap
    (προαιρετικά: --no-shap  μόνο το Α · --no-realbad  μόνο το Β · --n-explain 40 --n-bg 30 --trees 150)
"""
import argparse
import glob
import os
import time
import warnings
from math import factorial

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

import ul_dd_baseline as ul
import axis3_personalization as a3
import axis7_ablation as a7
from axis4_temporal import segments
# ξαναχρησιμοποιώ τα ίδια κομμάτια με το axis7, ώστε ο κριτής να είναι ακριβώς ο ίδιος
from axis7_ablation import build, feats, test_state, holm, FULL, ORDER_M, BAD_Q

warnings.filterwarnings("ignore")
BR, NC = ul.BRANCHES, ul.N_CLASSES
NB = len(BR)
BR_GR = {"bio": "βιομετρικά", "veh": "όχημα", "face": "πρόσωπο"}   # για τις εκτυπώσεις
CLS = ["χαμηλή", "μέτρια", "υψηλή"]
MAX_GROUPS = 10                      # 2^10 = 1024 συνδυασμοί· πάνω από αυτό οι μικρότερες ομάδες ενώνονται


def f1m(yt, yp):
    return f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0)


# =============================================================================
# Α. Πραγματικό κακό σήμα
# =============================================================================
def load_axis7(df, abl_dir):
    """Διαβάζει τις προβλέψεις του axis7 (σενάριο all) και βρίσκει σε ποιο παράθυρο αντιστοιχεί κάθε γραμμή.
    Το axis7 γράφει, για κάθε οδηγό (με αλφαβητική σειρά) και μέθοδο, τα παράθυρα ελέγχου με τη σειρά του αρχείου."""
    files = sorted(glob.glob(os.path.join(abl_dir, "ablation_predictions_seed*.csv")))
    if not files:
        return None
    groups = df.subject.astype(str).to_numpy()
    cal = a3.calib_mask(df, 4)
    # ίδια παράθυρα ελέγχου με το axis7 (χωρίς τα 4 λεπτά αναφοράς)
    te = {s: np.where((groups == s) & ~cal)[0] for s in sorted(np.unique(groups))}
    yv = df.y.to_numpy(int)
    out = []
    for f in files:
        r = pd.read_csv(f, dtype={"subject": str})
        r = r[r.scenario == "all"]
        for (sd, m, s), g in r.groupby(["seed", "method", "subject"], sort=False):
            ix = te.get(s)
            # έλεγχος: ίδιο πλήθος γραμμών και ίδιες ετικέτες, αλλιώς το features.csv δεν είναι το ίδιο
            if ix is None or len(ix) != len(g) or not (yv[ix] == g.y_true.to_numpy()).all():
                print(f"  [!] {os.path.basename(f)}: οι προβλέψεις του οδηγού {s} δεν ταιριάζουν με το features.csv "
                      "(άλλο features.csv από αυτό του axis7;). Παραλείπεται το μέρος Α.")
                return None
            # i = ποιο παράθυρο του df είναι κάθε γραμμή
            out.append(pd.DataFrame({"seed": sd, "method": m, "subject": s, "i": ix,
                                     "y_true": g.y_true.to_numpy(), "y_pred": g.y_pred.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def realbad_analysis(df, avail, Qall, abl_dir, out):
    print("\n" + "=" * 100 + "\nΑ. ΠΡΑΓΜΑΤΙΚΟ ΚΑΚΟ ΣΗΜΑ · ΛΕΙΠΕΙ Ο ΑΙΣΘΗΤΗΡΑΣ ή ΧΑΜΗΛΗ ΠΟΙΟΤΗΤΑ\n" + "=" * 100)
    res = load_axis7(df, abl_dir)
    if res is None:
        print(f"  Δεν βρέθηκαν (σωστές) προβλέψεις του axis7 στον φάκελο '{abl_dir}'. "
              "Τρέξτε πρώτα: python axis7_ablation.py --features features.csv --out results_ablation")
        return
    miss = np.stack([~avail[b] for b in BR], 1)                     # (n, 3) ο αισθητήρας λείπει εντελώς
    lowq = (Qall < BAD_Q) & ~miss                                   # υπάρχει, αλλά με ποιότητα < 0,9
    # αν λείπει κάτι -> "λείπει" (έχει προτεραιότητα), αλλιώς αν κάτι είναι χαμηλό -> "χαμηλή_ποιότητα"
    kind = np.where(miss.any(1), "λείπει", np.where(lowq.any(1), "χαμηλή_ποιότητα", "καλό"))
    res["kind"] = kind[res.i]
    seeds = sorted(res.seed.unique())
    meth = [m for m in ORDER_M if m in set(res.method)]

    # 1. πόσα παράθυρα
    # (ένας σπόρος, μία μέθοδος αρκούν, τα παράθυρα είναι ίδια παντού)
    one = res[(res.seed == seeds[0]) & (res.method == FULL)]
    ii = one.i.to_numpy()
    cnt = pd.DataFrame({
        "είδος": ["λείπει", "χαμηλή_ποιότητα", "καλό"],
        "παράθυρα": [int((kind[ii] == k).sum()) for k in ("λείπει", "χαμηλή_ποιότητα", "καλό")],
        "οδηγοί": [one[kind[ii] == k].subject.nunique() for k in ("λείπει", "χαμηλή_ποιότητα", "καλό")]})
    for b in range(NB):
        cnt[f"{BR_GR[BR[b]]}"] = [int(miss[ii, b].sum()), int(lowq[ii, b].sum()), 0]
    cnt["% ελέγχου"] = (100 * cnt["παράθυρα"] / len(ii)).round(1)
    print(f"\n=== Παράθυρα ελέγχου ανά είδος ({len(ii)} συνολικά · οι στήλες αισθητήρων: σε πόσα λείπει / "
          "έχει χαμηλή ποιότητα ο καθένας) ===")
    print(cnt.to_string(index=False))
    cnt.to_csv(f"{out}/realbad_counts.csv", index=False, encoding="utf-8-sig")

    # 2. F1 ανά είδος, μέσος όρος σπόρων
    def table(sel_fn, labels):
        # για κάθε κατηγορία: F1 ανά σπόρο/μέθοδο, μετά μέσος ± SD των σπόρων
        rows = []
        for lab in labels:
            sub = res[sel_fn(lab)]
            for (sd, m), g in sub.groupby(["seed", "method"]):
                rows.append(dict(seed=sd, method=m, cat=lab, F1=f1m(g.y_true, g.y_pred), n=len(g)))
        t = pd.DataFrame(rows)
        return t.groupby(["method", "cat"]).agg(mean=("F1", "mean"), std=("F1", "std"), n=("n", "first")).reset_index()

    cats = ["λείπει", "χαμηλή_ποιότητα", "όλα_τα_κακά", "καλό"]
    sel = {"λείπει": res.kind == "λείπει", "χαμηλή_ποιότητα": res.kind == "χαμηλή_ποιότητα",
           "όλα_τα_κακά": res.kind != "καλό", "καλό": res.kind == "καλό"}
    t1 = table(lambda c: sel[c], [c for c in cats if sel[c].any()])
    t1.to_csv(f"{out}/realbad_f1_by_kind.csv", index=False, encoding="utf-8-sig")
    cell = t1.assign(c=t1["mean"].map("{:.3f}".format) + "±" + t1["std"].fillna(0).map("{:.3f}".format))
    piv = cell.pivot_table(index="method", columns="cat", values="c", aggfunc="first")
    piv = piv.loc[[m for m in meth if m in piv.index], [c for c in cats if c in piv.columns]]
    print(f"\n=== macro F1 ανά είδος κακού σήματος · μέσος όρος {len(seeds)} σπόρων ± SD · χωρίς τεχνητή βλάβη ===")
    print(piv.to_string())

    # 3. ανά αισθητήρα (μόνο οι τρεις μέθοδοι που μας ενδιαφέρουν)
    key = [FULL, "Krishna_ίδια_εκπ", "-ανωμαλία"]
    labs, selb = [], {}
    for b in range(NB):
        for nm, M in (("λείπει", miss), ("χαμηλή", lowq)):
            lab = f"{nm}:{BR[b]}"                                        # π.χ. "λείπει:face"
            if M[ii, b].sum() >= 30:                                     # τουλάχιστον 30 παράθυρα
                labs.append(lab); selb[lab] = M[res.i.to_numpy(), b]
    if labs:
        t2 = table(lambda c: selb[c], labs)
        t2 = t2[t2.method.isin(key)]
        t2.to_csv(f"{out}/realbad_f1_by_sensor.csv", index=False, encoding="utf-8-sig")
        p2 = t2.pivot_table(index="method", columns="cat", values="mean").loc[[k for k in key if k in set(t2.method)], labs]
        nrow = t2.drop_duplicates("cat").set_index("cat").n.reindex(labs)
        print("\n=== macro F1 ανά αισθητήρα που λείπει / έχει χαμηλή ποιότητα (στήλες με ≥ 30 παράθυρα) ===")
        print(p2.round(3).to_string())
        print("  παράθυρα: " + " · ".join(f"{c} = {int(v)}" for c, v in nrow.items()))

    # 4. Wilcoxon ανά οδηγό (F1 οδηγού, μέσος όρος σπόρων) + Holm
    rows = []
    for c in ["λείπει", "χαμηλή_ποιότητα", "όλα_τα_κακά"]:
        sub = res[sel[c]]
        if sub.empty:
            continue
        ps = (sub.groupby(["seed", "method", "subject"]).apply(lambda g: pd.Series(
            dict(F1=f1m(g.y_true, g.y_pred), n=len(g)))).reset_index())
        # οδηγοί με < 5 τέτοια παράθυρα βγαίνουν (το F1 τους είναι πολύ ασταθές)
        ps = ps[ps.n >= 5].groupby(["method", "subject"]).F1.mean().unstack(0)
        for other in ("Krishna_ίδια_εκπ", "-ανωμαλία"):
            if other not in ps or FULL not in ps:
                continue
            d = (ps[FULL] - ps[other]).dropna()
            try:
                p = wilcoxon(d).pvalue if (d != 0).any() and len(d) >= 5 else np.nan
            except ValueError:
                p = np.nan
            rows.append(dict(είδος=c, σύγκριση=f"FULL έναντι {other}", οδηγοί=len(d),
                             νίκες=int((d > 0).sum()), ήττες=int((d < 0).sum()),
                             μέση_διαφορά=round(100 * d.mean(), 1), p=p))
    st = pd.DataFrame(rows)
    if not st.empty:
        st["H"] = holm(st.p.to_numpy())
        st["σημαντικό"] = np.where(st.H < 0.05, "*", "")
        st.to_csv(f"{out}/realbad_stats.csv", index=False, encoding="utf-8-sig")
        show = st.assign(p=st.p.map(lambda v: f"{v:.3f}"), H=st.H.map(lambda v: f"{v:.3f}"))
        print("\n=== Wilcoxon ανά οδηγό (οδηγοί με ≥ 5 τέτοια παράθυρα) · μέση διαφορά σε μονάδες F1 × 100 · "
              "H = p μετά Holm · * = σημαντικό ===")
        print(show.to_string(index=False))

    # 5. η υπόθεση
    # αν η ανωμαλία χαλάει κυρίως εκεί που το σήμα λείπει, τότε η διαφορά "-ανωμαλία − FULL"
    # θα είναι μεγαλύτερη στο "λείπει" απ' ό,τι στη "χαμηλή_ποιότητα"
    mn = t1.pivot_table(index="method", columns="cat", values="mean")
    if "-ανωμαλία" in mn.index and FULL in mn.index:
        print("\n=== Έλεγχος της υπόθεσης · (χωρίς ανωμαλία) − FULL σε μονάδες F1 × 100 "
              "(θετικό = ο δείκτης ανωμαλίας ΧΑΛΑΕΙ το αποτέλεσμα) ===")
        for c in ["λείπει", "χαμηλή_ποιότητα", "καλό"]:
            if c in mn.columns:
                print(f"  {c:<16} {100 * (mn.loc['-ανωμαλία', c] - mn.loc[FULL, c]):+5.1f}")
        print("  Η υπόθεση στηρίζεται αν η διαφορά είναι μεγαλύτερη στο «λείπει» απ' ό,τι στη «χαμηλή_ποιότητα».")


# =============================================================================
# Β. SHAP
# =============================================================================
def shapley_groups(f, X, bg, groups, chunk=300_000):
    """Ακριβείς τιμές Shapley ανά ομάδα χαρακτηριστικών (interventional).
    f: (N, F) -> (N, C) πιθανότητες · X: (n, F) τα παράθυρα που εξηγούνται · bg: (m, F) δείγμα αναφοράς ·
    groups: λίστα από πίνακες δεικτών στηλών. Για κάθε συνδυασμό S: v(S) = μέσος όρος του f όταν οι ομάδες του S
    παίρνουν τις τιμές του παραθύρου και οι υπόλοιπες τις τιμές του δείγματος αναφοράς.
    Οι συνδυασμοί υπολογίζονται μαζί, σε λίγες μεγάλες κλήσεις του μοντέλου (γρηγορότερο).
    Επιστρέφει phi (n, G, C), base (C,) και fx (n, C)."""
    # Κάθε συνδυασμός S είναι ένας ακέραιος 0 .. 2^G − 1: το bit g = 1 σημαίνει ότι η ομάδα g είναι "μέσα".
    # π.χ. G = 3, S = 5 = 101 -> οι ομάδες 0 και 2 παίρνουν τις πραγματικές τιμές, η 1 του background.
    n, F = X.shape
    m, G = len(bg), len(groups)
    per = n * m                                  # γραμμές ανά συνδυασμό (κάθε παράθυρο × κάθε background)
    v = np.empty((1 << G, n, NC))                # v[S] = μέση πρόβλεψη για τον συνδυασμό S
    batch, Zs = [], []

    def flush():
        # μία κλήση του μοντέλου για πολλούς συνδυασμούς μαζί
        P = f(np.vstack(Zs))
        for k, S in enumerate(batch):
            v[S] = P[k * per:(k + 1) * per].reshape(n, m, -1).mean(1)   # μέσος όρος πάνω στο background
        batch.clear(); Zs.clear()

    for S in range(1 << G):
        Z = np.repeat(bg[None], n, 0)                                   # (n, m, F)
        cols = [groups[g] for g in range(G) if S >> g & 1]              # στήλες των ομάδων που είναι στο S
        if cols:
            c = np.concatenate(cols)
            Z[:, :, c] = X[:, None, c]                                  # αυτές παίρνουν τις τιμές του παραθύρου
        batch.append(S); Zs.append(Z.reshape(per, F))
        if len(batch) * per >= chunk:                                   # όριο μνήμης
            flush()
    if batch:
        flush()
    C = NC
    phi = np.zeros((n, G, C))
    # τύπος Shapley: φ_g = Σ_S w(|S|) · [v(S ∪ {g}) − v(S)], για όλα τα S που δεν περιέχουν το g
    # w = |S|! (G − |S| − 1)! / G!
    for g in range(G):
        for S in range(1 << G):
            if S >> g & 1:
                continue
            k = bin(S).count("1")                                       # |S|
            w = factorial(k) * factorial(G - k - 1) / factorial(G)
            phi[:, g] += w * (v[S | (1 << g)] - v[S])
    # base = v(κενό), fx = v(όλες οι ομάδες) = κανονική πρόβλεψη
    return phi, v[0].mean(0), v[(1 << G) - 1]


def meta_names():
    """Ονόματα και ομάδες των εισόδων του κριτή, με τη σειρά του axis7.feats (όλα τα κομμάτια)."""
    # για κάθε στήλη: όνομα, σε ποιον κλάδο ανήκει, τι είδος είναι
    # ΠΡΟΣΟΧΗ: η σειρά πρέπει να είναι ίδια με το feats() (ελέγχεται με assert πιο κάτω)
    names, br, typ = [], [], []
    def add(nm, b, t):
        names.append(nm); br.append(b); typ.append(t)
    for b in BR:
        for c in CLS:
            add(f"P_{b}_{c}", b, "πιθανότητες")
    for b in BR: add(f"διαθέσιμο_{b}", b, "διαθεσιμότητα")
    for b in BR: add(f"ποιότητα_{b}", b, "ποιότητα σήματος")
    for b in BR: add(f"βεβαιότητα_{b}", b, "βεβαιότητα")
    for b in BR: add(f"ηλικία_LOCF_{b}", b, "ηλικία LOCF")
    for b in BR:
        add(f"ανωμαλία_ποσοστό_{b}", b, "ανωμαλία"); add(f"ανωμαλία_μέσο_z_{b}", b, "ανωμαλία")
    for b in BR: add(f"συμφωνία_{b}", b, "συμφωνία")
    for b in BR:
        for c in CLS:
            add(f"χρόνος_P_{b}_{c}", b, "χρόνος")
        add(f"χρόνος_πλήθος_{b}", b, "χρόνος")
    return names, np.array(br), np.array(typ)


TYPES = ["πιθανότητες", "διαθεσιμότητα", "ποιότητα σήματος", "βεβαιότητα", "ηλικία LOCF", "ανωμαλία", "συμφωνία",
         "χρόνος"]


def signal_of(col):
    """bio__HR__mean -> HR · veh__speed[m/s]__std -> speed · face__FAU_eye_closed -> FAU μάτια."""
    rest = col.split("__", 1)[1]                 # βγάζω το "bio__"
    tok = rest.split("__")[0].split("[")[0]      # βγάζω το "__mean" και τις μονάδες "[m/s]"
    # τα FAU είναι πολλά, τα μαζεύω σε ομάδες ανά περιοχή του προσώπου
    if tok.startswith("FAU_"):
        low = tok.lower()
        if any(k in low for k in ("jaw_drop", "mouth_stretch", "lips_part", "yawn")):
            return "FAU χασμουρητό"                 # AU25-27: άνοιγμα στόματος / σαγονιού
        if any(k in low for k in ("eye", "blink", "lid", "slit", "squint", "wink")):
            return "FAU μάτια"                      # AU5, 7, 41-46
        if any(k in low for k in ("lip", "chin", "mouth")):
            return "FAU χείλη"                      # AU10-28 χωρίς το χασμουρητό
        if "brow" in low:
            return "FAU φρύδια"                     # AU1, 2, 4
        return "FAU μάγουλα/μύτη"                   # AU6, 9, 11, 13, 14
    return tok


def signal_groups(cols):
    # ομαδοποίηση στηλών ενός κλάδου ανά σήμα
    lab = [signal_of(c) for c in cols]
    names = list(dict.fromkeys(lab))             # μοναδικά, με τη σειρά που εμφανίζονται
    groups = {nm: np.array([i for i, l in enumerate(lab) if l == nm]) for nm in names}
    while len(groups) > MAX_GROUPS:                                     # ενώνονται οι δύο μικρότερες ομάδες
        a, b = sorted(groups, key=lambda k: len(groups[k]))[:2]
        groups[f"{a}+{b}"] = np.concatenate([groups.pop(a), groups.pop(b)])
    return list(groups), [groups[k] for k in groups]


def importance(phi):
    """Σημαντικότητα ανά ομάδα: μέσο |φ| στις 3 κλάσεις και στα παράθυρα · και μόνο για την κλάση «υψηλή»."""
    # απόλυτη τιμή: μετράει πόσο επηρεάζει, όχι προς ποια κατεύθυνση
    return np.abs(phi).mean(axis=(0, 2)), np.abs(phi[:, :, 2]).mean(0)


def run_shap(df, X, avail, Qall, feat, seed, trees, n_explain, n_bg, noise_sd, out):
    print("\n" + "=" * 100 + "\nΒ. SHAP · ΠΟΙΟΙ ΑΙΣΘΗΤΗΡΕΣ ΚΑΙ ΠΟΙΑ ΣΗΜΑΤΑ ΚΡΙΝΟΥΝ ΤΗΝ ΑΠΟΦΑΣΗ (νέος οδηγός, LOSO)\n" + "=" * 100)
    ul.SEED = seed
    y = df.y.to_numpy(int)
    groups = df.subject.astype(str).to_numpy()
    seg, order = segments(df)
    n = len(df)
    sd = {b: np.nanstd(X[b], axis=0) for b in BR}
    cal = a3.calib_mask(df, 4)
    rng = np.random.default_rng(seed)
    erng = np.random.default_rng(seed + 99)                            # μόνο για τα δείγματα του SHAP
    srng = np.random.default_rng(seed + 7)
    masks = {"all": np.zeros((n, NB), bool), "random60": srng.random((n, NB)) < 0.6}   # όπως στο axis7
    mnames, mbr, mtyp = meta_names()
    # δύο τρόποι ομαδοποίησης των εισόδων του κριτή: ανά κλάδο / ανά είδος
    g_mod = [np.where(mbr == b)[0] for b in BR]
    g_typ = [np.where(mtyp == t)[0] for t in TYPES]
    bgroups = {b: signal_groups(feat[b]) for b in BR}
    for b in BR:
        print(f"  ομάδες σημάτων {BR_GR[b]}: " + ", ".join(bgroups[b][0]))
    rec_meta, rec_br, rec_subj = [], [], []
    add_err = 0.0                                # μέγιστο σφάλμα αθροιστικότητας (πρέπει να είναι ~0)
    subjects = sorted(np.unique(groups))
    for si, s in enumerate(subjects, 1):
        t0 = time.time()
        tr = np.where(groups != s)[0]
        sub = np.where(groups == s)[0]
        te = np.where((groups == s) & ~cal)[0]
        if len(te) == 0:
            continue
        # ίδια εκπαίδευση με το FULL του axis7
        B = build(X, avail, sd, y, groups, tr, sub, n, trees, rng, df, seg, order, Qall)
        copies = B["real"] + B["noisec"]
        Xm = np.vstack([feats(*c, tr, seg, order) for c in copies])
        assert Xm.shape[1] == len(mnames), f"{Xm.shape[1]} είσοδοι κριτή αντί για {len(mnames)}"
        meta = ul.fit_meta(Xm, np.concatenate([y[tr]] * len(copies)))
        bg = Xm[erng.choice(len(Xm), min(n_bg, len(Xm)), replace=False)]   # background από το train

        # Β1. κριτής
        for sc in ("all", "random60"):
            trng = np.random.default_rng(seed + 1000 + a7.SCEN.index(sc))   # ίδιο σενάριο με το axis7
            _, st = test_state(B, X, sd, avail, sub, n, sc, masks[sc], trng, seg, order, None, noise_sd)
            # εξηγώ μόνο n_explain παράθυρα ανά οδηγό (αλλιώς αργεί πολύ)
            rows = te if len(te) <= n_explain else np.sort(erng.choice(te, n_explain, replace=False))
            Xe = feats(*st, rows, seg, order)
            for scheme, grp, labels in (("αισθητήρας", g_mod, [BR_GR[b] for b in BR]), ("είδος", g_typ, TYPES)):
                phi, base, fx = shapley_groups(meta.predict_proba, Xe, bg, grp)
                add_err = max(add_err, float(np.abs(phi.sum(1) + base - fx).max()))
                imp, imp_hi = importance(phi)
                for lab, a, h in zip(labels, imp, imp_hi):
                    rec_meta.append(dict(subject=s, scenario=sc, scheme=scheme, group=lab, imp=a, imp_high=h,
                                         n=len(rows)))

        # Β2. μέσα σε κάθε κλάδο
        for b in BR:
            mdl = B["models"].get(b)
            ok_te = te[avail[b][te]]             # μόνο παράθυρα όπου ο κλάδος υπάρχει
            ok_tr = tr[avail[b][tr]]
            if mdl is None or len(ok_te) == 0:
                continue
            rows = ok_te if len(ok_te) <= n_explain else np.sort(erng.choice(ok_te, n_explain, replace=False))
            bgb = X[b][erng.choice(ok_tr, min(n_bg, len(ok_tr)), replace=False)]
            labels, grp = bgroups[b]
            phi, base, fx = shapley_groups(lambda Z: ul.full_proba(mdl, Z), X[b][rows], bgb, grp)
            add_err = max(add_err, float(np.abs(phi.sum(1) + base - fx).max()))
            imp, imp_hi = importance(phi)
            for lab, a, h in zip(labels, imp, imp_hi):
                rec_br.append(dict(subject=s, branch=b, signal=lab, imp=a, imp_high=h, n=len(rows)))
        print(f"  SHAP · LOSO {si}/{len(subjects)}: νέος οδηγός {s}  ({time.time() - t0:.0f} s)", flush=True)

    meta_df, br_df = pd.DataFrame(rec_meta), pd.DataFrame(rec_br)
    meta_df.to_csv(f"{out}/shap_meta_per_subject.csv", index=False, encoding="utf-8-sig")
    br_df.to_csv(f"{out}/shap_branch_per_subject.csv", index=False, encoding="utf-8-sig")
    # αν βγει ~0, ο υπολογισμός Shapley είναι σωστός
    print(f"\n  Έλεγχος αθροιστικότητας (Σ φ + βάση = πρόβλεψη): μέγιστο σφάλμα {add_err:.1e} "
          f"{'· σωστό' if add_err < 1e-6 else '· ΠΡΟΣΟΧΗ'}")

    # --- πίνακες
    def share(d, keys, col="imp"):
        # μέσος όρος στους οδηγούς, μετά % ώστε οι ομάδες να αθροίζουν 100
        g = d.groupby(keys)[col].mean().reset_index()
        tot = g.groupby(keys[:-1])[col].transform("sum")
        g["share"] = 100 * g[col] / tot
        return g

    sm = share(meta_df, ["scenario", "scheme", "group"])
    smh = share(meta_df, ["scenario", "scheme", "group"], "imp_high").rename(columns={"share": "share_high"})
    sm = sm.merge(smh[["scenario", "scheme", "group", "share_high"]], on=["scenario", "scheme", "group"])
    sm.to_csv(f"{out}/shap_meta.csv", index=False, encoding="utf-8-sig")
    for scheme, labels in (("αισθητήρας", [BR_GR[b] for b in BR]), ("είδος", TYPES)):
        t = sm[sm.scheme == scheme].pivot_table(index="group", columns="scenario", values="share").reindex(labels)
        th = sm[sm.scheme == scheme].pivot_table(index="group", columns="scenario", values="share_high").reindex(labels)
        t = t.round(1).astype(str) + "%"
        t.columns = [f"{c}" for c in t.columns]
        for c in th.columns:
            t[f"{c} · υψηλή"] = th[c].round(1).astype(str) + "%"
        print(f"\n=== Κριτής · μερίδιο της απόφασης ανά {scheme} (μέσο |SHAP|, άθροισμα = 100%) · "
              "«υψηλή» = μόνο για την κλάση υψηλής κόπωσης ===")
        print(t.to_string())

    # ανά οδηγό: ποιος αισθητήρας κυριαρχεί στον καθένα
    ps = meta_df[(meta_df.scheme == "αισθητήρας") & (meta_df.scenario == "all")]
    pt = ps.pivot_table(index="subject", columns="group", values="imp")[[BR_GR[b] for b in BR]]
    pt = (100 * pt.div(pt.sum(1), axis=0)).round(0)
    pt["κυρίαρχος"] = pt.idxmax(1)
    pt.to_csv(f"{out}/shap_meta_modality_per_subject.csv", encoding="utf-8-sig")
    print("\n=== Κριτής · μερίδιο ανά αισθητήρα για κάθε νέο οδηγό (σενάριο all, %) ===")
    print(pt.astype({BR_GR[b]: int for b in BR}).to_string())
    print("  κυρίαρχος αισθητήρας: " + " · ".join(f"{k} σε {v} οδηγούς" for k, v in pt["κυρίαρχος"].value_counts().items()))

    sb = share(br_df, ["branch", "signal"])
    sbh = share(br_df, ["branch", "signal"], "imp_high").rename(columns={"share": "share_high"})
    sb = sb.merge(sbh[["branch", "signal", "share_high"]], on=["branch", "signal"])
    sb.to_csv(f"{out}/shap_branch.csv", index=False, encoding="utf-8-sig")
    print("\n=== Μέσα σε κάθε κλάδο · μερίδιο ανά σήμα (μέσο |SHAP|, άθροισμα = 100% ανά κλάδο) ===")
    for b in BR:
        t = sb[sb.branch == b].sort_values("share", ascending=False)
        if t.empty:
            continue
        print(f"  {BR_GR[b]:<11} " + " · ".join(f"{r.signal} {r.share:.0f}% (υψηλή {r.share_high:.0f}%)"
                                                 for r in t.itertuples()))
    plots(sm, sb, out)


def plots(sm, sb, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (χωρίς matplotlib δεν φτιάχνονται γραφήματα)")
        return
    col = {"all": "#14213D", "random60": "#B4531A"}
    lab = {"all": "όλοι οι αισθητήρες", "random60": "60% τυχαία κενά"}
    # γράφημα 1: κριτής, αριστερά ανά αισθητήρα, δεξιά ανά είδος (μπάρες δίπλα-δίπλα για τα 2 σενάρια)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(width_ratios=[1, 2]))
    for k, (scheme, labels) in enumerate((("αισθητήρας", [BR_GR[b] for b in BR]), ("είδος", TYPES))):
        d = sm[sm.scheme == scheme]
        x = np.arange(len(labels))
        scs = [s for s in ("all", "random60") if s in set(d.scenario)]
        for j, sc in enumerate(scs):
            v = d[d.scenario == sc].set_index("group").share.reindex(labels).to_numpy()
            ax[k].bar(x + (j - (len(scs) - 1) / 2) * 0.38, v, 0.36, color=col[sc], label=lab[sc])
        ax[k].set_xticks(x); ax[k].set_xticklabels(labels, rotation=0 if k == 0 else 30, ha="center" if k == 0 else "right")
        ax[k].set_ylabel("μερίδιο της απόφασης (%)")
        ax[k].set_title("Κριτής · ανά αισθητήρα" if k == 0 else "Κριτής · ανά είδος εισόδου")
        ax[k].spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{out}/shap_kritis.png", dpi=150); plt.close(fig)

    # γράφημα 2: ένα panel ανά κλάδο, σήματα ταξινομημένα
    br = [b for b in BR if b in set(sb.branch)]
    if br:
        fig, ax = plt.subplots(1, len(br), figsize=(5 * len(br), 4.5))
        ax = np.atleast_1d(ax)
        for k, b in enumerate(br):
            t = sb[sb.branch == b].sort_values("share")
            ax[k].barh(t.signal, t.share, color="#14213D")
            ax[k].set_title(f"Κλάδος: {BR_GR[b]}")
            ax[k].set_xlabel("μερίδιο (%)")
            ax[k].spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(f"{out}/shap_kladoi.png", dpi=150); plt.close(fig)
    print(f"  Γραφήματα: {out}/shap_kritis.png, {out}/shap_kladoi.png")


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--ablation", default="results_ablation", help="φάκελος αποτελεσμάτων του axis7")
    ap.add_argument("--out", default="results_shap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--n-explain", type=int, default=40, help="παράθυρα που εξηγούνται ανά νέο οδηγό")
    ap.add_argument("--n-bg", type=int, default=30, help="μέγεθος δείγματος αναφοράς")
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--no-shap", action="store_true", help="μόνο το μέρος Α (πραγματικό κακό σήμα)")
    ap.add_argument("--no-realbad", action="store_true", help="μόνο το μέρος Β (SHAP)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 260)
    # ίδια προετοιμασία με το axis7
    df = a3.load(a.features)
    feat = {b: [c for c in df.columns if c.startswith(f"{b}__")] for b in BR}
    X = {b: a3.baseline_normalize(df[feat[b]].to_numpy(float), df, 4, "sessionA") for b in BR}
    avail = {b: ~np.isnan(df[feat[b]].to_numpy(float)).all(axis=1) for b in BR}
    qcols = [f"q__{b}" for b in BR]
    Qall = (df[qcols].fillna(0.0).to_numpy(float) if all(c in df for c in qcols)
            else np.stack([avail[b] for b in BR], 1).astype(float))
    print(f"{len(df)} παράθυρα, {df.subject.nunique()} οδηγοί")
    if not a.no_realbad:
        realbad_analysis(df, avail, Qall, a.ablation, a.out)
    if not a.no_shap:
        t0 = time.time()
        run_shap(df, X, avail, Qall, feat, a.seed, a.trees, a.n_explain, a.n_bg, a.noise, a.out)
        print(f"  SHAP: {(time.time() - t0) / 60:.1f} λεπτά")
    print(f"\nΑρχεία στον φάκελο '{a.out}': realbad_*.csv, shap_meta.csv, shap_branch.csv, "
          "shap_meta_modality_per_subject.csv, shap_*_per_subject.csv, shap_kritis.png, shap_kladoi.png")


if __name__ == "__main__":
    main()
