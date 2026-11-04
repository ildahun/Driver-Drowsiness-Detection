#!/usr/bin/env python3
"""
UL-DD: πρώτο πείραμα (baseline + late fusion + ελλιπείς τροπικότητες), με leave-one-subject-out.

Χρήση:
    pip install pandas numpy scikit-learn
    python ul_dd_baseline.py --root /path/to/UL-DD --labels /path/to/UL-DD/Labels.csv
    (Windows: python ul_dd_baseline.py --root "C:\\Users\\...\\UL-DD" --labels "C:\\Users\\...\\UL-DD\\Labels.csv")

Βγάζει: features.csv (χαρακτηριστικά ανά παράθυρο), results/summary_loso.csv (συνολικά αποτελέσματα),
        results/per_subject.csv (ακρίβεια ανά οδηγό).

Το --root είναι ο φάκελος που περιέχει τους CSV_Files και Extracted_Features (το script
ψάχνει αναδρομικά τα αρχεία <User>_<SIGNAL>_<Session>.csv, άρα δεν πειράζει η ακριβής δομή φακέλων).

Παραδοχές (ελέγξτε τες στην πρώτη εκτέλεση, το script τυπώνει διαγνωστικά):
  * Κάθε session είναι 40 λεπτά = 10 διαστήματα των 4 λεπτών, ένα KSS ανά διάστημα (Labels.csv).
  * Όλα τα σήματα ξεκινούν στο t=0 του session (το dataset είναι συγχρονισμένο).
  * Τάξεις: Alert = KSS<4, Medium = 4..6, High = KSS>6 (όπως στο άρθρο του UL-DD).
"""
# =====================================================================================================
# ΣΥΝΟΛΙΚΗ ΕΙΚΟΝΑ ΤΟΥ SCRIPT (διαβάστε το πρώτα)
# -----------------------------------------------------------------------------------------------------
# Στόχος: να προβλέψουμε το επίπεδο υπνηλίας του οδηγού (Alert / Medium / High) από τρεις "κλάδους"
# (τροπικότητες) δεδομένων του UL-DD:
#     1) bio  : βιομετρικά (καρδιακός ρυθμός, ηλεκτροδερμική δραστηριότητα, θερμοκρασία, BVP, επιτάχυνση, SpO2)
#     2) veh  : όχημα (τηλεμετρία προσομοιωτή + πίεση στις λαβές του τιμονιού)
#     3) face : πρόσωπο (EAR / PERCLOS από τα 68 σημεία του προσώπου + facial action units)
#
# Ροή εκτέλεσης (συνάρτηση main στο τέλος):
#   Α. build_dataset: διαβάζει όλα τα CSV, κόβει κάθε session σε παράθυρα (π.χ. 30 s) και για κάθε παράθυρο
#      υπολογίζει στατιστικά (μέσος, τυπική απόκλιση, min, max, κλίση) ανά σήμα. Κάθε παράθυρο παίρνει
#      την ετικέτα KSS του 4λεπτου διαστήματος στο οποίο ανήκει. Αποθηκεύεται στο features.csv (cache).
#   Β. run_loso: αξιολόγηση με leave-one-subject-out (ένας οδηγός στο test κάθε φορά) ή/και 5-fold.
#      Σε κάθε fold εκπαιδεύονται:
#         - ένα Random Forest ανά κλάδο (unimodal μοντέλα),
#         - διάφοροι meta-classifiers πάνω στις πιθανότητες των κλάδων (late fusion, συμπεριλαμβανομένης
#           της ΠΡΟΤΕΙΝΟΜΕΝΗΣ μεθόδου late_meta_reliability),
#         - μοντέλα πρώιμης σύντηξης (early fusion) σε όλα τα χαρακτηριστικά μαζί, για σύγκριση.
#      Επίσης δοκιμάζονται σενάρια όπου ένας κλάδος "λείπει" στο test (no_bio, no_veh, no_face).
#   Γ. summarize / axis1_stats / main_table: μετρικές (macro F1, recall High, kappa κ.λπ.), διαστήματα
#      εμπιστοσύνης, έλεγχος Wilcoxon, πίνακες και γραφήματα στον φάκελο results/.
#
# "Άξονας 1" = σύγκριση μεθόδων σύντηξης όταν υπάρχουν όλα τα δεδομένα.
# "Άξονας 2" = ανθεκτικότητα όταν λείπει κάποιος αισθητήρας.
# =====================================================================================================

# --- Βιβλιοθήκες της Python
import argparse                  # διάβασμα παραμέτρων από τη γραμμή εντολών (--root, --labels, ...)
import re                        # κανονικές εκφράσεις (για να "διαβάσουμε" τα ονόματα στο Labels.csv)
import warnings                  # για να κρύβουμε προειδοποιήσεις που δεν μας ενδιαφέρουν
from pathlib import Path         # χειρισμός διαδρομών αρχείων/φακέλων ανεξάρτητα από λειτουργικό

import numpy as np               # αριθμητικοί πίνακες
import pandas as pd              # πίνακες δεδομένων (DataFrame), ανάγνωση CSV
# --- Μοντέλα και εργαλεία από το scikit-learn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer          # συμπλήρωση ελλιπών τιμών (NaN)
import itertools                 # συνδυασμοί (π.χ. όλα τα ζεύγη κλάδων για το ablation)
import time                      # μέτρηση χρόνου πρόβλεψης

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold   # τρόποι διαχωρισμού train/test
from sklearn.pipeline import Pipeline             # αλυσίδα βημάτων (π.χ. imputer -> scaler -> μοντέλο)
from sklearn.decomposition import PCA             # μείωση διαστάσεων
from sklearn.dummy import DummyClassifier         # "τυχαίος" ταξινομητής (κάτω όριο σύγκρισης)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler   # κανονικοποίηση χαρακτηριστικών
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight      # βάρη ώστε οι σπάνιες τάξεις να μετράνε περισσότερο

# ----------------------------------------------------------------------------- σταθερές
SEED = 42                 # σπόρος τυχαιότητας: ίδια αποτελέσματα σε κάθε εκτέλεση (αναπαραγωγιμότητα)
N_CLASSES = 3             # τρεις τάξεις υπνηλίας
CLASS_NAMES = ["Alert", "Medium", "High"]   # 0 = Alert, 1 = Medium, 2 = High
INTERVAL_S = 240          # 4 λεπτά ανά ετικέτα KSS
SESSION_S = 2400          # 40 λεπτά

# Συχνότητες δειγματοληψίας (Hz / fps) όπως περιγράφονται στο άρθρο. Αλλάξτε τες αν διαφέρουν.
# Σημ.: το άρθρο γράφει 4 Hz για το BVP, αλλά στα πραγματικά αρχεία είναι 64 Hz (153.600 γραμμές / 40 λεπτά).
# Χρησιμεύουν για να μετατρέπουμε χρόνο (δευτερόλεπτα) σε αριθμό γραμμής: γραμμή = χρόνος * συχνότητα.
# HR=καρδιακός ρυθμός, EDA=ηλεκτροδερμική δραστηριότητα, TEMP=θερμοκρασία δέρματος, BVP=όγκος αίματος,
# ACC=επιταχυνσιόμετρο καρπού, O2M=οξύμετρο (SpO2, παλμοί, κίνηση), RGP/LGP=πίεση δεξιάς/αριστερής λαβής,
# Telemetry=τηλεμετρία προσομοιωτή, FL=facial landmarks (68 σημεία), FAU=facial action units.
RATES = {"HR": 1.0, "EDA": 4.0, "TEMP": 4.0, "BVP": 64.0, "ACC": 32.0, "O2M": 0.5,
         "RGP": 3.0, "LGP": 3.0, "Telemetry": 60.0, "FL": 60.0, "FAU": 60.0}

EAR_CLOSED = 0.20         # κατώφλι κλεισίματος ματιού για PERCLOS
MIN_VALID_FRAC = 0.5      # κάτω από αυτό το ποσοστό έγκυρων δειγμάτων στο παράθυρο => ο κλάδος "λείπει"
BRANCHES = ["bio", "veh", "face"]                       # οι τρεις κλάδοι (τροπικότητες)
BRANCH_PREFIXES = tuple(f"{b}__" for b in BRANCHES)      # ("bio__", "veh__", "face__"): πρόθεμα στηλών κάθε κλάδου


# ----------------------------------------------------------------------------- φόρτωση αρχείων
_INDEX = {}   # λεξικό: όνομα αρχείου (πεζά) -> λίστα διαδρομών όπου βρέθηκε. Γεμίζει από την index_files.


def index_files(root):
    """Χτίζει μία φορά τον κατάλογο όλων των CSV κάτω από το root (πολύ γρηγορότερο από αναζήτηση ανά αρχείο)."""
    _INDEX.clear()
    roots = root if isinstance(root, (list, tuple)) else [root]       # μπορούν να δοθούν πολλοί φάκελοι
    for r in roots:
        if not Path(r).exists():
            raise SystemExit(f"Ο φάκελος δεν υπάρχει: {r}\nΕλέγξτε το --root.")
        for p in Path(r).rglob("*.csv"):                    # rglob = αναζήτηση και σε όλους τους υποφακέλους
            _INDEX.setdefault(p.name.lower(), []).append(p)  # πεζά, ώστε "A_HR_A.csv" και "a_hr_a.csv" να ταιριάζουν
    if not _INDEX:
        raise SystemExit(f"Δεν βρέθηκε κανένα .csv κάτω από: {roots}\nΕλέγξτε το --root.")
    print(f"Βρέθηκαν {sum(len(v) for v in _INDEX.values())} αρχεία CSV κάτω από {roots}")


def find_file(root, user, signal, session):
    """Επιστρέφει τη διαδρομή του αρχείου <user>_<signal>_<session>.csv ή None αν δεν υπάρχει.
    (Το όρισμα root δεν χρησιμοποιείται πια, η αναζήτηση γίνεται στο _INDEX.)"""
    for sig in ([signal] if signal != "LGP" else ["LGP", "LGB"]):   # στο άρθρο το αριστερό grip γράφεται και "LGB"
        hits = _INDEX.get(f"{user}_{sig}_{session}.csv".lower(), [])
        if hits:
            if len(hits) > 1 and len({h.stat().st_size for h in hits}) > 1:   # ίδιο όνομα, διαφορετικό περιεχόμενο
                print(f"  [!] {len(hits)} διαφορετικά αρχεία με όνομα {hits[0].name}, χρησιμοποιείται το: {hits[0]}")
            return hits[0]
    return None


def read_numeric(path):
    """Διαβάζει CSV με ή χωρίς header. Επιστρέφει numeric DataFrame (ονόματα στηλών αν υπάρχουν)."""
    first = pd.read_csv(path, header=None, nrows=1)          # διαβάζει μόνο την 1η γραμμή (εξοικονόμηση μνήμης)
    # Αν κάποιο κελί της 1ης γραμμής δεν είναι αριθμός (π.χ. "timestamp"), τότε είναι γραμμή τίτλων (header).
    has_header = pd.to_numeric(first.iloc[0], errors="coerce").isna().any()
    df = pd.read_csv(path, header=0 if has_header else None, low_memory=False)
    if not has_header:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]   # χωρίς τίτλους -> ονόματα col_0, col_1, ...
    return df.apply(pd.to_numeric, errors="coerce")              # ό,τι δεν είναι αριθμός γίνεται NaN


def load_labels(path):
    """Διαβάζει το Labels.csv. Επιστρέφει λεξικό {(χρήστης, session): πίνακας με 10 τιμές KSS}.
    session = "A" (Alert/ξεκούραστος) ή "D" (Drowsy/νυσταγμένος)."""
    raw = pd.read_csv(path, header=None)
    labels = {}
    for _, row in raw.iterrows():
        name = re.sub(r"\.csv$", "", str(row.iloc[0]).strip(), flags=re.I)   # 1η στήλη = όνομα, χωρίς ".csv"
        # π.χ. "A_Alert", "A_Drowsy", "E_Labels_D" ή "E_D" -> (χρήστης, "A"/"D")
        m = re.match(r"^([A-Za-z]+)_(?:.*_)?(Alert|Drowsy|A|D)$", name, flags=re.I)
        if not m:
            continue                                            # π.χ. γραμμή header
        vals = pd.to_numeric(row.iloc[1:], errors="coerce").to_numpy(dtype=float)   # οι υπόλοιπες στήλες = KSS
        labels[(m.group(1).upper(), m.group(2)[0].upper())] = vals              # [0] -> κρατά μόνο "A" ή "D"
    if not labels:
        raise SystemExit("Δεν αναγνωρίστηκε καμία γραμμή ετικετών στο Labels.csv. Οι πρώτες γραμμές του αρχείου:\n"
                         + raw.head(3).to_string())
    return labels


def kss_to_class(k):
    """Μετατρέπει τιμή KSS (1-9) σε τάξη: 0 = Alert (KSS<4), 1 = Medium (4-6), 2 = High (KSS>6)."""
    if np.isnan(k):
        return np.nan
    return 0 if k < 4 else (1 if k <= 6 else 2)


# ----------------------------------------------------------------------------- στατιστικά παραθύρου
def stats(x, prefix):
    """Για ένα κομμάτι σήματος x επιστρέφει 5 χαρακτηριστικά: μέσος, τυπ. απόκλιση, min, max και κλίση
    (slope = πόσο ανεβαίνει/κατεβαίνει το σήμα μέσα στο παράθυρο, από ευθεία ελαχίστων τετραγώνων).
    Τα ονόματα έχουν τη μορφή <prefix>__mean, π.χ. "bio__HR__mean"."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]                              # αγνοούμε τα ελλιπή δείγματα
    names = ["mean", "std", "min", "max", "slope"]
    if len(x) < 2:                                   # πολύ λίγα δεδομένα -> όλα NaN
        return {f"{prefix}__{n}": np.nan for n in names}
    slope = np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 3 else 0.0   # κλίση ευθείας (βαθμός 1)
    return {f"{prefix}__mean": x.mean(), f"{prefix}__std": x.std(), f"{prefix}__min": x.min(),
            f"{prefix}__max": x.max(), f"{prefix}__slope": slope}


def sl(arr, fs, t0, t1):
    """Κόβει από το σήμα arr (συχνότητα fs Hz) το κομμάτι από t0 έως t1 δευτερόλεπτα."""
    return arr[int(t0 * fs):int(t1 * fs)]


def valid_frac(x):
    """Ποσοστό (0-1) των δειγμάτων που ΔΕΝ είναι NaN. Χρησιμοποιείται ως δείκτης ποιότητας σήματος."""
    x = np.asarray(x, dtype=float)
    return 0.0 if len(x) == 0 else float(np.mean(~np.isnan(x)))


# ----------------------------------------------------------------------------- EAR από 68 landmarks
# EAR (Eye Aspect Ratio) = πόσο "ανοιχτό" είναι το μάτι: λόγος κάθετων αποστάσεων προς την οριζόντια.
# Ανοιχτό μάτι ~0.25-0.35, κλειστό < 0.20. Το PERCLOS = ποσοστό χρόνου που το μάτι είναι κλειστό.
def ear_from_landmarks(lm):
    """lm: (n, 68, 2). Μέσος EAR των δύο ματιών (dlib: 36-41 και 42-47)."""
    def eye(a):
        # a: τα 6 σημεία ενός ματιού (0 και 3 = γωνίες, 1,2 = πάνω βλέφαρο, 4,5 = κάτω βλέφαρο)
        d = lambda i, j: np.linalg.norm(a[:, i] - a[:, j], axis=1)   # ευκλείδεια απόσταση σημείων i, j
        return (d(1, 5) + d(2, 4)) / (2.0 * d(0, 3))                  # τύπος EAR (Soukupová & Čech, 2016)
    with np.errstate(divide="ignore", invalid="ignore"):              # αγνοούμε διαιρέσεις με 0
        e = 0.5 * (eye(lm[:, 36:42]) + eye(lm[:, 42:48]))             # μέσος όρος αριστερού-δεξιού ματιού
    e[~np.isfinite(e)] = np.nan                                        # άπειρα -> NaN
    return e


def landmarks_to_ear(df):
    """Από τον πίνακα FL (136 στήλες = 68 σημεία x 2 συντεταγμένες) υπολογίζει EAR ανά frame.
    Επειδή δεν ξέρουμε σίγουρα τη σειρά των στηλών, δοκιμάζει δύο διατάξεις και κρατά την πιο λογική."""
    arr = df.to_numpy(dtype=float)
    if arr.shape[1] == 137:                      # πρώτη στήλη = αριθμός frame
        arr = arr[:, 1:]
    if arr.shape[1] != 136:
        raise ValueError(f"Αναμένονται 136 (ή 137) στήλες στο FL, βρέθηκαν {arr.shape[1]}")
    n = len(arr)
    # Δύο πιθανές διατάξεις:
    #   interleaved: x0, y0, x1, y1, ...      xy_blocks: x0..x67, y0..y67
    cands = {"interleaved": arr.reshape(n, 68, 2),
             "xy_blocks": np.stack([arr[:, :68], arr[:, 68:]], axis=2)}
    sample = slice(0, min(n, 3000))              # ελέγχουμε μόνο τα πρώτα 3000 frames (ταχύτητα)
    best, best_score = None, np.inf
    for name, lm in cands.items():
        e = ear_from_landmarks(lm[sample])
        med = np.nanmedian(e) if np.isfinite(e).any() else np.nan
        if np.isnan(med) or not (0.10 <= med <= 0.50):   # ρεαλιστικός EAR πρέπει να είναι σε αυτό το εύρος
            continue
        score = np.nanstd(e) / med               # σωστή διάταξη = πιο σταθερός EAR (μικρότερη σχετική διασπορά)
        if score < best_score:
            best, best_score = name, score
    if best is None:
        best = "interleaved"
        print("  [!] Δεν επιβεβαιώθηκε η διάταξη των landmarks, χρήση interleaved. Ελέγξτε το FL.")
    return ear_from_landmarks(cands[best]), best


# ----------------------------------------------------------------------------- κατασκευή χαρακτηριστικών
def session_signals(root, user, session):
    """Φορτώνει όλα τα σήματα ενός session (π.χ. οδηγός A, session D). Επιστρέφει λεξικό
    {όνομα σήματος: DataFrame ή None αν λείπει το αρχείο}."""
    sig = {}
    for name in RATES:
        p = find_file(root, user, name, session)
        sig[name] = read_numeric(p) if p else None
    # Στους αισθητήρες λαβής, η τιμή ακριβώς 0 σημαίνει ότι ο αισθητήρας δεν μετρά (π.χ. στον A το αριστερό
    # grip είναι 0 στο 94% της οδήγησης) -> το θεωρούμε ελλιπές (NaN) και όχι πραγματική μέτρηση.
    for name in ("RGP", "LGP"):
        if sig[name] is not None:
            sig[name] = sig[name].mask(sig[name] == 0)      # mask: όπου ισχύει η συνθήκη -> NaN
    if sig["Telemetry"] is not None:
        sig["Telemetry"] = regrid_telemetry(sig["Telemetry"], f"{user}_{session}")
    return sig


def regrid_telemetry(t, tag):
    """Η τηλεμετρία έχει κενά (π.χ. ο προσομοιωτής σταμάτησε για 7 λεπτά), οπότε δεν αρκεί η αρίθμηση των γραμμών:
    τοποθετούμε κάθε γραμμή στη σωστή χρονική θέση με βάση το timestamp, σε σταθερό πλέγμα 60 Hz.
    Τα κενά μένουν NaN (= ελλιπής τηλεμετρία). Επίσης η πορεία (heading, 0-360°) μετατρέπεται σε ρυθμό αλλαγής
    κατεύθυνσης ανά δείγμα, που είναι η πιο κοντινή διαθέσιμη ένδειξη για διορθώσεις τιμονιού."""
    fs = RATES["Telemetry"]
    # βρίσκουμε τη στήλη χρόνου (περιέχει "timestamp" αλλά όχι "raw")
    tcols = [c for c in t.columns if "timestamp" in str(c).lower() and "raw" not in str(c).lower()]
    if tcols:
        ts = t[tcols[0]].to_numpy(float)
        step = np.nanmedian(np.diff(ts))                                  # τυπικό βήμα μεταξύ γραμμών
        unit = 1e6 if step > 1000 else (1e3 if step > 1 else 1.0)       # μs, ms ή s
        idx = np.round((ts - ts[0]) / unit * fs).astype(int)            # σε ποια θέση του πλέγματος 60 Hz πάει κάθε γραμμή
    else:
        idx = np.arange(len(t))                                           # χωρίς timestamp: απλή αρίθμηση
    for c in t.columns:
        if "heading" in str(c).lower():
            d = np.diff(t[c].to_numpy(float), prepend=np.nan)            # διαφορά διαδοχικών γωνιών
            d = (d + 180) % 360 - 180                                     # αναδίπλωση 359°->0°
            d[np.abs(d) > 30] = np.nan                                    # άλματα (π.χ. επανεκκίνηση) -> ελλιπή
            t[c] = d
            t = t.rename(columns={c: "heading_rate"})
    n = int(SESSION_S * fs)                                               # 2400 s * 60 Hz = 144.000 θέσεις
    keep = (idx >= 0) & (idx < n)                                         # κρατάμε μόνο γραμμές μέσα στα 40 λεπτά
    out = pd.DataFrame(np.nan, index=np.arange(n), columns=t.columns)     # κενό πλέγμα γεμάτο NaN
    out.iloc[idx[keep]] = t.to_numpy()[keep]                              # τοποθετούμε τις γραμμές στη θέση τους
    cover = out.iloc[:, -1].notna().mean()                                # ποσοστό κάλυψης
    if cover < 0.95:
        print(f"  [!] {tag}: η τηλεμετρία καλύπτει το {cover:.0%} της οδήγησης (κενά = ελλιπή δεδομένα)")
    return out


def build_session_features(root, user, session, labels, window_s, info):
    """ΚΕΝΤΡΙΚΗ συνάρτηση εξαγωγής χαρακτηριστικών για ένα session.
    Για κάθε 4λεπτο διάστημα με ετικέτα, το σπάει σε παράθυρα των window_s δευτερολέπτων και για καθένα
    φτιάχνει μια γραμμή με: ταυτότητα (subject, session, interval, window), ετικέτα y,
    τα χαρακτηριστικά των τριών κλάδων (bio__*, veh__*, face__*) και την ποιότητα κάθε κλάδου (q__*).
    Το info είναι λεξικό όπου κρατάμε διαγνωστικές πληροφορίες για εκτύπωση."""
    sig = session_signals(root, user, session)
    rows = []

    # EAR για κάθε frame του βίντεο (υπολογίζεται μία φορά για όλο το session)
    ear, layout = None, None
    if sig["FL"] is not None:
        ear, layout = landmarks_to_ear(sig["FL"])
        info["fl_layout"] = layout
    # ποιες στήλες τηλεμετρίας θα χρησιμοποιηθούν ως χαρακτηριστικά (όχι οι στήλες χρόνου)
    tele_cols = None
    if sig["Telemetry"] is not None:
        t = sig["Telemetry"]
        named = not all(str(c).startswith("col_") for c in t.columns)   # έχει πραγματικά ονόματα στηλών;
        tele_cols = [c for c in t.columns if "time" not in str(c).lower()] if named else list(t.columns[4:])
        info["tele_cols"] = tele_cols

    # διαγνωστικά διάρκειας
    for name in ("HR", "Telemetry", "FL"):
        if sig[name] is not None:
            dur = len(sig[name]) / RATES[name]                # διάρκεια = γραμμές / συχνότητα
            if abs(dur - SESSION_S) > 180:                    # απόκλιση > 3 λεπτά -> προειδοποίηση
                print(f"  [!] {user}_{session}: διάρκεια {name} = {dur:.0f}s (αναμενόταν ~{SESSION_S}s)")

    n_int = len(labels)                                       # πλήθος 4λεπτων διαστημάτων (συνήθως 10)
    for i in range(n_int):
        cls = kss_to_class(labels[i])
        if np.isnan(cls):                                     # διάστημα χωρίς ετικέτα -> παράλειψη
            continue
        for w in range(INTERVAL_S // window_s):               # π.χ. 240/30 = 8 παράθυρα ανά διάστημα
            t0 = i * INTERVAL_S + w * window_s                # αρχή παραθύρου (δευτερόλεπτα από την αρχή)
            t1 = t0 + window_s                                # τέλος παραθύρου
            f = {"subject": user, "session": session, "interval": i, "window": w, "y": int(cls)}

            # --- Κλάδος 1: βιομετρικά
            bio = {}
            valid = []                                        # ποσοστό έγκυρων δειγμάτων κάθε σήματος
            for name in ("HR", "EDA", "TEMP", "BVP"):
                if sig[name] is not None:
                    x = sl(sig[name].iloc[:, 0].to_numpy(float), RATES[name], t0, t1)   # 1η στήλη, κομμάτι παραθύρου
                    bio.update(stats(x, f"bio__{name}")); valid.append(valid_frac(x))
            if sig["ACC"] is not None:
                a = sig["ACC"].iloc[:, :3].to_numpy(float)    # άξονες x, y, z
                mag = np.linalg.norm(a, axis=1)               # μέτρο επιτάχυνσης sqrt(x²+y²+z²)
                bio.update(stats(sl(mag, RATES["ACC"], t0, t1), "bio__ACC"))
            if sig["O2M"] is not None and sig["O2M"].shape[1] >= 3:
                for j, nm in enumerate(["SpO2", "PR", "Motion"]):   # κορεσμός οξυγόνου, παλμοί, κίνηση
                    x = sl(sig["O2M"].iloc[:, j].to_numpy(float), RATES["O2M"], t0, t1)
                    bio.update(stats(x, f"bio__{nm}"))
            f["q__bio"] = float(np.mean(valid)) if valid else 0.0       # ποιότητα σήματος: ποσοστό έγκυρων δειγμάτων
            if not valid or np.mean(valid) < MIN_VALID_FRAC:            # πολύ κακή ποιότητα -> ο κλάδος θεωρείται ότι λείπει
                bio = {k: np.nan for k in bio}
            f.update(bio)

            # --- Κλάδος 2: τηλεμετρία + πίεση λαβής
            veh = {}
            valid = []
            if sig["Telemetry"] is not None:
                for c in tele_cols:
                    x = sl(sig["Telemetry"][c].to_numpy(float), RATES["Telemetry"], t0, t1)
                    s = stats(x, f"veh__{c}")
                    veh.update({k: v for k, v in s.items() if k.endswith(("mean", "std"))})   # μόνο mean και std
                    # dstd = τυπ. απόκλιση των διαδοχικών διαφορών: πόσο "νευρικά" αλλάζει το σήμα
                    veh[f"veh__{c}__dstd"] = np.nanstd(np.diff(x)) if len(x) > 2 else np.nan
                    valid.append(valid_frac(x))
            for name in ("RGP", "LGP"):
                if sig[name] is not None:
                    x = sl(sig[name].iloc[:, 0].to_numpy(float), RATES[name], t0, t1)
                    s = stats(x, f"veh__{name}")
                    veh.update({k: v for k, v in s.items() if k.endswith(("mean", "std"))})
                    valid.append(valid_frac(x))
            if sig["RGP"] is not None and sig["LGP"] is not None:
                # ασυμμετρία λαβής: μέση απόλυτη διαφορά πίεσης δεξιού-αριστερού χεριού
                r = sl(sig["RGP"].iloc[:, 0].to_numpy(float), RATES["RGP"], t0, t1)
                l = sl(sig["LGP"].iloc[:, 0].to_numpy(float), RATES["LGP"], t0, t1)
                k = min(len(r), len(l))
                veh["veh__grip_asym"] = np.nanmean(np.abs(r[:k] - l[:k])) if k else np.nan
            f["q__veh"] = float(np.mean(valid)) if valid else 0.0
            if not valid or np.mean(valid) < MIN_VALID_FRAC:
                veh = {k: np.nan for k in veh}
            f.update(veh)

            # --- Κλάδος 3: πρόσωπο (EAR/PERCLOS από landmarks + μέσες τιμές FAU)
            face = {}
            qf = 0.0                                          # ποιότητα κλάδου προσώπου
            if ear is not None:
                x = sl(ear, RATES["FL"], t0, t1)
                qf = valid_frac(x)
                if valid_frac(x) >= MIN_VALID_FRAC:
                    face.update(stats(x, "face__EAR"))
                    xv = x[~np.isnan(x)]
                    closed = xv < EAR_CLOSED                  # True όπου το μάτι θεωρείται κλειστό
                    face["face__PERCLOS"] = float(closed.mean())   # ποσοστό χρόνου με κλειστά μάτια
                    # βλεφαρίσματα = πόσες φορές περνάμε από ανοιχτό (0) σε κλειστό (1)
                    face["face__blinks"] = int(np.sum(np.diff(closed.astype(int)) == 1))
                else:
                    face.update({f"face__EAR__{n}": np.nan for n in ["mean", "std", "min", "max", "slope"]})
                    face["face__PERCLOS"] = np.nan; face["face__blinks"] = np.nan
            if sig["FAU"] is not None:
                fau = sig["FAU"]
                fau = fau.iloc[:, 1:] if fau.shape[1] == 31 else fau    # αν υπάρχει στήλη frame, την αφαιρούμε
                seg = fau.iloc[int(t0 * RATES["FAU"]):int(t1 * RATES["FAU"])]   # κομμάτι παραθύρου
                if ear is None:                               # αν δεν υπάρχει FL, η ποιότητα κρίνεται από το FAU
                    qf = valid_frac(seg.iloc[:, 0].to_numpy(float))
                means = seg.mean(numeric_only=True)           # μέση τιμή κάθε action unit στο παράθυρο
                if valid_frac(seg.iloc[:, 0].to_numpy(float)) < MIN_VALID_FRAC:
                    means[:] = np.nan
                face.update({f"face__FAU_{c}": v for c, v in means.items()})   # π.χ. face__FAU_eye_closed
            f.update(face)
            f["q__face"] = qf
            rows.append(f)                                    # μία γραμμή ανά παράθυρο
    return rows


def build_dataset(root, labels_path, window_s):
    """Τρέχει τη build_session_features για όλα τα sessions και ενώνει τα αποτελέσματα σε ένα DataFrame."""
    index_files(root)
    labels = load_labels(labels_path)
    print(f"Labels: {len(labels)} sessions")
    all_rows, info = [], {}
    for (user, session), vals in sorted(labels.items()):
        print(f"Επεξεργασία {user}_{session} ...")
        all_rows += build_session_features(root, user, session, vals, window_s, info)
    df = pd.DataFrame(all_rows)
    # έλεγχος ότι βγήκε τουλάχιστον μία στήλη χαρακτηριστικών (αλλιώς μάλλον λάθος ονόματα αρχείων)
    if df.empty or not any(c.startswith(BRANCH_PREFIXES) for c in df.columns):
        raise SystemExit("Δεν εξήχθη κανένα χαρακτηριστικό. Πιθανώς τα ονόματα των αρχείων διαφέρουν από "
                         "<User>_<SIGNAL>_<Session>.csv. Παραδείγματα ονομάτων που βρέθηκαν: "
                         + ", ".join(list(_INDEX)[:8]))
    print(f"\nΔιαγνωστικά: διάταξη FL = {info.get('fl_layout')}, στήλες τηλεμετρίας = {info.get('tele_cols')}")
    print(f"Παράθυρα: {len(df)}, υποκείμενα: {df.subject.nunique()}, κατανομή τάξεων:\n"
          f"{df.y.value_counts().sort_index().rename(dict(enumerate(CLASS_NAMES)))}\n")
    return df


# ----------------------------------------------------------------------------- μοντέλα
def branch_model(trees):
    """Μοντέλο ενός κλάδου: συμπλήρωση ελλιπών με τη διάμεσο -> Random Forest.
    class_weight="balanced": οι σπάνιες τάξεις (π.χ. High) παίρνουν μεγαλύτερο βάρος."""
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("rf", RandomForestClassifier(n_estimators=trees, max_depth=8, min_samples_leaf=5,
                                                   class_weight="balanced", n_jobs=-1, random_state=SEED))])


def full_proba(model, X):
    """Πιθανότητες για ΟΛΕΣ τις 3 τάξεις. Αν στο train έλειπε κάποια τάξη, το μοντέλο βγάζει λιγότερες
    στήλες, οπότε τις τοποθετούμε στη σωστή θέση και οι υπόλοιπες μένουν 0."""
    p = model.predict_proba(X)
    classes = model.named_steps["rf"].classes_ if hasattr(model, "named_steps") else model.classes_
    out = np.zeros((len(X), N_CLASSES))
    out[:, classes.astype(int)] = p
    return out


def branch_oof(X, y, groups, avail, trees, n_splits=5, cv="loso"):
    """Προβλέψεις out-of-fold για τον κλάδο, ώστε ο meta-classifier να μην βλέπει leakage.
    LOSO: τα εσωτερικά folds χωρίζονται ανά οδηγό. 5-fold: ανά παράθυρο, όπως στο άρθρο."""
    # Γιατί OOF: αν ο meta-classifier εκπαιδευόταν σε πιθανότητες που έβγαλε ο κλάδος για δεδομένα που
    # ο ίδιος ο κλάδος είχε δει στην εκπαίδευση, θα ήταν υπερβολικά "σίγουρες" και ο meta θα μάθαινε λάθος.
    # Έτσι, κάθε γραμμή του train παίρνει πιθανότητα από μοντέλο που ΔΕΝ την είδε.
    P = np.full((len(y), N_CLASSES), np.nan)        # NaN = ο κλάδος δεν είναι διαθέσιμος σε αυτό το παράθυρο
    if cv == "loso":
        splits = GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(X, y, groups)
    else:
        splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(X, y)
    for tr, va in splits:
        tr, va = tr[avail[tr]], va[avail[va]]       # μόνο παράθυρα όπου ο κλάδος υπάρχει
        if len(tr) == 0 or len(va) == 0:
            continue
        P[va] = full_proba(branch_model(trees).fit(X[tr], y[tr]), X[va])
    return P


def meta_features(P_list, flags):
    """Είσοδος του απλού meta-classifier: [πιθανότητες bio | veh | face | δείκτες διαθεσιμότητας (0/1)]
    = 3*3 + 3 = 12 στήλες."""
    return np.hstack(P_list + [flags])


def mask_branch(P_list, flags, b):
    """Προσομοιώνει ότι ο κλάδος b λείπει: πιθανότητες -> NaN, δείκτης διαθεσιμότητας -> 0."""
    P2 = [p.copy() for p in P_list]
    f2 = flags.copy()
    P2[b][:] = np.nan
    f2[:, b] = 0
    return P2, f2


def augment_dropout(P_list, flags, y, rng, p=0.2, copies=2):
    """(Παλαιότερη εκδοχή του dropout_copies, δεν καλείται πουθενά στο script.)
    Δημιουργεί αντίγραφα του train όπου κάθε κλάδος χάνεται τυχαία με πιθανότητα p."""
    Xs, ys = [meta_features(P_list, flags)], [y]
    for _ in range(copies):
        P2 = [p_.copy() for p_ in P_list]
        f2 = flags.copy()
        for b in range(len(P2)):
            drop = rng.random(len(y)) < p
            P2[b][drop] = np.nan
            f2[drop, b] = 0
        Xs.append(meta_features(P2, f2)); ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def to_zero_input(M):
    """Από [πιθανότητες, δείκτες] κρατά μόνο τις πιθανότητες και βάζει 0 όπου λείπει ο κλάδος (Zero Imputation)."""
    return np.nan_to_num(M[:, :len(BRANCHES) * N_CLASSES], nan=0.0)


def dropout_copies(P_list, flags, Q, rng, p=0.2, copies=2):
    """Το αρχικό σύνολο + αντίγραφα όπου κάθε κλάδος «χάνεται» τυχαία με πιθανότητα p (modality dropout).
    Ο κλάδος που χάνεται: πιθανότητες = NaN, δείκτης διαθεσιμότητας = 0, ποιότητα σήματος = 0."""
    # Ιδέα: ο meta-classifier "εξασκείται" στην εκπαίδευση σε περιπτώσεις που λείπει αισθητήρας,
    # ώστε στο test να μην "χάνεται" όταν συμβεί πραγματικά. Με copies=2 το train τριπλασιάζεται.
    out = [(P_list, flags, Q)]
    for _ in range(copies):
        P2 = [p_.copy() for p_ in P_list]
        f2, Q2 = flags.copy(), Q.copy()
        for b in range(len(P2)):
            drop = rng.random(len(flags)) < p     # τυχαία επιλογή παραθύρων όπου ο κλάδος b "χάνεται"
            P2[b][drop] = np.nan
            f2[drop, b] = 0
            Q2[drop, b] = 0.0
        out.append((P2, f2, Q2))
    return out


def reliability_features(P_list, flags, Q):
    """Είσοδος του meta-classifier με επίγνωση αξιοπιστίας: πιθανότητες κλάδων + διαθεσιμότητα
    + ποιότητα σήματος (ποσοστό έγκυρων δειγμάτων στο παράθυρο) + βεβαιότητα κάθε κλάδου (μέγιστη πιθανότητα)."""
    # Σύνολο στηλών: 9 πιθανότητες + 3 διαθεσιμότητα + 3 ποιότητα + 3 βεβαιότητα = 18.
    # Αυτή είναι η είσοδος της ΠΡΟΤΕΙΝΟΜΕΝΗΣ μεθόδου.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conf = np.stack([np.nanmax(P, axis=1) for P in P_list], axis=1)   # max πιθανότητα ανά κλάδο
    return np.hstack(P_list + [flags, Q, conf])


def branch_columns(b):
    """Στήλες του reliability_features που ανήκουν στον κλάδο b (για τη σημαντικότητα κλάδων)."""
    nb = len(BRANCHES)
    cols = list(range(b * N_CLASSES, (b + 1) * N_CLASSES))   # οι 3 πιθανότητες του κλάδου
    base = nb * N_CLASSES                                     # = 9, εκεί ξεκινούν οι υπόλοιπες στήλες
    return cols + [base + b, base + nb + b, base + 2 * nb + b]   # + διαθεσιμότητα, ποιότητα, βεβαιότητα


def fit_meta(X, y):
    """Εκπαιδεύει τον meta-classifier (gradient boosting δέντρων). Χειρίζεται τα NaN μόνος του,
    οπότε δεν χρειάζεται imputer. Τα βάρη "balanced" ισορροπούν τις τάξεις."""
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, early_stopping=False,
                                       random_state=SEED)
    m.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return m


GENERIC = {"early_LogReg", "early_kNN", "early_GradBoost"}   # γενικοί ταξινομητές, μόνο με --all-baselines
ALL_BASELINES = False      # αλλάζει σε True αν δοθεί --all-baselines

# Ποια μέθοδος αντιστοιχεί σε ποια σχετική εργασία της βιβλιογραφίας
RELATED = {
    "early_paper_SVM":        ("Bodaghi et al. [1]", "Πρώιμη σύντηξη, MinMax + PCA, SVM (όπως στο άρθρο)"),
    "early_paper_RF":         ("Bodaghi et al. [1]", "Πρώιμη σύντηξη, MinMax + PCA, Random Forest (όπως στο άρθρο)"),
    "early_MLP":              ("Bodaghi et al. [1]", "Πρώιμη σύντηξη με νευρωνικό δίκτυο (απλοποιημένη εκδοχή)"),
    "early_fusion_RF":        ("Bodaghi et al. [1]", "Πρώιμη σύντηξη, Random Forest στα ίδια χαρακτηριστικά"),
    "late_soft_voting":       ("Das et al. [2]", "Όψιμη σύντηξη, μέσος όρος πιθανοτήτων"),
    "late_hard_voting":       ("Das et al. [2]", "Όψιμη σύντηξη, πλειοψηφία αποφάσεων"),
    "late_stacking_LR":       ("Κλασικό stacking", "Όψιμη σύντηξη, γραμμικός meta-classifier"),
    "late_meta":              ("Παραλλαγή", "Δενδρικός meta-classifier χωρίς modality dropout"),
    "late_meta_dropout":      ("Krishna et al. [3]", "Δενδρικός meta-classifier + modality dropout"),
    "late_meta_zero_dropout": ("Αρχικό σχέδιο", "Zero imputation χωρίς δείκτες διαθεσιμότητας"),
    "unimodal_bio":           ("Kontras et al. [4]", "Μόνο ο κλάδος βιομετρικών"),
    "unimodal_veh":           ("Kontras et al. [4]", "Μόνο ο κλάδος οχήματος"),
    "unimodal_face":          ("Kontras et al. [4]", "Μόνο ο κλάδος προσώπου"),
}


def early_baselines(trees):
    """Μέθοδοι πρώιμης σύντηξης για σύγκριση (όλες στα ίδια χαρακτηριστικά, ό,τι λείπει -> διάμεσος).
    paper_SVM / paper_RF: αναπαραγωγή της διαδικασίας των Bodaghi et al. (MinMax, PCA 90%, ίδιες υπερπαράμετροι)."""
    # "Πρώιμη σύντηξη" = όλα τα χαρακτηριστικά όλων των κλάδων σε ΕΝΑ διάνυσμα -> ΕΝΑ μοντέλο.
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "chance_stratified": DummyClassifier(strategy="stratified", random_state=SEED),
        # PCA(0.90) = κρατά όσες συνιστώσες εξηγούν το 90% της διακύμανσης
        "early_paper_SVM": Pipeline([("imp", imp()), ("mm", MinMaxScaler()), ("pca", PCA(0.90, random_state=SEED)),
                                     ("svm", SVC(kernel="rbf", C=0.1, gamma="auto", class_weight="balanced"))]),
        "early_paper_RF": Pipeline([("imp", imp()), ("mm", MinMaxScaler()), ("pca", PCA(0.90, random_state=SEED)),
                                    ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10,
                                                                  n_jobs=-1, random_state=SEED))]),
        "early_LogReg": Pipeline([("imp", imp()), ("sc", StandardScaler()),
                                  ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "early_kNN": Pipeline([("imp", imp()), ("sc", StandardScaler()), ("knn", KNeighborsClassifier(15))]),
        "early_GradBoost": HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05,
                                                          early_stopping=False, class_weight="balanced",
                                                          random_state=SEED),
        "early_MLP": Pipeline([("imp", imp()), ("sc", StandardScaler()),
                               ("mlp", MLPClassifier((64, 32), alpha=1e-3, max_iter=300, early_stopping=True,
                                                     random_state=SEED))]),
    }


def hard_vote(P_list):
    """Πλειοψηφία των αποφάσεων των διαθέσιμων κλάδων· σε ισοπαλία αποφασίζει ο μέσος όρος πιθανοτήτων."""
    # votes: για κάθε παράθυρο η πρόβλεψη κάθε κλάδου, ή -1 αν ο κλάδος λείπει
    votes = np.stack([np.where(np.isnan(P[:, 0]), -1, np.nan_to_num(P).argmax(1)) for P in P_list], axis=1)
    soft = soft_vote(P_list)
    out = np.empty(len(votes), dtype=int)
    for i, v in enumerate(votes):
        v = v[v >= 0]                                  # μόνο οι διαθέσιμοι κλάδοι
        if len(v) == 0:
            out[i] = soft[i]; continue
        c = np.bincount(v, minlength=N_CLASSES)        # πόσες ψήφους πήρε κάθε τάξη
        out[i] = c.argmax() if (c == c.max()).sum() == 1 else soft[i]   # ισοπαλία -> soft voting
    return out


def soft_vote(P_list):
    """Μέσος όρος των πιθανοτήτων των διαθέσιμων κλάδων, επιλογή της τάξης με τη μεγαλύτερη."""
    stack = np.stack(P_list)                      # (3, n, C)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        avg = np.nanmean(stack, axis=0)           # nanmean: αγνοεί τους κλάδους που λείπουν
    avg[np.isnan(avg).all(axis=1)] = 1.0 / N_CLASSES   # αν λείπουν όλοι -> ίσες πιθανότητες
    return avg.argmax(1)


# ----------------------------------------------------------------------------- LOSO
PROPOSED = "late_meta_reliability"   # η προτεινόμενη μέθοδος, με την οποία συγκρίνονται όλες οι άλλες


def run_loso(df, trees, cv="loso"):
    """cv="loso": κάθε φορά ένας οδηγός στο test (νέος οδηγός).
    cv="kfold": stratified 5-fold σε όλα τα παράθυρα, όπως στο άρθρο του UL-DD (ο ίδιος οδηγός και στα δύο)."""
    # ---- προετοιμασία πινάκων
    feat = {b: [c for c in df.columns if c.startswith(f"{b}__")] for b in BRANCHES}   # στήλες ανά κλάδο
    X = {b: df[feat[b]].to_numpy(float) for b in BRANCHES}                            # πίνακας χαρακτηριστικών ανά κλάδο
    avail = {b: ~np.isnan(X[b]).all(axis=1) for b in BRANCHES}                        # True αν ο κλάδος υπάρχει στο παράθυρο
    y = df.y.to_numpy(int)                       # ετικέτες
    groups = df.subject.to_numpy()               # ποιος οδηγός είναι κάθε παράθυρο
    all_cols = [c for b in BRANCHES for c in feat[b]]
    Xall = df[all_cols].to_numpy(float)          # όλα τα χαρακτηριστικά μαζί (για early fusion)
    rng = np.random.default_rng(SEED)
    scenarios = ["all"] + [f"no_{b}" for b in BRANCHES]   # all, no_bio, no_veh, no_face
    records = []                                   # (method, scenario, idx, y_true, y_pred)

    def log(method, scen, idx, pred):
        """Καταγράφει τις προβλέψεις μιας μεθόδου σε ένα σενάριο (μία εγγραφή ανά παράθυρο)."""
        for i, p in zip(idx, pred):
            records.append((method, scen, int(i), int(y[i]), int(p)))

    timing = []                                    # (μέθοδος, ms ανά παράθυρο)
    importance = []                                # (κλάδος, πτώση macro F1 όταν «ανακατεύεται» ο κλάδος)
    conflict_idx = []                              # παράθυρα όπου οι διαθέσιμοι κλάδοι διαφωνούν
    sizes = {}
    qcols = [f"q__{b}" for b in BRANCHES]
    if all(c in df.columns for c in qcols):
        Qall = df[qcols].fillna(0.0).to_numpy(float)     # ποιότητα σήματος κάθε κλάδου ανά παράθυρο
    else:                                          # παλιό features.csv χωρίς δείκτες ποιότητας
        print("  [!] Δεν βρέθηκαν δείκτες ποιότητας σήματος (q__*) - τρέξτε με --rebuild. Χρήση μόνο διαθεσιμότητας.")
        Qall = np.stack([avail[b] for b in BRANCHES], axis=1).astype(float)

    # ---- ορισμός των folds (train/test διαχωρισμών)
    if cv == "loso":
        subjects = sorted(np.unique(groups))
        # για κάθε οδηγό s: train = όλοι οι άλλοι, test = ο s
        folds = [(f"test = {s}", np.where(groups != s)[0], np.where(groups == s)[0]) for s in subjects]
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        folds = [(f"5-fold #{i}", tr, te) for i, (tr, te) in enumerate(skf.split(Xall, y), 1)]

    # ---- κύριος βρόχος: για κάθε fold εκπαίδευση όλων των μεθόδων και πρόβλεψη στο test
    for k, (name, tr, te) in enumerate(folds, 1):
        print(f"{cv.upper()} fold {k}/{len(folds)}: {name}")
        t_branch = 0.0                             # συνολικός χρόνος πρόβλεψης των κλάδων

        # 1) κλάδοι: OOF στο train, μοντέλο στο train → πρόβλεψη στο test
        Ptr, Pte, branch_models = [], [], []       # πιθανότητες κλάδων στο train (OOF) και στο test
        for b in BRANCHES:
            Ptr.append(branch_oof(X[b][tr], y[tr], groups[tr], avail[b][tr], trees, cv=cv))
            p = np.full((len(te), N_CLASSES), np.nan)
            ok = avail[b][te]
            if avail[b][tr].sum() < 10:                  # ο κλάδος σχεδόν δεν υπάρχει στο train -> θεωρείται ελλιπής
                Pte.append(p)
                continue
            # τελικό μοντέλο κλάδου σε όλο το train (μόνο όπου ο κλάδος υπάρχει)
            m = branch_model(trees).fit(X[b][tr][avail[b][tr]], y[tr][avail[b][tr]])
            branch_models.append(m)
            if ok.any():
                t0 = time.perf_counter()
                p[ok] = full_proba(m, X[b][te][ok])
                t_branch += time.perf_counter() - t0
            Pte.append(p)
            # unimodal baseline (μόνο σε διαθέσιμα παράθυρα)
            if ok.any():
                log(f"unimodal_{b}", "all", te[ok], p[ok].argmax(1))
        # δείκτες διαθεσιμότητας: 1 αν ο κλάδος έδωσε πιθανότητα, 0 αν όχι
        flags_tr = np.stack([~np.isnan(P[:, 0]) for P in Ptr], axis=1).astype(int)
        flags_te = np.stack([~np.isnan(P[:, 0]) for P in Pte], axis=1).astype(int)

        # 2) meta-classifiers
        Qtr, Qte = Qall[tr] * flags_tr, Qall[te] * flags_te   # ποιότητα = 0 όπου ο κλάδος λείπει
        # παράθυρα σύγκρουσης: τουλάχιστον δύο διαθέσιμοι κλάδοι με διαφορετική πρόβλεψη
        votes = np.stack([np.where(np.isnan(P[:, 0]), -1, np.nan_to_num(P).argmax(1)) for P in Pte], axis=1)
        conflict_idx.extend(te[[len(set(v[v >= 0])) >= 2 for v in votes]].tolist())

        # (α) late_meta: απλός meta-classifier, χωρίς dropout
        meta_plain = fit_meta(meta_features(Ptr, flags_tr), y[tr])
        # (β) late_meta_dropout: ίδιος, αλλά εκπαιδευμένος και σε αντίγραφα με τυχαία "χαμένους" κλάδους
        copies = dropout_copies(Ptr, flags_tr, Qtr, rng)
        yd = np.concatenate([y[tr]] * len(copies))      # οι ετικέτες επαναλαμβάνονται για κάθε αντίγραφο
        Xd = np.vstack([meta_features(P, f) for P, f, _ in copies])
        meta_drop = fit_meta(Xd, yd)
        # η προτεινόμενη: + ποιότητα σήματος και βεβαιότητα κάθε κλάδου («δυναμική στάθμιση αξιοπιστίας»)
        Xr = np.vstack([reliability_features(P, f, Q) for P, f, Q in copies])
        meta_rel = fit_meta(Xr, yd)

        # σημαντικότητα κλάδων: πόσο πέφτει το F1 όταν ανακατεύονται τυχαία οι είσοδοι ενός κλάδου
        # (permutation importance: αν ανακατέψουμε έναν κλάδο και το F1 πέσει πολύ, ο κλάδος είναι σημαντικός)
        Xte_rel = reliability_features(Pte, flags_te, Qte)
        base = subject_f1(y[te], meta_rel.predict(Xte_rel))
        for b in range(len(BRANCHES)):
            Xp = Xte_rel.copy()
            cols = branch_columns(b)
            Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]   # ανακάτεμα γραμμών μόνο στις στήλες του κλάδου
            importance.append((BRANCHES[b], base - subject_f1(y[te], meta_rel.predict(Xp))))
        # έκδοση όπως περιγράφεται στη διπλωματική: ο κλάδος που λείπει γίνεται 0, χωρίς δείκτες διαθεσιμότητας
        # (ίδιο modality dropout με το meta_drop, ώστε η σύγκριση να είναι δίκαιη)
        meta_zero = fit_meta(to_zero_input(Xd), yd)

        # 3) early fusion (RF πάνω σε όλα τα χαρακτηριστικά)
        early = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("rf", RandomForestClassifier(n_estimators=trees, max_depth=8, min_samples_leaf=5,
                                                        class_weight="balanced", n_jobs=-1, random_state=SEED))])
        early.fit(Xall[tr], y[tr])
        # επιπλέον baselines πρώιμης σύντηξης (όλα στα ίδια χαρακτηριστικά)
        early_models = {"early_fusion_RF": early}
        for name, model in early_baselines(trees).items():
            if name in GENERIC and not ALL_BASELINES:     # οι γενικοί τρέχουν μόνο με --all-baselines
                continue
            early_models[name] = model.fit(Xall[tr], y[tr])
        # επιπλέον baselines όψιμης σύντηξης
        # (η Logistic Regression δεν δέχεται NaN, οπότε οι κλάδοι που λείπουν γίνονται 1/3 = "δεν ξέρω")
        Xs_tr = np.nan_to_num(meta_features(Ptr, flags_tr), nan=1.0 / N_CLASSES)
        stack_lr = Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(Xs_tr, y[tr])

        # 3β) ablation (Άξονας 1): meta-classifier μόνο με δύο από τους τρεις κλάδους
        for combo in itertools.combinations(range(len(BRANCHES)), 2):   # (bio,veh), (bio,face), (veh,face)
            cols = [c for b in combo for c in range(b * N_CLASSES, (b + 1) * N_CLASSES)]   # πιθανότητες
            cols += [len(BRANCHES) * N_CLASSES + b for b in combo]                         # + δείκτες
            m_ab = fit_meta(meta_features(Ptr, flags_tr)[:, cols], y[tr])
            name_ab = "ablation_" + "+".join(BRANCHES[b] for b in combo)
            log(name_ab, "all", te, m_ab.predict(meta_features(Pte, flags_te)[:, cols]))

        # 3γ) χρόνος πρόβλεψης ανά παράθυρο (ms)
        t0 = time.perf_counter(); meta_rel.predict(reliability_features(Pte, flags_te, Qte))
        timing.append((PROPOSED, 1000 * (t_branch + time.perf_counter() - t0) / len(te)))   # κλάδοι + meta
        for name, model in early_models.items():
            t0 = time.perf_counter(); model.predict(Xall[te])
            timing.append((name, 1000 * (time.perf_counter() - t0) / len(te)))
        if k == len(folds):                        # μέγεθος των μοντέλων (KB) στο τελευταίο fold
            import pickle                          # pickle = σειριοποίηση μοντέλου σε bytes, για μέτρηση μεγέθους
            sizes = {PROPOSED: (sum(len(pickle.dumps(m)) for m in branch_models) + len(pickle.dumps(meta_rel))) / 1024,
                     PROPOSED + " (μόνο meta-classifier)": len(pickle.dumps(meta_rel)) / 1024}
            sizes.update({n: len(pickle.dumps(m)) / 1024 for n, m in early_models.items()})

        # 4) σενάρια ελλιπών τροπικοτήτων στο test
        for scen in scenarios:
            P_s, f_s, Xe, Q_s = Pte, flags_te, Xall[te].copy(), Qte
            if scen != "all":
                b = BRANCHES.index(scen[3:])                       # "no_face"[3:] -> "face"
                P_s, f_s = mask_branch(Pte, flags_te, b)          # για τις late μεθόδους
                Q_s = Qte.copy(); Q_s[:, b] = 0.0
                Xe[:, [all_cols.index(c) for c in feat[scen[3:]]]] = np.nan   # για τις early: στήλες κλάδου -> NaN
            # πρόβλεψη κάθε μεθόδου στο σενάριο και καταγραφή
            for name, model in early_models.items():
                log(name, scen, te, model.predict(Xe))
            log("late_soft_voting", scen, te, soft_vote(P_s))
            log("late_hard_voting", scen, te, hard_vote(P_s))
            log("late_stacking_LR", scen, te,
                stack_lr.predict(np.nan_to_num(meta_features(P_s, f_s), nan=1.0 / N_CLASSES)))
            log("late_meta", scen, te, meta_plain.predict(meta_features(P_s, f_s)))
            log("late_meta_dropout", scen, te, meta_drop.predict(meta_features(P_s, f_s)))
            log("late_meta_zero_dropout", scen, te, meta_zero.predict(to_zero_input(meta_features(P_s, f_s))))
            log(PROPOSED, scen, te, meta_rel.predict(reliability_features(P_s, f_s, Q_s)))

    # ---- συγκέντρωση αποτελεσμάτων όλων των folds
    res = pd.DataFrame(records, columns=["method", "scenario", "idx", "y_true", "y_pred"])
    # υποσύνολο σύγκρουσης: οι ίδιες προβλέψεις, μόνο στα παράθυρα όπου οι κλάδοι διαφωνούν
    conf = res[(res.scenario == "all") & res.idx.isin(set(conflict_idx))].copy()
    conf["scenario"] = "conflict"
    res = pd.concat([res, conf], ignore_index=True)
    tim = pd.DataFrame(timing, columns=["method", "ms_per_window"]).groupby("method").ms_per_window.mean()
    imp = (pd.DataFrame(importance, columns=["branch", "F1_drop"]).groupby("branch").F1_drop
           .agg(["mean", "std"]).sort_values("mean", ascending=False))
    extras = {"importance": imp, "sizes_kb": pd.Series(sizes, name="KB"),
              "conflict_share": len(set(conflict_idx)) / len(y)}
    return res, groups, tim, extras


def subject_f1(yt, yp):
    """Macro F1 ενός οδηγού, μόνο στις τάξεις που εμφανίζονται στις ετικέτες του."""
    # (αν ένας οδηγός δεν είχε ποτέ "High", δεν τον "τιμωρούμε" με F1=0 για αυτή την τάξη)
    return f1_score(yt, yp, labels=np.unique(yt), average="macro", zero_division=0)


def bootstrap_ci(yt, yp, subj, n_boot=1000):
    """95% διάστημα εμπιστοσύνης του macro F1 με επαναδειγματοληψία οδηγών."""
    # 1000 φορές: διαλέγουμε τυχαία οδηγούς με επανάθεση, υπολογίζουμε F1. Το 2.5% και 97.5% εκατοστημόριο
    # των 1000 τιμών δίνουν το διάστημα. Δειγματοληψία ανά ΟΔΗΓΟ (όχι ανά παράθυρο) γιατί τα παράθυρα
    # του ίδιου οδηγού δεν είναι ανεξάρτητα.
    rng = np.random.default_rng(SEED)
    us = np.unique(subj)
    idx = {s: np.where(subj == s)[0] for s in us}
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[s] for s in rng.choice(us, size=len(us), replace=True)])
        vals.append(f1_score(yt[pick], yp[pick], average="macro", labels=[0, 1, 2], zero_division=0))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def summarize(res, groups):
    """Υπολογίζει όλες τις μετρικές για κάθε (μέθοδο, σενάριο). Επιστρέφει:
    (1) συνολικό πίνακα και (2) πίνακα ανά οδηγό."""
    rows, per_subj = [], []
    for (m, sc), g in res.groupby(["method", "scenario"]):
        yt, yp = g.y_true.to_numpy(), g.y_pred.to_numpy()
        subj = groups[g.idx.to_numpy()]
        us = np.unique(subj)
        accs = [accuracy_score(yt[subj == s], yp[subj == s]) for s in us]   # ακρίβεια ανά οδηγό
        f1s = [subject_f1(yt[subj == s], yp[subj == s]) for s in us]        # F1 ανά οδηγό
        lo, hi = bootstrap_ci(yt, yp, subj) if sc == "all" else (np.nan, np.nan)   # CI μόνο στο "all" (χρόνος)
        rows.append({"method": m, "scenario": sc, "n_windows": len(g),
                     "accuracy": accuracy_score(yt, yp), "balanced_acc": balanced_accuracy_score(yt, yp),
                     "precision_macro": precision_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
                     "recall_macro": recall_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
                     "macro_F1": f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
                     "F1_CI95_low": lo, "F1_CI95_high": hi,
                     # recall_High = τι ποσοστό των πραγματικά νυσταγμένων παραθύρων εντοπίστηκε
                     "recall_High": recall_score(yt, yp, labels=[2], average="macro", zero_division=0),
                     # quadratic kappa: τιμωρεί περισσότερο το λάθος Alert<->High από το Alert<->Medium
                     "kappa_quadratic": cohen_kappa_score(yt, yp, weights="quadratic"),
                     "subj_F1_mean": np.mean(f1s), "subj_F1_std": np.std(f1s),
                     "subj_acc_mean": np.mean(accs), "subj_acc_std": np.std(accs)})
        for s, a, f in zip(us, accs, f1s):
            per_subj.append({"method": m, "scenario": sc, "subject": s, "accuracy": a, "macro_F1": f})
    return pd.DataFrame(rows).sort_values(["scenario", "method"]), pd.DataFrame(per_subj)


def axis1_stats(per_subj):
    """Άξονας 1: η προτεινόμενη μέθοδος έναντι κάθε άλλης, ανά οδηγό (Wilcoxon, νίκες)."""
    # Wilcoxon signed-rank: μη παραμετρικός έλεγχος για ζευγαρωτές μετρήσεις (ίδιοι οδηγοί, δύο μέθοδοι).
    # p < 0.05 => η διαφορά είναι στατιστικά σημαντική.
    from scipy.stats import wilcoxon
    ps = per_subj[per_subj.scenario == "all"].pivot(index="subject", columns="method", values="macro_F1")
    rows = []
    for m in ps.columns:
        if m == PROPOSED or PROPOSED not in ps.columns:
            continue
        both = ps[[PROPOSED, m]].dropna()
        d = both[PROPOSED] - both[m]               # διαφορά F1 ανά οδηγό (θετική = κερδίζει η προτεινόμενη)
        try:
            p = wilcoxon(d).pvalue if (d != 0).any() and len(d) >= 5 else np.nan
        except ValueError:
            p = np.nan
        rows.append({"baseline": m, "n_subjects": len(d), "mean_F1_proposed": both[PROPOSED].mean(),
                     "mean_F1_baseline": both[m].mean(), "mean_difference": d.mean(),
                     "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
                     "wilcoxon_p": p, "significant_p<0.05": bool(p < 0.05) if not np.isnan(p) else False})
    return pd.DataFrame(rows).sort_values("mean_difference", ascending=False)


# Οικογένεια μεθόδου ανάλογα με το πρόθεμα του ονόματος (για τη στήλη "family" του MAIN_TABLE)
FAMILY = {"chance": "Τυχαίο", "early": "Πρώιμη σύντηξη", "late": "Όψιμη σύντηξη",
          "unimodal": "Ένας κλάδος", "ablation": "Ablation (2 κλάδοι)"}


def main_table(summary, st, tim, sizes):
    """Ένας πίνακας με όλες τις τιμές ανά μέθοδο: απόδοση, ανθεκτικότητα, κόστος, σύγκριση με την προτεινόμενη."""
    a = summary[summary.scenario == "all"].set_index("method")
    t = a[["accuracy", "precision_macro", "recall_macro", "macro_F1", "F1_CI95_low", "F1_CI95_high", "recall_High", "kappa_quadratic", "balanced_acc",
           "subj_F1_mean", "subj_F1_std"]].copy()
    # F1 σε κάθε σενάριο ελλιπούς κλάδου, ως ξεχωριστή στήλη
    for sc in ["no_bio", "no_veh", "no_face"]:
        s = summary[summary.scenario == sc].set_index("method").macro_F1
        t[f"F1_{sc}"] = s
    miss = [c for c in ["F1_no_bio", "F1_no_veh", "F1_no_face"] if c in t]
    if miss:
        t["F1_missing_mean"] = t[miss].mean(axis=1)
        t["retention_%"] = 100 * t["F1_missing_mean"] / t["macro_F1"]   # % της απόδοσης που διατηρείται
    c = summary[summary.scenario == "conflict"].set_index("method").macro_F1
    t["F1_conflict"] = c
    t["ms_per_window"] = tim
    t["model_KB"] = sizes
    s2 = st.set_index("baseline")
    t["vs_proposed_wins/ties/losses"] = (s2.wins.astype(str) + "/" + s2.ties.astype(str) + "/" + s2.losses.astype(str))
    t["vs_proposed_p"] = s2.wilcoxon_p
    t.insert(0, "family", [FAMILY.get(m.split("_")[0], "Προτεινόμενη") if m != PROPOSED else "ΠΡΟΤΕΙΝΟΜΕΝΗ"
                           for m in t.index])
    return t.sort_values("macro_F1", ascending=False)


def importance_plot(imp, path):
    """Οριζόντιο ραβδόγραμμα με τη συνεισφορά κάθε κλάδου (αποθηκεύεται ως PNG)."""
    try:
        import matplotlib
        matplotlib.use("Agg")          # σχεδίαση χωρίς παράθυρο, απευθείας σε αρχείο
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = {"bio": "Βιομετρικά", "veh": "Όχημα (λαβή, τηλεμετρία)", "face": "Πρόσωπο"}
    s = imp.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh([names.get(b, b) for b in s.index], s["mean"], xerr=s["std"].fillna(0), color="#2F6690", capsize=3)
    ax.set_xlabel("Πτώση macro F1 όταν αφαιρείται η πληροφορία του κλάδου")
    ax.set_title("Συνεισφορά κάθε κλάδου στην απόφαση")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def axis1_plot(summary, path):
    """Ραβδόγραμμα macro F1 όλων των μεθόδων με τα 95% CI· η προτεινόμενη χρωματίζεται πορτοκαλί."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (το matplotlib δεν είναι εγκατεστημένο, παραλείπεται το γράφημα: pip install matplotlib)")
        return
    s = summary[summary.scenario == "all"].sort_values("macro_F1")
    err = np.vstack([s.macro_F1 - s.F1_CI95_low, s.F1_CI95_high - s.macro_F1])   # μήκη "μουστακιών" σφάλματος
    colors = ["#B4531A" if m == PROPOSED else "#8FA3BF" for m in s.method]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(s) + 1.5))
    ax.barh(s.method, s.macro_F1, xerr=err, color=colors, capsize=3)
    ax.axvline(1 / 3, color="#555555", ls="--", lw=1)                 # γραμμή τυχαίας απόδοσης (1/3)
    ax.text(1 / 3, len(s) - 0.4, " τυχαία", color="#555555", fontsize=9)
    ax.set_xlabel("Macro F1 (95% CI, bootstrap ανά οδηγό)")
    ax.set_title("Άξονας 1: σύγκριση μεθόδων σύντηξης")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    # ---- παράμετροι γραμμής εντολών
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, nargs="+",
                    help="ένας ή περισσότεροι φάκελοι με τα CSV του UL-DD (ψάχνει και στους υποφακέλους)")
    ap.add_argument("--labels", required=True, help="διαδρομή του Labels.csv")
    ap.add_argument("--window", type=int, default=30, help="μήκος παραθύρου σε δευτερόλεπτα (διαιρέτης του 240)")
    ap.add_argument("--trees", type=int, default=150)          # πλήθος δέντρων στα Random Forest
    ap.add_argument("--cache", default="features.csv")         # αρχείο όπου αποθηκεύονται τα χαρακτηριστικά
    ap.add_argument("--rebuild", action="store_true", help="ξαναϋπολογισμός χαρακτηριστικών")
    ap.add_argument("--out", default="results")                # φάκελος αποτελεσμάτων
    ap.add_argument("--cv", choices=["loso", "kfold", "both"], default="both",
                    help="loso = νέος οδηγός, kfold = stratified 5-fold όπως στο άρθρο, both = και τα δύο")
    ap.add_argument("--all-baselines", action="store_true",
                    help="τρέχει και γενικούς ταξινομητές (LogReg, kNN, GradBoost) εκτός από τις μεθόδους των σχετικών εργασιών")
    a = ap.parse_args()
    global ALL_BASELINES
    ALL_BASELINES = a.all_baselines
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # π.χ. μέσος όρος παραθύρου χωρίς έγκυρες τιμές
    assert INTERVAL_S % a.window == 0, "το --window πρέπει να διαιρεί το 240"

    # ---- Βήμα Α: χαρακτηριστικά (από cache αν υπάρχει, αλλιώς υπολογισμός από τα CSV)
    if Path(a.cache).exists() and not a.rebuild:
        print(f"Φόρτωση χαρακτηριστικών από {a.cache} (χρησιμοποιήστε --rebuild για ανανέωση)")
        df = pd.read_csv(a.cache, dtype={"subject": str, "session": str})
    else:
        df = build_dataset(a.root, a.labels, a.window)
        df.to_csv(a.cache, index=False)

    # τιμές άπειρες ή παράλογα μεγάλες (π.χ. διαίρεση με 0 σε κάποιο αρχείο) -> ελλιπείς
    num = df.select_dtypes("number").columns.drop(["interval", "window", "y"], errors="ignore")
    bad = ~np.isfinite(df[num].to_numpy(float)) & df[num].notna().to_numpy()      # ±άπειρο
    bad |= np.abs(np.nan_to_num(df[num].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)) > 1e9   # > 1 δισ.
    if bad.any():
        cols = num[bad.any(axis=0)].tolist()
        print(f"[!] {int(bad.sum())} μη έγκυρες τιμές (άπειρο/υπερβολικά μεγάλες) έγιναν ελλιπείς, στις στήλες: {cols[:6]}")
        vals = df[num].to_numpy(float)
        vals[bad] = np.nan
        df[num] = vals

    Path(a.out).mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    # ---- Βήματα Β και Γ: για κάθε τρόπο αξιολόγησης (loso / kfold)
    for cv in (["loso", "kfold"] if a.cv == "both" else [a.cv]):
        title = "leave-one-subject-out (νέος οδηγός)" if cv == "loso" else "stratified 5-fold (όπως στο άρθρο)"
        res, groups, tim, extras = run_loso(df, a.trees, cv=cv)      # εκπαίδευση + προβλέψεις
        summary, per_subj = summarize(res, groups)                   # μετρικές
        summary.to_csv(f"{a.out}/summary_{cv}.csv", index=False)
        per_subj.to_csv(f"{a.out}/per_subject_{cv}.csv", index=False)
        st = axis1_stats(per_subj)                                   # στατιστικοί έλεγχοι
        st.to_csv(f"{a.out}/axis1_stats_{cv}.csv", index=False)
        tim.to_csv(f"{a.out}/timing_{cv}.csv")
        axis1_plot(summary, f"{a.out}/axis1_macroF1_{cv}.png")

        # ---- εκτυπώσεις στην οθόνη
        cols = ["method", "accuracy", "precision_macro", "recall_macro", "macro_F1", "F1_CI95_low", "F1_CI95_high", "recall_High", "kappa_quadratic",
                "balanced_acc", "subj_F1_mean", "subj_F1_std"]
        print(f"\n=== Άξονας 1 · {title} · όλοι οι κλάδοι διαθέσιμοι ===")
        print(summary[summary.scenario == "all"].sort_values("macro_F1", ascending=False)[cols]
              .round(3).to_string(index=False))
        print(f"\n=== Άξονας 1 · στατιστικός έλεγχος: {PROPOSED} έναντι κάθε άλλης μεθόδου (ανά οδηγό) ===")
        print(st.round(4).to_string(index=False))
        print("\nΧρόνος πρόβλεψης (ms ανά παράθυρο 30 s):")
        print(tim.round(3).to_string())
        print("\nΜέγεθος μοντέλων (KB):")
        print(extras["sizes_kb"].round(1).to_string())
        extras["sizes_kb"].to_csv(f"{a.out}/model_size_{cv}.csv")

        print(f"\n=== Άξονας 1 · επίλυση συγκρούσεων: παράθυρα όπου οι κλάδοι διαφωνούν "
              f"({extras['conflict_share']:.0%} των παραθύρων) ===")
        print(summary[summary.scenario == "conflict"].sort_values("macro_F1", ascending=False)
              [["method", "n_windows", "macro_F1", "recall_High", "accuracy"]].round(3).to_string(index=False))

        print(f"\n=== Άξονας 1 · ερμηνευσιμότητα: σημαντικότητα κάθε κλάδου για τη μέθοδο {PROPOSED} ===")
        print("(πτώση του macro F1 όταν οι είσοδοι του κλάδου ανακατεύονται τυχαία· μεγαλύτερη = πιο σημαντικός)")
        print(extras["importance"].round(3).to_string())
        extras["importance"].to_csv(f"{a.out}/branch_importance_{cv}.csv")
        importance_plot(extras["importance"], f"{a.out}/branch_importance_{cv}.png")

        print(f"\n=== Άξονας 2 · {title} · σενάρια με αισθητήρα που λείπει ===")
        print(summary[summary.scenario.str.startswith("no_")][["method", "scenario", "macro_F1", "recall_High",
                                                                "accuracy"]].round(3).to_string(index=False))
        g = res[(res.method == PROPOSED) & (res.scenario == "all")]   # προβλέψεις της προτεινόμενης (για confusion matrix)
        mt = main_table(summary, st, tim, extras["sizes_kb"])
        mt.to_csv(f"{a.out}/MAIN_TABLE_{cv}.csv", encoding="utf-8-sig")   # utf-8-sig: σωστά ελληνικά στο Excel
        print(f"\n=== ΣΥΓΚΕΝΤΡΩΤΙΚΟΣ ΠΙΝΑΚΑΣ · {title} (αρχείο MAIN_TABLE_{cv}.csv) ===")
        show = ["family", "accuracy", "precision_macro", "recall_macro", "macro_F1", "recall_High", "kappa_quadratic", "F1_missing_mean",
                "retention_%", "ms_per_window", "vs_proposed_wins/ties/losses", "vs_proposed_p"]
        print(mt[[c for c in show if c in mt]].round(3).to_string())

        # πίνακας μόνο με τις μεθόδους των σχετικών εργασιών + την προτεινόμενη
        rel = mt[mt.index.isin(list(RELATED) + [PROPOSED, "chance_stratified"])].copy()
        rel.insert(0, "related_work", [RELATED.get(m, ("Η παρούσα εργασία" if m == PROPOSED else "Κάτω όριο", ""))[0]
                                       for m in rel.index])
        rel.insert(1, "description", [RELATED.get(m, ("", "Προτεινόμενη: meta-classifier + dropout + αξιοπιστία"
                                                      if m == PROPOSED else "Τυχαία πρόβλεψη"))[1] for m in rel.index])
        rel.to_csv(f"{a.out}/RELATED_TABLE_{cv}.csv", encoding="utf-8-sig")
        print(f"\n=== ΣΥΓΚΡΙΣΗ ΜΕ ΤΙΣ ΜΕΘΟΔΟΥΣ ΤΩΝ ΣΧΕΤΙΚΩΝ ΕΡΓΑΣΙΩΝ · {title} (αρχείο RELATED_TABLE_{cv}.csv) ===")
        print(rel[["related_work", "accuracy", "precision_macro", "recall_macro", "macro_F1", "recall_High", "F1_missing_mean",
                   "vs_proposed_wins/ties/losses", "vs_proposed_p"]].round(3).to_string())

        # confusion matrix: γραμμές = πραγματική τάξη, στήλες = προβλεπόμενη
        print(f"\nConfusion matrix, {PROPOSED}, όλες οι τροπικότητες (γραμμές=πραγματικό):")
        print(pd.DataFrame(confusion_matrix(g.y_true, g.y_pred, labels=[0, 1, 2]),
                           index=CLASS_NAMES, columns=CLASS_NAMES))
        print(f"\nΑρχεία στον φάκελο '{a.out}': summary_{cv}.csv, axis1_stats_{cv}.csv, per_subject_{cv}.csv, "
              f"timing_{cv}.csv, axis1_macroF1_{cv}.png")
    print("""
Μέθοδοι:
  unimodal_bio / _veh / _face : ένας κλάδος μόνος του (βιομετρικά / τηλεμετρία+λαβή / πρόσωπο)
  early_fusion_RF             : όλα τα χαρακτηριστικά μαζί σε ένα Random Forest (ό,τι λείπει -> διάμεσος)
  chance_stratified           : τυχαία πρόβλεψη με τις συχνότητες των τάξεων (κάτω όριο)
  early_paper_SVM / _RF       : αναπαραγωγή των Bodaghi et al.: MinMax, PCA 90%, SVM ή RF με τις υπερπαραμέτρους τους
  early_LogReg / _kNN         : απλοί ταξινομητές πρώιμης σύντηξης
  early_GradBoost             : δέντρα ενίσχυσης σε όλα τα χαρακτηριστικά (ίδια οικογένεια με τον meta-classifier)
  early_MLP                   : νευρωνικό δίκτυο (2 κρυφά επίπεδα) πρώιμης σύντηξης
  late_hard_voting            : πλειοψηφία των αποφάσεων των κλάδων
  late_stacking_LR            : κλασικό stacking: logistic regression πάνω στις πιθανότητες των κλάδων
  late_soft_voting            : μέσος όρος των πιθανοτήτων των κλάδων
  late_meta                   : meta-classifier (δέντρα ενίσχυσης) σε πιθανότητες + δείκτες διαθεσιμότητας
  late_meta_dropout           : όπως το late_meta, εκπαιδευμένο με modality dropout 20%
  late_meta_reliability       : Η ΠΡΟΤΕΙΝΟΜΕΝΗ: όπως το late_meta_dropout + ποιότητα σήματος και βεβαιότητα κάθε κλάδου
  late_meta_zero_dropout      : όπως στη διπλωματική: κλάδος που λείπει = 0, χωρίς δείκτες, με modality dropout
  ablation_X+Y                : meta-classifier μόνο με δύο κλάδους (δείχνει τι προσθέτει ο τρίτος)
Μετρικές: macro_F1 (κύρια, με 95% CI), recall_High (εντοπισμός νυσταγμένων), kappa_quadratic,
  subj_F1 = macro F1 ανά οδηγό. Στατιστικά: wins/losses = σε πόσους οδηγούς κέρδισε/έχασε η προτεινόμενη,
  wilcoxon_p < 0.05 = η διαφορά δεν είναι τυχαία.
Σενάρια: all = όλοι οι κλάδοι διαθέσιμοι, no_bio / no_veh / no_face = ο κλάδος αφαιρείται στο test,
  conflict = μόνο τα παράθυρα όπου οι διαθέσιμοι κλάδοι δίνουν διαφορετική πρόβλεψη.
MAIN_TABLE: F1_missing_mean = μέσο F1 όταν λείπει ένας κλάδος, retention_% = πόσο % της απόδοσης διατηρείται.

ΣΗΜΕΙΩΣΗ: οι ετικέτες είναι μία ανά 4 λεπτά, άρα τα παράθυρα του ίδιου διαστήματος δεν είναι ανεξάρτητα.
Το άρθρο του UL-DD χρησιμοποιεί stratified 5-fold (όχι subject-independent), άρα οι αριθμοί δεν είναι
άμεσα συγκρίσιμοι με το 88,03%.""")


# Εκτελείται μόνο όταν τρέχουμε το αρχείο απευθείας (όχι όταν γίνεται import από άλλο script)
if __name__ == "__main__":
    main()
