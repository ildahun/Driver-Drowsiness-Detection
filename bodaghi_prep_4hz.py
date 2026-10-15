"""
bodaghi_prep_4hz.py — Η προεπεξεργασία των Bodaghi et al. (UL-DD) σε ένα script.

Αναπαράγει πιστά τα notebooks τους:
  1.Combine_Data_and_Change_Frequency      → ένωση οδηγών, αλλαγή συχνότητας σε 4 Hz
  2_1 / 2_2 / 2_3 Concatenate_and_Labeling → συνένωση ανά τροπικότητα + ετικέτες KSS
  3_1 / 3_2 / 3_3 (μόνο τα κελιά που σώζουν) → 4HZ_Individual_Data/final_*_4HZ.csv

Ίδιοι κανόνες με τον κώδικά τους (και οι ιδιαιτερότητές του κρατήθηκαν σκόπιμα):
  - ACC 32→4 Hz, BVP 64→4, Telemetry/FAU/PL/FL 60→4: ένα δείγμα κάθε k
  - HR 1→4 Hz, O2M 0,5→4 Hz: επανάληψη γραμμών
  - LGP/RGP 3→4 Hz: σε κάθε τριάδα επαναλαμβάνεται η τελευταία γραμμή
  - EDA/TEMP είναι ήδη 4 Hz
  - Telemetry του A_Alert λείπει → 144000 γραμμές NaN, μετά «μεταφορά» από άλλη οδήγηση (3_2)
  - Τα κενά frames του βίντεο γεμίζουν με NaN, μετά γραμμική παρεμβολή ανά οδηγό/οδήγηση
  - Συνένωση των αρχείων κάθε τροπικότητας «κατά στήλες» με τη σειρά των γραμμών

Διαβάζει κατευθείαν από τα zip (δεν χρειάζεται να αποσυμπιέσεις το Extracted_Features.zip
των 4,5 GB) ή από αποσυμπιεσμένους φακέλους.

Χρήση:
  python bodaghi_prep_4hz.py ^
      --csv "C:\\Users\\Ilda hunci\\ULDD\\CSV_Files" ^
      --features "C:\\Users\\Ilda hunci\\Downloads\\Extracted_Features.zip" ^
      --labels "C:\\...\\Labels.csv" ^
      --out "C:\\Users\\Ilda hunci\\ULDD\\4HZ_Individual_Data"
"""
import argparse, io, os, sys, time, zipfile, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

SESS = lambda fn: "Drowsy" if os.path.splitext(fn)[0].split("_")[-1].startswith("D") else "Alert"


# ---------------------------------------------------------------- πηγές αρχείων (φάκελος ή zip)
class Source:
    def __init__(self, path):
        self.path = path
        if os.path.isfile(path) and path.lower().endswith(".zip"):
            self.zf = zipfile.ZipFile(path)
            self.names = sorted(n for n in self.zf.namelist() if n.lower().endswith(".csv"))
        elif os.path.isdir(path):
            self.zf = None
            self.names = sorted(os.path.relpath(os.path.join(r, f), path)
                                for r, _, fs in os.walk(path) for f in fs if f.lower().endswith(".csv"))
        else:
            sys.exit(f"Δεν βρέθηκε: {path}")

    def find(self, tag):
        """αρχεία όπως στο notebook: '_TAG_D' ή '_TAG_A' στο όνομα, ταξινομημένα (σειρά os.walk στα Windows)"""
        out = []
        for n in self.names:
            fn = os.path.basename(n)
            if f"_{tag}_D" in fn or f"_{tag}_A" in fn:
                out.append((n, fn.split("_")[0], SESS(fn)))
        return out

    def read(self, name, **kw):
        if self.zf:
            with self.zf.open(name) as f:
                return pd.read_csv(io.BytesIO(f.read()), **kw)
        return pd.read_csv(os.path.join(self.path, name), **kw)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- notebook 1
def change_frequency(df, current_hz, target_hz):
    def resample_group(group):
        if target_hz < current_hz:
            return group.iloc[::int(current_hz / target_hz), :].reset_index(drop=True)
        elif target_hz > current_hz:
            return group.loc[group.index.repeat(int(target_hz / current_hz))].reset_index(drop=True)
        return group
    parts = [resample_group(g) for _, g in df.groupby(["user", "session"], sort=True)]
    return pd.concat(parts, ignore_index=True)


def combine(src, tag, names=None, **kw):
    files = src.find(tag)
    if not files:
        sys.exit(f"Δεν βρέθηκαν αρχεία {tag} στο {src.path}")
    dfs = []
    for n, user, sess in files:
        d = src.read(n, header=None, names=names) if names else src.read(n, low_memory=False)
        d["user"], d["session"] = user, sess
        dfs.append(d)
    log(f"  {tag}: {len(files)} αρχεία")
    return pd.concat(dfs, axis=0, ignore_index=True)


def grip_3to4(df):
    # όπως στο notebook: ομάδες των 3 γραμμών στο ενιαίο ευρετήριο + επανάληψη της τελευταίας
    n = len(df); idx = []
    for s in range(0, n, 3):
        e = min(s + 3, n)
        idx.extend(range(s, e)); idx.append(e - 1)
    return df.iloc[idx].reset_index(drop=True)


def video(src, tag, frame_col):
    """FAU / PL / FL: κενά frames → NaN, μετά 60→4 Hz, ανά (user, session)"""
    files = src.find(tag)
    if not files:
        sys.exit(f"Δεν βρέθηκαν αρχεία {tag} στο {src.path}")
    groups = {}
    for n, user, sess in files:
        groups.setdefault((user, sess), []).append(n)
    out = []
    for (user, sess) in sorted(groups):
        g = pd.concat([src.read(n, low_memory=False) for n in groups[(user, sess)]], ignore_index=True)
        g["user"], g["session"] = user, sess
        g = g.drop_duplicates(subset=frame_col, keep="first")
        fr = np.arange(g[frame_col].min(), g[frame_col].max() + 1)
        g = pd.merge(pd.DataFrame({frame_col: fr}), g, on=frame_col, how="left")[:len(fr)]
        g["user"], g["session"] = user, sess
        out.append(g.iloc[::15, :].reset_index(drop=True))
    log(f"  {tag}: {len(files)} αρχεία")
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- notebooks 2_x
def concat_modality(frames):
    """frames: λίστα DataFrame με τη σειρά os.listdir (αλφαβητική) του φακέλου τους"""
    user_session = None; parts = []
    for df in frames:
        if "user" in df.columns and "session" in df.columns:
            if user_session is None:
                user_session = df[["user", "session"]]
            df = df.drop(["user", "session"], axis=1)
        parts.append(df)
    return pd.concat(parts + [user_session], axis=1)


def label(combined, labels):
    labels = labels.copy()
    labels.columns = ["user_session"] + [f"minute_{i}" for i in range(1, 11)]
    spi = 4 * 4 * 60
    expanded = []
    for (user, session), group in combined.groupby(["user", "session"]):
        key = f"{user}_{session}"
        if key in labels["user_session"].values:
            row = labels[labels["user_session"] == key].iloc[0, 1:]
            lst = []
            for lb in row:
                lst.extend([lb] * spi)
            lst = lst[:len(group)]
            if len(lst) < len(group):
                lst.extend([np.nan] * (len(group) - len(lst)))
            expanded.extend([[user, session, l] for l in lst])
        else:
            expanded.extend([[user, session, np.nan]] * len(group))
    lab = pd.DataFrame(expanded, columns=["user", "session", "label"])
    return pd.concat([combined.reset_index(drop=True), lab["label"].reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------- notebooks 3_x (μόνο αποθήκευση)
def interp_groups(df, cols):
    df[cols] = df.groupby(["user", "session"])[cols].apply(
        lambda g: g.interpolate(method="linear", axis=0, limit_direction="forward")
    ).reset_index(drop=True, level=["user", "session"])
    return df


def final_biometric(lab, out):
    df = lab.copy()
    df[["SPO2", "pulse_rate"]] = df[["SPO2", "pulse_rate"]].apply(pd.to_numeric, errors="coerce")
    df[["SPO2", "pulse_rate"]] = df.groupby(["user", "session"])[["SPO2", "pulse_rate"]].apply(
        lambda g: g.interpolate(method="linear", limit_direction="forward")
    ).reset_index(drop=True)[["SPO2", "pulse_rate"]]
    df.to_csv(os.path.join(out, "final_biometric_4HZ.csv"), index=False)
    return len(df)


def final_behavioral(lab, out):
    tc = ["heading[deg]", "pitch[deg]", "roll[deg]", "speed[m/s]", "rpm", "gear"]
    # --- telemetry (3_2)
    df_tel = lab[tc + ["user", "session", "label"]].copy()
    mask = (df_tel["user"] == "A") & (df_tel["session"] == "Alert")
    if mask.any():
        target = df_tel.loc[mask, "label"].iloc[0]
        sim = df_tel[(df_tel["label"] == target) & ~(df_tel["user"] == "A") & ~df_tel[tc].isna().any(axis=1)]
        groups = sim.groupby(["user", "session"])
        n = mask.sum()
        keys = list(groups.groups.keys())
        if keys:
            sel = groups.get_group(keys[0])
            if len(sel) < n:
                sel = pd.concat([sel] * int(np.ceil(n / len(sel))), ignore_index=True)
            df_tel.loc[mask, tc] = sel[tc].iloc[:n].reset_index(drop=True).values
            log(f"  A_Alert telemetry: μεταφορά από {keys[0][0]}_{keys[0][1]} (όπως στο 3_2)")
    num = df_tel.select_dtypes(include=["number"]).columns
    df_tel = interp_groups(df_tel, num)
    df_tel.to_csv(os.path.join(out, "final_telemetry_4HZ.csv"), index=False)
    # --- grip: αποθηκεύεται ΠΡΙΝ την παρεμβολή, όπως στο 3_2
    lab[["LGP", "RGP", "user", "session", "label"]].to_csv(os.path.join(out, "final_grip_4HZ.csv"), index=False)
    # --- pose
    cols = [c for c in lab.columns if "_x" in c or "_y" in c or "_z" in c] + ["user", "session", "label"]
    df_pose = lab[cols].copy()
    df_pose = interp_groups(df_pose, df_pose.select_dtypes(include=["number"]).columns)
    df_pose.to_csv(os.path.join(out, "final_pose_4HZ.csv"), index=False)
    return len(lab)


def final_facial(lab, out):
    cols = [c for c in lab.columns if "X" in c or "Y" in c] + ["user", "session", "label"]
    df_fl = lab[cols].copy()
    df_fl = interp_groups(df_fl, df_fl.select_dtypes(include=["number"]).columns)
    df_fl.to_csv(os.path.join(out, "final_FL_4HZ.csv"), index=False)
    drop = [c for c in lab.columns if "X" in c or "Y" in c] + ["Frame", "Frame_Number"]
    df_fau = lab.drop(columns=[c for c in drop if c in lab.columns]).copy()
    df_fau.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_fau = interp_groups(df_fau, df_fau.select_dtypes(include=["number"]).columns)
    df_fau.to_csv(os.path.join(out, "final_FAU_4HZ.csv"), index=False)
    return len(lab)


# ---------------------------------------------------------------- main
def find_labels(given, roots):
    cands = [given] if given else []
    for r in roots:
        cands += [os.path.join(r, "Labels.csv")]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    for r in roots:
        for root, _, fs in os.walk(r):
            if "Labels.csv" in fs and "AppData" not in root:
                return os.path.join(root, "Labels.csv")
    sys.exit("Δεν βρέθηκε το Labels.csv · δώσ' το με --labels \"...\\Labels.csv\"")


def main():
    home = os.path.expanduser("~")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(home, "ULDD", "CSV_Files"))
    ap.add_argument("--features", default=os.path.join(home, "Downloads", "Extracted_Features.zip"))
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default=os.path.join(home, "ULDD", "4HZ_Individual_Data"))
    ap.add_argument("--only", default="bio,beh,fac", help="π.χ. bio για μόνο βιομετρικά")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    labels_path = find_labels(a.labels, [os.getcwd(), os.path.join(home, "ULDD"),
                                         os.path.join(home, "Downloads"), os.path.dirname(a.csv)])
    labels = pd.read_csv(labels_path, header=None)
    log(f"Ετικέτες: {labels_path} ({len(labels)} οδηγήσεις)")
    csv = Source(a.csv)
    log(f"Αρχικά δεδομένα: {a.csv} ({len(csv.names)} αρχεία csv)")
    only = set(a.only.split(","))
    t0 = time.time()

    if "bio" in only:
        log("Βιομετρικά (notebook 1 → 2_1 → 3_1)")
        acc = change_frequency(combine(csv, "ACC", ["X", "Y", "Z"]), 32, 4)
        bvp = change_frequency(combine(csv, "BVP", ["BVP"]), 64, 4)
        eda = combine(csv, "EDA", ["EDA"])
        hr = change_frequency(combine(csv, "HR", ["HR"]), 1, 4)
        o2m = combine(csv, "O2M", ["SPO2", "pulse_rate", "motion"])
        for c in ["SPO2", "pulse_rate", "motion"]:
            o2m[c] = pd.to_numeric(o2m[c], errors="coerce")
        o2m.replace([np.inf, -np.inf], np.nan, inplace=True)
        o2m = change_frequency(o2m, 0.5, 4)
        temp = combine(csv, "TEMP", ["TEMP"])
        # αλφαβητική σειρά αρχείων στον φάκελο Biometrics_4HZ: ACC, BVP, EDA, HR, O2M, TEMP
        lab = label(concat_modality([acc, bvp, eda, hr, o2m, temp]), labels)
        del acc, bvp, eda, hr, o2m, temp
        n = final_biometric(lab, a.out); del lab
        log(f"  → final_biometric_4HZ.csv ({n:,} γραμμές)")

    feats = None
    if "beh" in only or "fac" in only:
        feats = Source(a.features)
        log(f"Χαρακτηριστικά βίντεο: {a.features} ({len(feats.names)} αρχεία csv)")

    if "beh" in only:
        log("Συμπεριφορά (notebook 1 → 2_2 → 3_2) · το βίντεο θέλει αρκετά λεπτά")
        lgp = grip_3to4(combine(csv, "LGP", ["LGP"]))
        rgp = grip_3to4(combine(csv, "RGP", ["RGP"]))
        tel = combine(csv, "Telemetry")
        nan = pd.DataFrame({"user": ["A"] * 144000, "session": ["Alert"] * 144000,
                            **{c: [np.nan] * 144000 for c in tel.columns if c not in ["user", "session"]}})
        tel = change_frequency(pd.concat([nan, tel], axis=0, ignore_index=True), 60, 4)
        pl = video(feats, "PL", "frame")
        # αλφαβητική σειρά στον φάκελο Behavioral_4HZ: LGP, PL, RGP, Telemetry
        lab = label(concat_modality([lgp, pl, rgp, tel]), labels)
        del lgp, rgp, tel, pl
        n = final_behavioral(lab, a.out); del lab
        log(f"  → final_telemetry / final_grip / final_pose_4HZ.csv ({n:,} γραμμές)")

    if "fac" in only:
        log("Πρόσωπο (notebook 1 → 2_3 → 3_3) · το βίντεο θέλει αρκετά λεπτά")
        fau = video(feats, "FAU", "Frame")
        fl = video(feats, "FL", "Frame_Number")
        # αλφαβητική σειρά στον φάκελο Facial_4HZ: FAU, FL
        lab = label(concat_modality([fau, fl]), labels)
        del fau, fl
        n = final_facial(lab, a.out); del lab
        log(f"  → final_FL / final_FAU_4HZ.csv ({n:,} γραμμές)")

    log(f"Τέλος σε {(time.time()-t0)/60:.1f} λεπτά. Αρχεία στο: {a.out}")
    print("\nΕπόμενο βήμα:")
    print(f'  python bodaghi_4hz.py --data "{a.out}" --out results_bodaghi_4hz')


if __name__ == "__main__":
    main()
