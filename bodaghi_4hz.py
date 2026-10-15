"""
bodaghi_4hz.py — Ο κώδικας των Bodaghi et al. στα δικά τους δείγματα των 4 Hz,
με τρία πρωτόκολλα αξιολόγησης:

  A  ακριβώς όπως το άρθρο: StratifiedKFold(5, shuffle) πάνω στα δείγματα των 4 Hz,
     MinMaxScaler + PCA(0.90) σε ΟΛΑ τα δεδομένα πριν τον χωρισμό
  B  5-fold ανά οδηγό (StratifiedGroupKFold): κανένας οδηγός δεν είναι ταυτόχρονα
     σε εκπαίδευση και έλεγχο, scaler + PCA μόνο στην εκπαίδευση
  C  LOSO: εκπαίδευση σε 18 οδηγούς, έλεγχος στον 19ο (×19)

Ερώτημα: φουσκώνει η τυχαία ανάμιξη των δειγμάτων των 4 Hz τα νούμερα του άρθρου;
(γειτονικά δείγματα απέχουν 0,25 s και είναι σχεδόν ίδια)

Μοντέλα ακριβώς όπως στο notebook 4.Multimodal_Analysis.ipynb:
  SVC(rbf, C=0.1, gamma='auto', class_weight='balanced')
  RandomForest(100 δέντρα, max_depth=5, min_samples_leaf=10)

Χρήση (από τον φάκελο που έχει το 4HZ_Individual_Data ή με --data):
  python bodaghi_4hz.py --data "C:\\...\\Codes\\4HZ_Individual_Data" --out results_bodaghi_4hz

Το SVM σε ~300.000 δείγματα θέλει πολλές ώρες, γι' αυτό εκπαιδεύεται σε τυχαίο
υποσύνολο (--svm-max-train, προεπιλογή 20000) — ίδιο σε όλα τα πρωτόκολλα, ώστε
η σύγκριση να είναι δίκαιη. Το RF εκπαιδεύεται σε όλα τα δείγματα.
"""
import argparse, os, sys, time, glob
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
FILES = {"bio": "final_biometric_4HZ.csv", "grip": "final_grip_4HZ.csv",
         "tel": "final_telemetry_4HZ.csv", "pose": "final_pose_4HZ.csv",
         "fl": "final_FL_4HZ.csv", "fau": "final_FAU_4HZ.csv"}
COMBOS = {"Bio+Behavioral": ["bio", "grip", "tel", "pose"],
          "Bio+Facial": ["bio", "fl", "fau"],
          "Bio+Behavioral+Facial": ["bio", "grip", "tel", "fl", "fau", "pose"]}
META = ["session", "user", "label"]


def find_data(path):
    if path and os.path.isfile(os.path.join(path, FILES["bio"])):
        return path
    start = path or "."
    hits = glob.glob(os.path.join(start, "**", FILES["bio"]), recursive=True)
    if not hits:
        sys.exit(f"Δεν βρέθηκε το {FILES['bio']} κάτω από '{os.path.abspath(start)}'.\n"
                 "Δώσε τον φάκελο με --data \"...\\4HZ_Individual_Data\".\n"
                 "Αν δεν υπάρχει, δημιουργείται από τα notebooks προεπεξεργασίας των Bodaghi (Codes).")
    return os.path.dirname(hits[0])


def load(folder):
    dfs = {}
    for k, f in FILES.items():
        t = time.time()
        dfs[k] = pd.read_csv(os.path.join(folder, f), low_memory=False)
        print(f"  {f}: {len(dfs[k]):,} γραμμές, {dfs[k].shape[1]} στήλες ({time.time()-t:.0f} s)")
    n = {k: len(d) for k, d in dfs.items()}
    if len(set(n.values())) != 1:
        print(f"  Προσοχή: διαφορετικός αριθμός γραμμών {n} · η συνένωση γίνεται κατά θέση, όπως στο notebook τους")
    return dfs


def build(dfs, parts):
    # όπως στο notebook: συνένωση κατά στήλες, οι μεταβλητές session/user/label από το τελευταίο αρχείο
    X = pd.concat([dfs[p].drop(columns=[c for c in META if c in dfs[p].columns]).reset_index(drop=True)
                   for p in parts], axis=1)
    meta = dfs[parts[-1]][META].reset_index(drop=True).reindex(range(len(X)))
    keep = meta["user"].notna().to_numpy()
    if not keep.all():
        print(f"  Προσοχή: {(~keep).sum():,} γραμμές χωρίς οδηγό (από τη συνένωση κατά θέση) αφαιρούνται")
        X, meta = X[keep].reset_index(drop=True), meta[keep].reset_index(drop=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().any()]
    nan = int(X.isna().sum().sum())
    if nan:
        X = X.fillna(X.median())
    y = meta["label"].apply(lambda x: 0 if x < 4 else 1 if x < 7 else 2).to_numpy()
    groups = meta["user"].astype(str).to_numpy()
    return X.to_numpy(np.float32), y, groups, nan


def models():
    return {"SVM": lambda: SVC(kernel="rbf", C=0.1, gamma="auto", class_weight="balanced",
                               cache_size=1000, random_state=SEED),
            "RF": lambda: RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10,
                                                 random_state=SEED, n_jobs=-1)}


def subsample(idx, y, k, rng):
    if k is None or len(idx) <= k:
        return idx
    # στρωματοποιημένο τυχαίο υποσύνολο
    out = []
    for c in np.unique(y[idx]):
        ic = idx[y[idx] == c]
        out.append(rng.choice(ic, size=max(1, int(round(k * len(ic) / len(idx)))), replace=False))
    return np.concatenate(out)


def evaluate(X, y, groups, splits, leak, name, svm_k, skip_svm, rng):
    """leak=True: scaler+PCA σε όλα τα δεδομένα (όπως το άρθρο)."""
    if leak:
        Z = PCA(n_components=0.90, random_state=SEED).fit_transform(MinMaxScaler().fit_transform(X))
    pred = {m: np.full(len(y), -1) for m in models()}
    for f, (tr, te) in enumerate(splits, 1):
        t = time.time()
        if leak:
            Ztr, Zte = Z[tr], Z[te]
        else:
            sc = MinMaxScaler().fit(X[tr])
            pca = PCA(n_components=0.90, random_state=SEED).fit(sc.transform(X[tr]))
            Ztr, Zte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
        for m, make in models().items():
            if m == "SVM" and skip_svm:
                continue
            sub = np.arange(len(tr))
            if m == "SVM":
                sub = subsample(np.arange(len(tr)), y[tr], svm_k, rng)
            pred[m][te] = make().fit(Ztr[sub], y[tr][sub]).predict(Zte)
        print(f"    {name} fold {f} ({time.time()-t:.0f} s)", flush=True)
    rows, per = [], []
    for m, p in pred.items():
        if (p < 0).all():
            continue
        rows.append({"model": m, "accuracy": 100 * accuracy_score(y, p),
                     "macro_F1": 100 * f1_score(y, p, average="macro")})
        for g in np.unique(groups):
            ig = groups == g
            per.append({"model": m, "driver": g, "accuracy": 100 * accuracy_score(y[ig], p[ig]),
                        "macro_F1": 100 * f1_score(y[ig], p[ig], average="macro",
                                                   labels=np.unique(y[ig]))})
    return rows, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="φάκελος 4HZ_Individual_Data")
    ap.add_argument("--out", default="results_bodaghi_4hz")
    ap.add_argument("--protocols", default="A,B,C")
    ap.add_argument("--combos", default=",".join(COMBOS))
    ap.add_argument("--svm-max-train", type=int, default=20000)
    ap.add_argument("--skip-svm", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    folder = find_data(a.data)
    print(f"Δεδομένα 4 Hz από: {os.path.abspath(folder)}")
    dfs = load(folder)
    f_res = os.path.join(a.out, "bodaghi_4hz_results.csv")
    f_per = os.path.join(a.out, "bodaghi_4hz_per_driver.csv")
    # συνέχεια από εκεί που σταμάτησε: ό,τι έχει ήδη αποθηκευτεί δεν ξανατρέχει
    allrows = pd.read_csv(f_res).to_dict("records") if os.path.isfile(f_res) else []
    allper = pd.read_csv(f_per).to_dict("records") if os.path.isfile(f_per) else []
    done = {(r["combo"], r["protocol"]) for r in allrows}
    if done:
        print(f"Βρέθηκαν αποθηκευμένα αποτελέσματα για {len(done)} συνδυασμούς/πρωτόκολλα · δεν ξανατρέχουν")
    labels = {"A": "A_άρθρο", "B": "B_ανά_οδηγό", "C": "C_LOSO"}
    for ci, combo in enumerate([c.strip() for c in a.combos.split(",")]):
        todo = [p.strip() for p in a.protocols.split(",") if (combo, labels[p.strip()]) not in done]
        if not todo:
            print(f"\n=== {combo}: έτοιμο από προηγούμενη εκτέλεση")
            continue
        X, y, groups, nan = build(dfs, COMBOS[combo])
        print(f"\n=== {combo}: {X.shape[0]:,} δείγματα, {X.shape[1]} χαρακτηριστικά, "
              f"{len(np.unique(groups))} οδηγοί, κλάσεις {np.bincount(y).tolist()}"
              + (f", {nan} κενές τιμές → διάμεσος" if nan else ""))
        for P in todo:
            label = labels[P]
            if P == "A":
                sp, leak = StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y), True
            elif P == "B":
                sp, leak = StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups), False
            else:
                sp, leak = LeaveOneGroupOut().split(X, y, groups), False
            print(f"  πρωτόκολλο {label}")
            rng = np.random.default_rng(SEED + 10 * ci + "ABC".index(P))
            rows, per = evaluate(X, y, groups, list(sp), leak, P, a.svm_max_train, a.skip_svm, rng)
            for r in rows:
                r.update(combo=combo, protocol=label)
                print(f"    → {r['model']}: accuracy {r['accuracy']:.1f} · macro F1 {r['macro_F1']:.1f}")
            for r in per:
                r.update(combo=combo, protocol=label)
            allrows += rows; allper += per
            # αποθήκευση μετά από κάθε πρωτόκολλο
            pd.DataFrame(allrows).to_csv(f_res, index=False)
            pd.DataFrame(allper).to_csv(f_per, index=False)

    res = pd.DataFrame(allrows)[["combo", "protocol", "model", "accuracy", "macro_F1"]]
    per = pd.DataFrame(allper)

    pd.set_option("display.width", 200)
    print("\n=== Κώδικας Bodaghi στα 4 Hz · accuracy και macro F1 (%) ===")
    print("A = όπως το άρθρο (τυχαία δείγματα) · B = 5-fold ανά οδηγό · C = LOSO")
    t = res.pivot_table(index=["combo", "model"], columns="protocol", values=["accuracy", "macro_F1"])
    print(t.round(1).to_string())
    if "C_LOSO" in set(res["protocol"]):
        d = per[per["protocol"] == "C_LOSO"].groupby(["combo", "model"])["macro_F1"]
        print("\n=== LOSO ανά οδηγό · macro F1 (%) ===")
        print(pd.DataFrame({"μέσος": d.mean(), "ελάχ.": d.min(), "μέγ.": d.max()}).round(1).to_string())
    print(f"\nΑρχεία στον φάκελο '{a.out}': bodaghi_4hz_results.csv, bodaghi_4hz_per_driver.csv")
    if not a.skip_svm:
        print(f"Σημείωση: το SVM εκπαιδεύεται σε τυχαίο υποσύνολο {a.svm_max_train:,} δειγμάτων ανά fold.")


if __name__ == "__main__":
    main()
