# Πολυτροπική ανίχνευση κόπωσης οδηγού στο UL-DD

Κώδικας και αποτελέσματα της διπλωματικής εργασίας «Πολυτροπική ανίχνευση κόπωσης οδηγού στο UL-DD».

## Οι άξονες και τα αρχεία

| Άξονας | Τι εξετάζει | Αρχείο |
|---|---|---|
| 1 | Όψιμη σύντηξη με meta-classifier, έναντι των σχετικών εργασιών | `ul_dd_baseline.py` |
| 2 | Ελλιπείς αισθητήρες: τυχαία κενά 0–60%, βλάβη αισθητήρα, πραγματικά κενά · 15 μέθοδοι σύγκρισης | `axis2_missing.py` |
| 3 | Εξατομίκευση για νέο οδηγό (κανονικοποίηση + fine-tuning σε PyTorch) | `axis3_personalization.py` |
| 1+2+3 | Ενοποίηση σε νέο οδηγό (LOSO), με και χωρίς εξατομίκευση | `axis13_unified.py` |
| Έλεγχος | Ο κώδικας των Bodaghi et al. στα δικά μας χαρακτηριστικά | `bodaghi_exact.py` |
| Έλεγχος | Η προεπεξεργασία των Bodaghi et al. στα 4 Hz, από τα αρχικά δεδομένα | `bodaghi_prep_4hz.py` |
| Έλεγχος | Τα μοντέλα των Bodaghi et al. στα 4 Hz: τυχαίος χωρισμός, ανά οδηγό, LOSO | `bodaghi_4hz.py` |

## Δεδομένα

Χρησιμοποιείται το dataset **UL-DD**. Τα δεδομένα και ό,τι προκύπτει απευθείας από αυτά
(`features.csv`, `4HZ_Individual_Data/`) **δεν** περιλαμβάνονται, λόγω της άδειας χρήσης του dataset.
Κατεβαίνουν από την επίσημη πηγή (`CSV_Files/`, `Extracted_Features/`, `Labels.csv`).

## Εγκατάσταση

```
python -m pip install -r requirements.txt
```

## Εκτέλεση

```
# Άξονας 1 (δημιουργεί και το features.csv)
python ul_dd_baseline.py --root CSV_Files Extracted_Features --labels Labels.csv

# Άξονας 2
python axis2_missing.py --features features.csv --out results_axis2
python axis2_missing.py --features features.csv --out results_axis2_loso_norm --cv loso --normalize

# Άξονας 3
python axis3_personalization.py --features features.csv --out results_axis3

# Ενοποίηση
python axis13_unified.py --features features.csv --out results_unified

# Έλεγχος του άρθρου των Bodaghi et al. στα 4 Hz
python bodaghi_prep_4hz.py --csv CSV_Files --features Extracted_Features.zip --labels Labels.csv --out 4HZ_Individual_Data
python bodaghi_4hz.py --data 4HZ_Individual_Data --out results_bodaghi_4hz
```

## Κύρια αποτελέσματα

| Πείραμα | Αποτέλεσμα |
|---|---|
| Άξονας 2, τυχαία κενά 60% (5-fold) | Η υβριδική μέθοδος (+ LOCF) διατηρεί το 100% του F1 · η συμπλήρωση με 0 το 47% |
| Άξονας 2, βλάβη αισθητήρα 40 λεπτών (5-fold) | Όψιμη σύντηξη με dropout ≈ 0,87 · LOCF 0,73 · stacking 0,69 |
| Ενοποίηση, νέος οδηγός (LOSO) | macro F1 0,362 → **0,626** με εξατομίκευση (18/19 οδηγοί, p < 0,001) |
| Bodaghi στα 4 Hz, RF όλες οι τροπικότητες | 75,2% με τυχαίο χωρισμό (όπως το άρθρο) → 29,5–33,8% με χωρισμό ανά οδηγό |

Οι φάκελοι `results_*` περιέχουν τα πλήρη αποτελέσματα (CSV και γραφήματα).

## Πρωτόκολλα αξιολόγησης

- **Stratified 5-fold**: όπως στο άρθρο του UL-DD, για άμεση σύγκριση.
- **LOSO (leave-one-subject-out)**: ο οδηγός του ελέγχου δεν έχει δει ποτέ το μοντέλο.

Μετρικές: accuracy, macro precision / recall / F1, recall της τάξης High, quadratic kappa,
ψευδείς συναγερμοί, έλεγχος Wilcoxon ανά οδηγό.
