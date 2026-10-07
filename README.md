# Πολυτροπική ανίχνευση κόπωσης οδηγού στο UL-DD

Κώδικας της διπλωματικής εργασίας «Πολυτροπική ανίχνευση κόπωσης οδηγού στο UL-DD».

## Οι τρεις άξονες

| Άξονας | Τι εξετάζει | Αρχείο |
|---|---|---|
| 1 | Όψιμη σύντηξη (late fusion) με meta-classifier, έναντι baselines | `ul_dd_baseline.py` |
| 2 | Ανθεκτικότητα όταν λείπει μια τροπικότητα (αισθητήρας) | `ul_dd_baseline.py` (σενάρια no_bio / no_veh / no_face) |
| 3 | Εξατομίκευση για νέο οδηγό (κανονικοποίηση + fine-tuning σε PyTorch) | `axis3_personalization.py` |

## Δεδομένα

Χρησιμοποιείται το δημόσιο dataset **UL-DD**. Τα δεδομένα **δεν** περιλαμβάνονται σε αυτό το αποθετήριο·
πρέπει να κατεβούν από την επίσημη πηγή τους και να μπουν σε φακέλους `CSV_Files/`, `Extracted_Features/`
μαζί με το `Labels.csv`.

## Εγκατάσταση

```
python -m pip install -r requirements.txt
```

## Εκτέλεση

```
# Άξονες 1 και 2 (δημιουργεί και το features.csv)
python ul_dd_baseline.py --root CSV_Files Extracted_Features --labels Labels.csv

# Άξονας 3 (χρησιμοποιεί το features.csv)
python axis3_personalization.py --features features.csv --out results_axis3
```

## Πρωτόκολλα αξιολόγησης

- **Stratified 5-fold**: όπως στο άρθρο του UL-DD, για άμεση σύγκριση.
- **LOSO (leave-one-subject-out)**: ο οδηγός του ελέγχου δεν έχει δει ποτέ το μοντέλο.

Μετρικές: accuracy, macro precision / recall / F1 (με 95% bootstrap CI), recall της τάξης High,
quadratic kappa, Wilcoxon ανά οδηγό.
