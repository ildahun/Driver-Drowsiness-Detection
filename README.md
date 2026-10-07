# Πολυτροπική ανίχνευση κόπωσης οδηγού στο UL-DD

Κώδικας της διπλωματικής εργασίας «Πολυτροπική ανίχνευση κόπωσης οδηγού στο UL-DD».

## Οι τρεις άξονες

| Άξονας | Τι εξετάζει | Αρχείο |
|---|---|---|
| 1 | Όψιμη σύντηξη (late fusion) με meta-classifier, έναντι baselines | `ul_dd_baseline.py` |
| 2 | Ανθεκτικότητα όταν λείπει μια τροπικότητα (αισθητήρας) | `ul_dd_baseline.py` (σενάρια no_bio / no_veh / no_face) |
| 3 | Εξατομίκευση για νέο οδηγό (κανονικοποίηση + fine-tuning σε PyTorch) | `axis3_personalization.py` |

## Δεδομένα

Χρησιμοποιείται το dataset **UL-DD** (Bodaghi et al., 2025), διαθέσιμο στο Zenodo με περιορισμένη
πρόσβαση: https://doi.org/10.5281/zenodo.17978727. Τα αρχικά αρχεία του dataset δεν περιλαμβάνονται
σε αυτό το αποθετήριο, σύμφωνα με την άδεια χρήσης του.

Περιλαμβάνονται:
- `features.csv`: τα χαρακτηριστικά που εξήχθησαν (2.800 παράθυρα των 30 s, 19 οδηγοί).
  Με αυτό ο κώδικας τρέχει χωρίς τα αρχικά σήματα.
- `results_axis1_v2/`: αποτελέσματα Αξόνων 1 και 2 (πίνακες MAIN_TABLE και RELATED_TABLE, στατιστικοί έλεγχοι).
- `results_axis3/`: αποτελέσματα Άξονα 3 (εξατομίκευση).
- `axis3_figure.png`: γράφημα του Άξονα 3.

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
