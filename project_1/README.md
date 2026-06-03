# Protein Functional Group Classification
**Student Name:** Rohith Dinesh  
**Course and Section:** CSE 163 AC


## Project Overview

This project investigates the varience of biochemical features derived from protein amino acid sequences and whether they distinguish between three functional protein groups: enzymes, receptors, and structural proteins. The analysis uses primary exploratory analysis, including graphical observations of the population distribution. Then, the project conducts a statistical testing (one-way ANOVA) to prove significant differences across our target "features" and implements a machine learning (Decision Tree classification) model to use such features and train to predict which functional group a protein belongs to given specific properties.


## Required Libraries

Install all dependencies before running any scripts:

```
pip install pandas scikit-learn scipy seaborn matplotlib requests
```

---

## File Descriptions

| File | Description |
|---|---|
| `load_data.py` | Fetches raw protein data from the UniProt REST API for three functional groups and saves the result as `protein_dataset.csv`. Run this first to generate the dataset. |
| `data_processing.py` | Contains shared utility functions for loading the dataset, computing biochemical feature ratios from amino acid sequences, and splitting data by functional group. Imported by other modules — not run directly. |
| `eda.py` | Run this file second to generate graphical observations of the dataset, including boxplots, heatmaps, and histograms that compare biochemical property and size variables. The observations explored were the basis behind the selected statistical test. |
| `statistical_analysis.py` | Performs one-way ANOVA tests with Bonferroni correction, accounting for statistical test number, on all biochemical and size features across protein groups. Outputs results to `anova_results.txt`. Results outputted were the basis behind which features used in the Machine Learning model. |
| `split_analysis.py` | Evaluates Decision Tree classifier accuracy across multiple train/test split ratios (10%–50%) by averaging over 30 random trials per ratio. Outputs `split_results.txt` identifying the optimal split used in the final model. Run before `ml_analysis.py` to verify the chosen split. |
| `ml_analysis.py` | Trains and evaluates a Decision Tree classifier to predict protein functional group. Outputs `decision_tree.png`, `confusion_matrix.png`, and `ml_results.txt`. |
| `test_project.py` | Contains unit tests for core project functions using a small manually constructed dataset with known values. |

---

## How to Reproduce Results

### Step 1 — Generate the Dataset

Run the data collection script to download protein data from UniProt and save it locally. This takes a couple seconds.

```
python load_data.py
```

This creates `protein_dataset.csv` in your current directory (500 proteins per functional group, 1500 total).

### Step 2 — Run Exploratory Analysis

```
python eda.py
```

This file takes care of calculating biochemical property presence through proportional calculations querying from `data_processing.py`. Then, it creates graphical representaitons of features data including boxplots, histograms, and heatmaps, allowing for quick brief analysis over trends found in the data with given calulcated biochemical and size properties in the csv.

### Step 3 — Run Statistical Analysis

```
python statistical_analysis.py
```

Outputs `anova_results.txt` containing F-statistics, p-values, and Bonferroni-corrected significance conclusions for each biochemical feature.

### Step 4 — Run Split Analysis

```
python split_analysis.py
```

Outputs `split_results.txt` containing mean accuracy and standard deviation for each tested split ratio across 30 random trials, and identifies the optimal train/test split for the Decision Tree classifier.

### Step 5 — Run Machine Learning Analysis

```
python ml_analysis.py
```

Outputs:
- `ml_results.txt` — model accuracy and classification report
- `confusion_matrix.png` — heatmap of predicted vs. observed labels
- `decision_tree.png` — visualization of the trained decision tree

### Step 6 — Run Tests

```
python test_project.py
```

All tests should pass with no errors. A `SmallSampleWarning` from scipy is expected and can be safely ignored as it is a consequence of the small test dataset size, not an error.

---

## Notes

- All scripts must be run from the same directory as `protein_dataset.csv`.
- `data_processing.py` is a shared module and does not need to be run directly.
- The dataset is fetched from the publicly available UniProt REST API
  (`https://rest.uniprot.org`) and does not require authentication.
- Results may vary slightly across runs due to potential changes in the
  UniProt database, but model accuracy should remain approximately 75-80%. Test_train split was calibrated in a way such to prevent both overfitting and underfitting.
