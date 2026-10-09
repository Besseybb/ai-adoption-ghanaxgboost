# SHAPxAdopt-GH

An explainable XGBoost approach for predicting AI-adoption readiness (Low / Medium / High)
among public educational institutions in Greater Accra, Ghana, with SHAP explanations
and an institution-tier fairness audit.

## Files

| File | What it does |
|---|---|
| `shapxadopt_revision_experiments.py` | Runs every analysis in the study and writes all tables and figures to `revision_outputs/` |
| `predict.py` | Sample inference script: trains the primary model and predicts the readiness tier of new survey responses, with the three main SHAP drivers of each prediction |
| `AI_Readiness_Cleaned_Workbook_300_1.xlsx` | The anonymised survey data (300 responses); sheet `Modeling_Data` is used by the scripts |
| `SHAPxAdopt-GH_Questionnaire.pdf` | The survey questionnaire, with wording and answer options exactly as given to respondents |
| `SHAPxAdopt_GH_Run_Experiment.ipynb` | Google Colab notebook that runs everything in a web browser, with nothing to install |
| `requirements.txt` | Exact package versions used (Python 3.13) |

Keep all the files in the same folder: `predict.py` reuses the feature definitions in
`shapxadopt_revision_experiments.py`.

## Data

The anonymised survey data (`AI_Readiness_Cleaned_Workbook_300_1.xlsx`, 300 responses) are included in this
repository. They contain no names, e-mail addresses or institution names; institutions are identified only by type.
The scripts read the sheet `Modeling_Data`; `Raw_Data` holds the original form responses and `Documentation`
describes the cleaning steps. The questionnaire is in `SHAPxAdopt-GH_Questionnaire.pdf`.

## How to run

```bash
pip install -r requirements.txt

# 1. Reproduce all results (about two minutes on a standard CPU)
python shapxadopt_revision_experiments.py --data AI_Readiness_Cleaned_Workbook_300_1.xlsx

# 2. Train the primary model and score the 45 held-out test cases (macro F1 = 0.894)
python predict.py train --data AI_Readiness_Cleaned_Workbook_300_1.xlsx
python predict.py predict

# 3. Score new survey responses (same columns as the Modeling_Data sheet)
python predict.py predict --new new_responses.xlsx
```

All random processes use fixed seeds (primary split: `random_state = 42`;
stability analysis: seeds 42, 7, 13, 21, 99, 123, 2024, 88, 55, 3).

## Expected headline results (primary test split, n = 45)

| Model | Macro F1 | Macro AUC-ROC | Macro AUC-PR | Brier |
|---|---|---|---|---|
| XGBoost (proposed) | 0.894 | 0.979 | 0.958 | 0.043 |
| Logistic Regression | 0.917 | 0.979 | 0.955 | 0.046 |
| Random Forest | 0.916 | 0.968 | 0.940 | 0.046 |
| SVM (RBF) | 0.896 | 0.973 | 0.937 | 0.049 |
| LightGBM | 0.894 | 0.962 | 0.935 | 0.063 |

See `revision_outputs/` after running step 1 for every table and figure.
