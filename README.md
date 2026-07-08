# Cross-Industry Project Delay Prediction

Binary classification models that predict whether a project will be delayed, benchmarked across two very different domains: **large infrastructure projects** (World Bank) and **software engineering issues** (JIRA/EMSE2017). The goal is to test whether the same modeling approach generalizes across industries with completely different scale, features, and delay dynamics.

## Datasets

| | World Bank | JIRA (EMSE2017) |
|---|---|---|
| Domain | International development / infrastructure | Open-source software issue tracking |
| Size (after cleaning) | 12,468 closed projects | 64,747 issues across 8 projects (Apache, Duraspace, JavaNet, JBoss, JIRA, Moodle, MuleSoft, WSO2) |
| Target | `is_delayed` = actual duration > median duration for closed projects | `is_delayed` = `delaydays` > 0 |
| Class balance | 50 / 50 (by construction) | 78.6% on-time / 21.4% delayed (natural imbalance) |
| Key features | Project cost, IBRD/IDA commitments, grant amount, lending instrument, environmental risk category, region, financing type | Discussion activity, workload, comment count, fix-version changes, reporter reputation, topic model features |

## Approach

1. **Cleaning & feature engineering** — date parsing and duration calculation for World Bank; per-project CSV loading and merge for the 8 JIRA projects; missing-value imputation (median for numeric, `"Unknown"` for categorical); one-hot encoding for categorical features.
2. **Modeling** — Logistic Regression, Random Forest, and XGBoost trained on an 80/20 stratified split, with class weighting on the imbalanced JIRA target.
3. **Evaluation** — accuracy, precision, recall, F1, and AUC-ROC on the held-out test set, plus 5-fold stratified cross-validation to sanity-check stability.
4. **Explainability** — Random Forest feature importances to compare what actually drives delay in each domain.

## Results

| Dataset | Best model | Accuracy | AUC-ROC |
|---|---|---|---|
| World Bank | XGBoost | 79.2% | 0.866 |
| JIRA (EMSE2017) | Random Forest | 89.1% | 0.923 |

**Top predictive features**
- **World Bank**: project cost, log-transformed project cost, and total IBRD/IDA/grant commitment dominate — delay in infrastructure projects tracks financing scale.
- **JIRA**: remaining time to due date, fix-version change frequency, workload, and reporter reputation dominate — delay in software issues tracks engagement and process signals, not size.

This split is the main takeaway: **cost drives delay in infrastructure projects; team activity and process friction drive delay in software projects.** The same pipeline generalizes across domains, but the features that matter are domain-specific.

## Tech stack

Python, pandas, numpy, scikit-learn, XGBoost, matplotlib, seaborn — all in a single reproducible Jupyter notebook (`main.ipynb`).

## Running it

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl
jupyter notebook main.ipynb
```

## Structure

```
project-delay-prediction-model/
├── main.ipynb                       # Full pipeline: load → clean → model → evaluate → compare
└── data/
    ├── all.xlsx                     # World Bank project dataset
    └── EMSE2017/datasets/           # Per-project JIRA issue CSVs (8 projects)
```
