# Loan Default Prediction Model

This project builds an interpretable machine learning model to predict whether a loan applicant will default using a synthetic loan application dataset.

The notebook includes:

- exploratory data analysis
- feature engineering
- logistic regression modelling
- comparison against an existing rule-based system
- fairness analysis across employment groups

---

# How to Run

1. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

2. Ensure the dataset is located at:

`data/loan_applications.csv`

3. Run the notebook:

```bash
jupyter notebook
```

Then open the notebook and run all cells from top to bottom. All results (figures, model training, evaluation, and fairness analysis) will reproduce automatically.

---

# Approach

## Data Issues

A key decision made was that only loans with known outcomes were used for training.

```
defaulted -> 1
repaid -> 0
```

Loans with `ongoing` status were removed because their final outcome is not yet observed.

---

## Model Choice

A **logistic regression model** was used because:

- the prediction task is binary
- the dataset is relatively small
- coefficients provide clear interpretability

The model was trained using a **scikit-learn pipeline** with:

- median imputation
- feature scaling
- one-hot encoding for employment status
- class weighting to address imbalance

---

## Evaluation

The model was compared against the existing **rule-based system** using:

- ROC AUC
- precision / recall / F1
- confusion matrices

This allows evaluation of both **ranking quality** and **actual decision behaviour**.

---

# Possible Improvements

With more time, improvements could include:

- hyperparameter tuning using cross-validation  
- optimizing the decision threshold based on business costs  
- exploring nonlinear interpretable models (e.g., decision trees or boosting)  
- incorporating additional financial behaviour features  
- modelling ongoing loans using survival analysis