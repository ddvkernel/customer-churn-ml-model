# Customer Churn Prediction Model

A complete end-to-end machine learning project predicting customer churn for a telecom company. Built using Logistic Regression, covering data exploration, feature engineering, model selection, hyperparameter tuning, and interpretation.

## Project Overview

**Goal:** Identify at-risk customers likely to leave, enabling proactive retention campaigns.

**Dataset:** Telco Customer Churn (Kaggle)
- 7,043 customers
- 21 features (demographics, services, billing)
- Target: Binary churn (Yes/No)
- **Class imbalance:** 73% No Churn, 27% Churn

**Model:** Logistic Regression
- **ROC-AUC:** 0.8422
- **F1-Score:** 0.6049
- **Churners Caught:** 209 out of 374 (56% recall)

---

## Key Findings

### Top Churn Protective Factors (Green)
1. **2-year contract** (coeff: -1.33) — Strongest protection
2. **Customer tenure** (coeff: -1.25) — Longevity reduces churn risk
3. **1-year contract** (coeff: -0.69) — Still protective, but less so
4. **Phone service** (coeff: -0.50) — Service bundling increases stickiness

### Top Churn Risk Factors (Red)
1. **Fiber optic internet** (coeff: +1.19) — Highest risk segment
2. **Total charges** (coeff: +0.52) — Price-sensitive customers
3. **Streaming services** (coeff: +0.38) — Weak risk (entertainment add-ons)
4. **Electronic check payment** (coeff: +0.38) — Autopay adoption matters

### Business Insights
- **Fiber optic customers churn 3x more** — Investigate service quality, pricing, or competition
- **Contract type is critical** — Month-to-month contracts are risky; incentivize 2-year upgrades
- **First 6 months are critical** — Tenure shows inverse relationship with churn; focus retention early
- **Service bundling helps** — Customers with phone service are stickier

---

## Project Structure

```
churn-model/
├── README.md                      # This file
├── CustomerChurn.py               # Complete ML pipeline
├── CHURN_MODEL_SUMMARY.md        # Technical summary & actionable insights
├── data/
│   └── telco_churn.csv           # Raw dataset (7,043 rows, 21 features)
└── outputs/
    ├── 06_feature_coefficients.png # Top 15 feature importance
    └── 06_confusion_matrix.png    # Prediction accuracy breakdown
```

---

## How to Run

### 1. Setup (30 minutes)

```bash
# Clone the repo
git clone <repo-url>
cd churn-model

# Create virtual environment
python -m venv churn_env
source churn_env/bin/activate  # On Windows: churn_env\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### 2. Download Data

Download from Kaggle: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

Save as `data/telco_churn.csv`

### 3. Run the Full Pipeline

```bash
python CustomerChurn.py
```

This will:
- Load and preprocess data (Days 1-3)
- Train 3 baseline models and compare (Day 4)
- Tune the best model with GridSearchCV (Day 5)
- Extract feature importance & generate plots (Day 6)
- Output model artifacts and summary (Day 7)

### 4. Use the Trained Model

```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Preprocess new data (same steps as training)
# Then predict churn probability
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]

# Identify high-risk customers (prob > 0.5)
high_risk = X_new[probabilities > 0.5]
```

---

## Model Evaluation

### Confusion Matrix (Test Set)
```
                 Predicted
                No Churn | Churn
Actual No Churn    927   |  108    (Specificity: 89.6%)
Actual Churn       165   |  209    (Sensitivity/Recall: 55.9%)
```

**Interpretation:**
- **True Negatives (927):** Correctly identified loyal customers
- **False Positives (108):** Loyal customers flagged as at-risk (low cost)
- **False Negatives (165):** At-risk customers missed (revenue loss)
- **True Positives (209):** Correctly identified churners for intervention

### Metrics
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.8422 |
| F1-Score | 0.6049 |
| Accuracy | 0.8062 |
| Precision (Churn) | 0.66 |
| Recall (Churn) | 0.56 |

**ROC-AUC of 0.84** means the model ranks a random churner higher than a random non-churner 84% of the time—excellent discrimination.

---

## Feature Engineering Pipeline

1. **Missing Value Handling:** TotalCharges filled with `MonthlyCharges × tenure`
2. **Categorical Encoding:** One-hot encoding for all categorical features (drop_first=True)
3. **Numerical Scaling:** StandardScaler on tenure, MonthlyCharges, TotalCharges
4. **Class Imbalance:** `class_weight='balanced'` in LogisticRegression
5. **Train/Test Split:** 80/20 with stratification to preserve class ratios

---

## Next Steps

### Short Term (Production Ready)
- [ ] Deploy model as REST API (Flask/FastAPI)
- [ ] Build dashboard to score customers in real-time
- [ ] Implement retention campaign targeting top 20% at-risk customers
- [ ] A/B test offer effectiveness (e.g., "Switch to 2-year contract, get $10/mo discount")

### Medium Term (Model Improvement)
- [ ] Implement SHAP values for per-customer explainability
- [ ] Collect feedback on false positives (were they actually at-risk?)
- [ ] Retrain quarterly with fresh data
- [ ] Test ensemble methods (stacking LR + RF + XGBoost)

### Long Term (Business Impact)
- [ ] Track ROI of retention campaigns vs. churn rate improvement
- [ ] Segment customers (high-value, low-value) and apply different strategies
- [ ] Investigate fiber optic segment in detail; consider service quality audit
- [ ] Build predictive churn score as standard metric in CRM

---

## Technologies Used

- **Python 3.14**
- **scikit-learn** — Model training & evaluation
- **pandas** — Data manipulation
- **numpy** — Numerical computing
- **matplotlib & seaborn** — Visualization
- **joblib** — Model serialization

---

## Key Learning Outcomes

✅ End-to-end ML workflow (data → deployment)  
✅ Class imbalance handling (ROC-AUC > Accuracy)  
✅ Hyperparameter tuning with GridSearchCV  
✅ Model interpretation (coefficients as feature importance)  
✅ Cross-validation for generalization  
✅ Confusion matrix analysis for business context  

---

## Author

Built as a learning project to master supervised machine learning fundamentals through hands-on practice.

---

## References

- Dataset: [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- Scikit-learn Docs: [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Metric Guide: [ROC-AUC vs F1 for Imbalanced Data](https://scikit-learn.org/stable/modules/model_evaluation.html)
