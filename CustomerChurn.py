"""
#=====================================================
Tested multiple models before finalizing the Baseline LogisticRegression as the perfect model
for the given purpose.
#=====================================================

#=====================================================
GridSearchCV:

param_grid = [
    {'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs'],
    'penalty': ['l2']
    },
    {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['saga'],
    'penalty': ['l1', 'l2']
    }
]

lr_base = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')

grid_search = GridSearchCV(
    lr_base,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print('Starting GridSearchCV...')
print('\ncan take 2-5 minutes')

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}") 
print(f"Best ROC_AUC parameters: {grid_search.best_score_: .4f}")

lr_tuned = grid_search.best_estimator_

y_pred_tuned = lr_tuned.predict(X_test) 
y_pred_proba_tuned = lr_tuned.predict_proba(X_test)[:,1]

roc_auc_tuned = roc_auc_score(y_test, y_pred_proba_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)

#=====================================================
RandomForest:
# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=7)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)

#=====================================================
# XGBoost
xgb_model = XGBClassifier(n_estimators=100, max_depth=7, random_state=42, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:,1]

roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

#=====================================================
Print commands:

# print(f'Feature shape: {X.shape}')
# print(f'Target dist:\n{y.value_counts()}')

# print(df.shape)
# print(df.dtypes)
# print(df.isnull().sum())
# print(df['Churn'].value_counts())

# print(df.head())
# print(df.info())

# print(df[['tenure', 'MonthlyCharges', 'Contract', 'Churn']].head(10))

# print(f"Train set size: {X_train.shape[0]}")
# print(f"Test set size: {X_test.shape[0]}")
# print(f"\nTrain churn distribution: {y_train.value_counts()} \n Test churn distribution: {y_test.value_counts()} ")

# print("Missing values per column:\n", df.isnull().sum())

print("Winner: Logistic Regression\nGridSearch")
print("===================================")
print(f"\nTuned Model:")
print(f"ROC-AUC: {roc_auc_tuned: 0.4f}")
print(f"F1-Score: {f1_tuned: .4f}")

# print(f"\nClassification Report: ")
# print(classification_report(y_test, y_pred, target_names = ['No Churn', 'Churn']))

print(f"Improved ROC-AUC by: {(roc_auc_tuned-roc_auc):.4f}")


print("Random Forest: ")
print("===================================")
print(f"ROC-AUC: {roc_auc_rf:.4f}\nF1-Score: {f1_rf: .4f}")
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred_rf))

print("XGBoost")
print("===================================")
print(f"ROC-AUC: {roc_auc_xgb:.4f}\nF1-Score: {f1_xgb: .4f}")
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred_xgb))

"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load Data
csv_path = Path(__file__).parent / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(csv_path)


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])


X = df.drop(['Churn', 'customerID'], axis=1)
y = (df['Churn'] == 'Yes').astype(int)

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(X[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Train Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

coeff = model.coef_[0]
feature_names = X.columns.tolist()

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coeff,
    'Abs_Coeff': np.abs(coeff)
}).sort_values('Abs_Coeff', ascending=False)

print("\nTop 15 features: ")
print("="*80)
print(coef_df.head(15).to_string(index=False))

fig, ax = plt.subplots(figsize=(10,6))
top_15 = coef_df.head(15)
colors = ['green' if x < 0 else 'red' for x in top_15['Coefficient']]
ax.barh(range(len(top_15)), top_15['Coefficient'], color=colors)
ax.set_yticks(range(len(top_15)))
ax.set_yticklabels(top_15['Feature'])
ax.set_xlabel('Coefficient Value')
ax.set_title('Top 15 Features: Impact on Churn Probability')
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig('06_feature_coefficients.png', dpi=100, bbox_inches='tight')
print("\nFeature importance plot saved to 06_feature_coefficients.png")
plt.close()

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
acc = model.score(X_test, y_test) 

c = confusion_matrix(y_test, y_pred)

print("Logistic Regression Baseline:")
print("===================================")
print(f'ROC-AUC: {roc_auc:.4f}\nF1-Score; {f1:.4f}\nAccuracy: {acc:.4f}')
print("*"*80)

print("Confusion Matrix")
print("-"*80)
print(f"True Negatives(TN): {c[0,0]} | False Positives(FP): {c[0,1]}\nTrue Positives(TP): {c[1,0]} | False Negatives(FN): {c[1,1]}")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(c, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
ax.set_title('Confusion Matrix - Logistic Regression')
plt.tight_layout()
plt.savefig('06_confusion_matrix.png', dpi=100, bbox_inches='tight')
print("Confusion matrix plot saved to 06_confusion_matrix.png")
plt.close()