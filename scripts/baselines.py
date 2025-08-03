
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score
)

# Load your dataset
data_path = "SwahiliCulturalPsychData.csv"
df = pd.read_csv(data_path)

# Dimensions to Evaluate
dimensions = [
    ("Text_SubjectiveLit", "Label_SubjectiveLit", "subjective literacy"),
    ("Text_TrustPhys", "Label_TrustPhys", "trust in physicians"),
    ("Text_Anxiety", "Label_Anxiety", "anxiety"),
    ("Text_Numeracy", "Label_Numeracy", "numeracy")
]

# LINEAR REGRESSION (continuous)
def run_linear_regression(data, text_col, label_col, name, n_folds=5):
    texts = data[text_col].astype(str).values
    labels = data[label_col].values
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    pearsons, rmses = [], []
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 4))

    for fold, (train_idx, val_idx) in enumerate(kf.split(texts), start=1):
        print(f"\n[Linear] Fold {fold}: {name}")
        X_train, X_val = texts[train_idx], texts[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)

        model = LinearRegression()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_val_vec)

        r = np.corrcoef(y_val, y_pred)[0, 1]
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        pearsons.append(r)
        rmses.append(rmse)

        print(f"  Pearson r = {r:.4f}, RMSE = {rmse:.4f}")

    return {
        "Dimension": name,
        "Model": "LinearRegression",
        "Pearson r Mean": np.mean(pearsons),
        "Pearson r Std": np.std(pearsons),
        "RMSE Mean": np.mean(rmses),
        "RMSE Std": np.std(rmses),
    }

# LOGISTIC REGRESSION (binary classification)
def run_logistic_regression(data, text_col, label_col, name, n_folds=5):
    texts = data[text_col].astype(str).values
    labels = (data[label_col].values >= np.median(data[label_col].values)).astype(int)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    accs, aucs, f1s, precs, recs = [], [], [], [], []
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 3))

    for fold, (train_idx, val_idx) in enumerate(kf.split(texts), start=1):
        print(f"\n[Logistic] Fold {fold}: {name}")
        X_train, X_val = texts[train_idx], texts[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)

        model = LogisticRegression(max_iter=10000, solver='liblinear')
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_val_vec)
        y_proba = model.predict_proba(X_val_vec)[:, 1]

        accs.append(accuracy_score(y_val, y_pred))
        aucs.append(roc_auc_score(y_val, y_proba))
        f1s.append(f1_score(y_val, y_pred))
        precs.append(precision_score(y_val, y_pred))
        recs.append(recall_score(y_val, y_pred))

        print(f"  Accuracy = {accs[-1]:.4f}, AUC = {aucs[-1]:.4f}, F1 = {f1s[-1]:.4f}")

    return {
        "Dimension": name,
        "Model": "LogisticRegression",
        "Accuracy Mean": np.mean(accs),
        "Accuracy Std": np.std(accs),
        "AUC Mean": np.mean(aucs),
        "AUC Std": np.std(aucs),
        "F1 Mean": np.mean(f1s),
        "F1 Std": np.std(f1s),
        "Precision Mean": np.mean(precs),
        "Recall Mean": np.mean(recs),
    }

# Run all models
lin_results = []
log_results = []

for text_col, label_col, name in dimensions:
    lin_results.append(run_linear_regression(df, text_col, label_col, name))
    log_results.append(run_logistic_regression(df, text_col, label_col, name))

# Save results
out_dir = "./results"
os.makedirs(out_dir, exist_ok=True)

pd.DataFrame(lin_results).to_csv(f"{out_dir}/LinearRegression_Results.csv", index=False)
pd.DataFrame(log_results).to_csv(f"{out_dir}/LogisticRegression_Results.csv", index=False)