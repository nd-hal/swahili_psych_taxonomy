import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error, accuracy_score, f1_score, roc_auc_score
)
from scipy.stats import pearsonr
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-5

data_path = "./Data/SwahiliCulturalPsychData.csv"

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, task_type):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_type = task_type

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float if self.task_type == "regression" else torch.long)
        }

# Loader
def create_loader(df, tokenizer, max_len, batch_size, text_col, label_col, task_type):
    dataset = CustomDataset(
        texts=df[text_col].astype(str).values,
        labels=df[label_col].values,
        tokenizer=tokenizer,
        max_len=max_len,
        task_type=task_type
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=2)

# Evaluation
def run_kfold(model_name, tokenizer_class, model_class, df, text_col, label_col, task_type,
              max_len, batch_size, device, epochs, lr, n_folds=5):
    tokenizer = tokenizer_class.from_pretrained(model_name)
    data = df[[text_col, label_col]].copy()
    data.columns = ["text", "label"]
    data.reset_index(drop=True, inplace=True)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
        print(f"\nFold {fold + 1} - {model_name} - {label_col}")
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]

        train_loader = create_loader(train_data, tokenizer, max_len, batch_size, "text", "label", task_type)
        val_loader = create_loader(val_data, tokenizer, max_len, batch_size, "text", "label", task_type)

        num_labels = 1 if task_type == "regression" else 2
        model = model_class.from_pretrained(model_name, num_labels=num_labels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        if task_type == "regression":
            loss_fn = torch.nn.MSELoss()
        else:
            weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_data["label"].values)
            loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(device))

        model.train()
        for _ in range(epochs):
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=mask).logits
                outputs = outputs.squeeze(-1) if task_type == "regression" else outputs

                loss = loss_fn(outputs, labels if task_type == "regression" else labels.long())
                loss.backward()
                optimizer.step()

        model.eval()
        preds, true = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=mask).logits
                if task_type == "regression":
                    preds.extend(outputs.squeeze(-1).cpu().numpy())
                else:
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    preds.extend(probs)
                true.extend(labels.cpu().numpy())

        preds = np.array(preds)
        true = np.array(true)

        if task_type == "regression":
            rmse = np.sqrt(mean_squared_error(true, preds))
            r, _ = pearsonr(true, preds)
            se_rmse = np.std((true - preds) ** 2, ddof=1) / (2 * rmse * np.sqrt(len(true)))
            se_r = np.std(preds, ddof=1) / np.sqrt(len(true))
            fold_metrics.append((rmse, se_rmse, r, se_r))
            print(f"RMSE = {rmse:.4f} ± {se_rmse:.4f}, Pearson’s r = {r:.4f} ± {se_r:.4f}")
        else:
            pred_labels = (preds >= 0.5).astype(int)
            acc = accuracy_score(true, pred_labels)
            f1 = f1_score(true, pred_labels)
            try:
                auc = roc_auc_score(true, preds)
            except:
                auc = np.nan
            se_acc = np.sqrt((acc * (1 - acc)) / len(true))
            se_f1 = np.std([f1], ddof=1) if len(true) > 1 else 0
            fold_metrics.append((acc, se_acc, f1, se_f1, auc))
            print(f"Accuracy = {acc:.4f} ± {se_acc:.4f}, F1 = {f1:.4f} ± {se_f1:.4f}, AUC = {auc:.4f}")

    return fold_metrics


# Run All Tasks and Models
df = pd.read_csv(data_path)

# Apply binarization for classification tasks
for col in ["Label_SubjectiveLit", "Label_TrustPhys", "Label_Anxiety", "Label_Numeracy"]:
    df[col] = df[col].astype(float)

models = {
    "bert-base-multilingual-uncased": (BertTokenizer, AutoModelForSequenceClassification),
    "castorini/afriberta_large": (AutoTokenizer, AutoModelForSequenceClassification),
    "xlm-roberta-base": (AutoTokenizer, AutoModelForSequenceClassification),
    "pranaydeeps/SwahBERT-base-cased": (AutoTokenizer, AutoModelForSequenceClassification),
}

regression_tasks = [
    ("Text_SubjectiveLit", "Label_SubjectiveLit"),
    ("Text_TrustPhys", "Label_TrustPhys"),
    ("Text_Anxiety", "Label_Anxiety"),
    ("Text_Numeracy", "Label_Numeracy"),
]

classification_tasks = [
    ("Text_SubjectiveLit", "Label_SubjectiveLit"),
    ("Text_TrustPhys", "Label_TrustPhys"),
    ("Text_Anxiety", "Label_Anxiety"),
    ("Text_Numeracy", "Label_Numeracy"),
]

# Binarize labels just for classification
for _, label_col in classification_tasks:
    df[label_col] = (df[label_col] > df[label_col].median()).astype(int)

for model_name, (tok_cls, mod_cls) in models.items():
    for text_col, label_col in regression_tasks:
        print(f"\n REGRESSION: {model_name} - {label_col}")
        run_kfold(model_name, tok_cls, mod_cls, df, text_col, label_col,
                  "regression", MAX_LEN, BATCH_SIZE, DEVICE, EPOCHS, LEARNING_RATE)

    for text_col, label_col in classification_tasks:
        print(f"\n CLASSIFICATION: {model_name} - {label_col}")
        run_kfold(model_name, tok_cls, mod_cls, df, text_col, label_col,
                  "classification", MAX_LEN, BATCH_SIZE, DEVICE, EPOCHS, LEARNING_RATE)