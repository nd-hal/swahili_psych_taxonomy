

import os, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import replicate
import openai

# Set API keys
os.environ["REPLICATE_API_TOKEN"] = "API token goes here"
openai.api_key = "Deepinfra API token goes here"
openai.api_base = "https://api.deepinfra.com/v1/openai"

# Paths
Data_path = "./Data/SwahiliCulturalPsychData.csv"
Output_csv = "./Data/binary_eval_summary.csv"
os.makedirs(os.path.dirname(Output_csv), exist_ok=True)

# Dimensions
Dimensions = [
    ("New_Text_SubjectiveLit", "Label_SubjectiveLit", "subjective literacy"),
    ("New_Text_TrustPhys", "Label_TrustPhys", "trust in physicians"),
    ("New_Text_Anxiety", "Label_Anxiety", "anxiety"),
    ("New_Text_Numeracy", "Label_Numeracy", "numeracy"),
]

# Model config
Models = [
    {"name": "Qwen-7B", "provider": "qwen", "model_id": "Qwen/Qwen2.5-7B-Instruct", "enabled": True}, #set to 'False' if not running
    {"name": "Qwen-72B", "provider": "qwen", "model_id": "Qwen/Qwen2.5-72B-Instruct", "enabled": True},
    {"name": "Llama-8B", "provider": "llama", "model_id": "meta/meta-llama-3-8b-instruct", "enabled": True},
    {"name": "Llama-405B", "provider": "llama", "model_id": "meta/meta-llama-3.1-405b-instruct", "enabled": True}
]

# Helper functions
def median_discretize(labels):
    return (labels >= np.median(labels)).astype(int)

def create_prompt(text, dimension_name, examples):
    few_shot_examples = "\n".join([
        f'Response: "{txt}"\nAnswer: {"Yes" if lbl == 1 else "No"}\n'
        for txt, lbl in examples
    ])
    return f"""
You are a psychometric AI trained to analyze Swahili and English text. Your task is to evaluate whether a response demonstrates the "{dimension_name}" being tested.

Examples:
{few_shot_examples}

Now evaluate this response:
{text}

Reply only with "Yes" or "No".
""".strip()

def parse_response_binary(response, provider):
    try:
        if provider == "qwen":
            content = response["choices"][0]["message"]["content"].strip().lower()
        else:
            if isinstance(response, list):
                content = ''.join(response).strip().lower()
            elif isinstance(response, str):
                content = response.strip().lower()
            else:
                return 0
        return 1 if content == "yes" else 0
    except Exception:
        return 0

def get_prediction(prompt, model, retries=3, delay=2):
    for attempt in range(retries):
        try:
            if model["provider"] == "qwen":
                response = openai.ChatCompletion.create(
                    model=model["model_id"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.3
                )
            else:
                response = replicate.run(
                    model["model_id"],
                    input={
                        "prompt": prompt,
                        "max_tokens": 100,
                        "temperature": 0.3,
                        "system_prompt": "You are a helpful assistant for binary psychometric classification. Respond only with Yes or No."
                    }
                )
            return parse_response_binary(response, model["provider"])
        except Exception as e:
            print(f" {model['name']} error (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return 0

# Main
def evaluate():
    df = pd.read_csv(Data_path)
    results = []

    for text_col, label_col, dimension_name in Dimensions:
        print(f"\nEvaluating: {dimension_name}")
        X = df[text_col].astype(str).values
        y = median_discretize(df[label_col].values)

        for model in Models:
            if not model["enabled"]:
                continue

            print(f"Model: {model['name']}")
            accs, f1s, aucs = [], [], []

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
                print(f" Fold {fold}")
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                few_shot = list(zip(X_train[:3], y_train[:3]))

                y_pred = []
                for text in tqdm(X_test, desc=f"{model['name']} Fold {fold}"):
                    prompt = create_prompt(text, dimension_name, few_shot)
                    pred = get_prediction(prompt, model)
                    y_pred.append(pred)

                accs.append(accuracy_score(y_test, y_pred))
                f1s.append(f1_score(y_test, y_pred))
                aucs.append(roc_auc_score(y_test, y_pred))

            results.append({
                "Model": model["name"],
                "Dimension": dimension_name,
                "Accuracy Mean": np.mean(accs),
                "Accuracy Std": np.std(accs),
                "F1 Mean": np.mean(f1s),
                "F1 Std": np.std(f1s),
                "AUC Mean": np.mean(aucs),
                "AUC Std": np.std(aucs)
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(Output_csv, index=False)

# Run
if __name__ == "__main__":
    evaluate()
