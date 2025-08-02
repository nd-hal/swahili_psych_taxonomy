
# Imports
import os, time, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import openai
import replicate

# Config
openai.api_key = 'Deepinfra API token goes here'
openai.api_base = 'https://api.deepinfra.com/v1/openai'
os.environ['REPLICATE_API_TOKEN'] = "API token goes here"

data_path = "./Data/SwahiliCulturalPsychData.csv"
results_dir = "./Data/PredictionResults"
os.makedirs(results_dir, exist_ok=True)

# Dimensions to score
dimensions = [
    ("New_Text_SubjectiveLit", "Label_SubjectiveLit", "subjective literacy"),
    ("New_Text_TrustPhys", "Label_TrustPhys", "trust in physicians"),
    ("New_Text_Anxiety", "Label_Anxiety", "anxiety"),
    ("New_Text_Numeracy", "Label_Numeracy", "numeracy")
]

# Model definitions
model_configs = [
    {"name": "Qwen-7B", "provider": "qwen", "model_id": "Qwen/Qwen2.5-7B-Instruct", "enabled": True}, #change to 'False' to stop running
    {"name": "Qwen-72B", "provider": "qwen", "model_id": "Qwen/Qwen2.5-72B-Instruct", "enabled": False}, #change to 'True' to run
    {"name": "Llama-8B", "provider": "llama", "model_id": "meta/meta-llama-3-8b-instruct", "enabled": False},
    {"name": "Llama-405B", "provider": "llama", "model_id": "meta/meta-llama-3.1-405b-instruct", "enabled": True}
]

#  Utility functions
def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r = pearsonr(y_true, y_pred)[0]
    return rmse, r

def generate_few_shot_examples(df, text_col, label_col, n=3):
    df_clean = df[[text_col, label_col]].dropna()
    return list(zip(*df_clean.sample(n=n, random_state=42).values.T))

# Prompt and inference
def create_prompt_qwen(text, dimension_name, examples):
    few_shot_examples = "\n\n".join([f'TEXT: "{txt}"\nSCORE: {lbl:.4f}' for txt, lbl in examples])
    return f"""
### TASK: Assign a Score for "{dimension_name}"
You are a psychometric AI trained to analyze Swahili and English text.

---
HOW TO SCORE:
- 0.0 - 0.2 → No or very weak presence of {dimension_name}.
- 0.3 - 0.5 → Moderate presence.
- 0.6 - 0.8 → Strong presence.
- 0.9 - 1.0 → Very strong and explicit presence.

### The scores of these examples were composite of different Likert scale scores.

EXAMPLES:
{few_shot_examples}

---
TEXT TO EVALUATE:
{text}

---
RESPONSE FORMAT (IMPORTANT):
Only return a number between 0 and 1.
""".strip()

def create_prompt_llama(text, dimension_name, examples):
    prompt = f"You are a helpful assistant specialized in psychometric analysis.\n\n"
    for txt, lbl in examples:
        prompt += f'Text: "{txt}"\nScore: {lbl:.2f}\n\n'
    prompt += f'Text: "{text}"\nRespond with a single number between 0 and 1.'
    return prompt.strip()

def get_qwen_prediction(prompt, model_id):
    try:
        response = openai.Completion.create(
            model=model_id,
            prompt=prompt,
            max_tokens=100,
            temperature=0.3
        )
        match = re.search(r"\d\.\d+", response.choices[0].text.strip())
        return float(match.group(0)) if match else 0.5
    except Exception as e:
        print(f"Qwen error: {e}")
        return 0.5

def get_llama_prediction(prompt, model_id, retries=3):
    for _ in range(retries):
        try:
            response = replicate.run(
                model_id,
                input={"prompt": prompt, "max_tokens": 100, "temperature": 0.3}
            )
            text = ''.join(response).strip()
            match = re.search(r"\d\.\d+", text)
            return float(match.group(0)) if match else 0.5
        except Exception as e:
            print(f"Llama error: {e}")
            time.sleep(2)
    return 0.5

# Main evaluation
df = pd.read_csv(data_path)
results = []

for text_col, label_col, dimension_name in dimensions:
    print(f"\nEvaluating: {dimension_name}")
    X = df[text_col].values
    y = df[label_col].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for model in model_configs:
        if not model["enabled"]:
            continue

        model_name = model["name"]
        provider = model["provider"]
        model_id = model["model_id"]

        rmse_scores, r_scores = [], []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
            print(f" {model_name} - Fold {fold}")

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            train_df = pd.DataFrame({text_col: X_train, label_col: y_train})
            examples = generate_few_shot_examples(train_df, text_col, label_col)

            preds = []
            for text in tqdm(X_test, desc=f"Predicting"):
                prompt = (
                    create_prompt_qwen(text, dimension_name, examples)
                    if provider == "qwen"
                    else create_prompt_llama(text, dimension_name, examples)
                )
                pred = (
                    get_qwen_prediction(prompt, model_id)
                    if provider == "qwen"
                    else get_llama_prediction(prompt, model_id)
                )
                preds.append(pred)

            rmse, r = compute_metrics(y_test, preds)
            rmse_scores.append(rmse)
            r_scores.append(r)

        results.append({
            "Model": model_name,
            "Dimension": dimension_name,
            "RMSE Mean": np.mean(rmse_scores),
            "RMSE Std": np.std(rmse_scores),
            "Pearson r Mean": np.nanmean(r_scores),
            "Pearson r Std": np.nanstd(r_scores)
        })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(results_dir, "model_metrics_summary.csv"), index=False)
