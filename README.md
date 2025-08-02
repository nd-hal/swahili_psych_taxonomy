# Benchmarking Sociolinguistic Diversity in Swahili NLP

This repository accompanies the ACL submission:  
**"Benchmarking Sociolinguistic Diversity in Swahili NLP: A Taxonomy-Guided Approach"**

---

## Overview

We introduce the first taxonomy-guided evaluation of Swahili NLP using a new dataset of 2,170 free-text responses collected from Kenyan speakers in response to health-related psychometric tasks.

The dataset captures rich sociolinguistic variation in real-world Swahili, including:

- Sheng (urban Swahili-English hybrid)
- Tribal lexicons
- Code-mixing
- Loanwords

We benchmark both pre-trained and instruction-tuned models (e.g., mBERT, XLM-RoBERTa, SwahBERT, LLaMA, Qwen) and analyze fairness and performance across demographic and linguistic axes.

---

## Repository Structure
```
swahili-psych-taxonomy/
│
├── README.md               # Project overview
├── LICENSE                 # Open source license
├── .gitignore              # Files to ignore in Git
│
├── data/
│   ├── raw/                # Original collected data (text + metadata)
│   └── processed/          # Cleaned/standardized data for modeling
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_error_analysis.ipynb
│   └── 04_fairness_evaluation.ipynb
│
├── models/
│   ├── pretrained/         # Pretrained model configs or links
│   └── LLMs/               # Large Language Model checkpoints/results
│
├── scripts/
│   ├── preprocess_data.py
│   ├── train_model.py
│   ├── evaluate_fairness.py
│   └── run_taxonomy_analysis.py

---
```
## Dataset Summary

- 2,170 open-ended Swahili text responses
- Four psychometric tasks: Anxiety, Trust, Literacy, Numeracy
- Annotated with demographic metadata: tribe, gender, age, education, income
- Includes non-standard language features: Sheng, loanwords, dialectal Swahili, and code-mixing

---

## Tasks and Benchmarks

We evaluate a range of multilingual and instruction-tuned models using both regression and classification tasks. Metrics include:

- Pearson correlation and RMSE for regression
- AUC and F1 for classification
- Group fairness metrics such as Disparate Impact, ∆xAUC, and Fairness Violation

---

## Citation

If you use this work, please cite:

TBD

---

## License

This repository is licensed under the MIT License. See `LICENSE` for details.

