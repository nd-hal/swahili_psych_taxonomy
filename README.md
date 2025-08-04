# Benchmarking Sociolinguistic Diversity in Swahili NLP

This repository accompanies the paper submission:  
**"Benchmarking Sociolinguistic Diversity in Swahili NLP: A Taxonomy-Guided Approach"**

---

## Overview

We introduce the first taxonomy-guided evaluation of Swahili NLP using a new dataset of 2,170 free-text responses collected from Kenyan speakers in response to health-related psychometric tasks.

The dataset captures rich sociolinguistic variation in real-world Swahili, including:

- Sheng (urban Swahili-English hybrid)
- Tribal lexicons
- Code-mixing
- Loanwords

We benchmark both pre-trained and instruction-tuned models (e.g., mBERT, XLM-RoBERTa, SwahBERT, Llama, Qwen) and analyze fairness and performance across demographic and linguistic aspects.

---

# Repository Structure

```
📂 Data/             # Contains the Swahili dataset and model predictions
📂 Scripts/          # Python and R script(s) for model evaluation and plot generation
📜 .gitattributes    # Git configuration file
📜 README.md         # This file
📜 poetry.lock       # Dependency lock file for reproducibility
📜 pyproject.toml    # Configuration for managing dependencies with Poetry
📜 requirements.txt  # Contains a list of dependencies required to run the scripts
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

