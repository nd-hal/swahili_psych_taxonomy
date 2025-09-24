# Benchmarking Sociolinguistic Diversity in Swahili NLP: A Taxonomy-Guided Approach

This repository accompanies the paper:  
**Add PDF of the paper**

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
📂 Results/          # Folder to store outputs
📂 Scripts/          # Python and R script(s) for model evaluation and plot generation
📜 README.md         # This file
📜 poetry.lock       # Dependency lock file for reproducibility
📜 pyproject.toml    # Configuration for managing dependencies with Poetry
```
## Dataset Summary
- 2,170 open-ended Swahili text responses
- Four psychometric tasks: Anxiety, Trust, Literacy, Numeracy
- Annotated with demographic metadata: tribe, gender, age, education, income
- Includes non-standard languistic features: Sheng, loanwords, dialectal Swahili, and code-mixing

---

## Tasks and Benchmarks

We evaluate a range of multilingual pre-trained language and instruction-tuned models using both regression and classification tasks. Metrics include:

- Pearson correlation and RMSE for regression
- AUC and F1 for classification
- Group fairness metrics such as Disparate Impact, ∆xAUC, and Fairness Violation

---
# Setup & Installation

This project uses **Poetry** for dependency management.

1. Download pipx: https://pipx.pypa.io/stable/installation/
2. Install poetry: https://python-poetry.org/docs/#installing-with-pipx
3. To generate figures 2 and 6:

```{python}
poetry run python scripts/model_fairness_plots.py
```

4. To generate figure 3:
This is an R-based visualization script that can be launched via a Python wrapper using Poetry.
 Make sure you have:
- Poetry installed (`pip install poetry`)
- R and `Rscript` installed on your system
- The file `scripts/intersectionalBiasPlot.R` in the repo

### Then run the script
```{python}
poetry run python scripts/poetry run run-r-analysis
```
5. To generate figure 5:

```{python}
poetry run python scripts/error_analysis_plot.py
```
6. To generate PLMs predictions:

```{python}
poetry run python scripts/plms_eval.py
```
7.Run the following commands to generate the baseline predictions along with binary and continuous predictions from the LLMs:
```{python}
poetry run python scripts/baselines.py

poetry run python scripts/llms_binary_eval.py

poetry run python scripts/llms_continuous_eval.py
```
## Citation

If you use this work, please cite:
@misc{oketch2025benchmarkingsociolinguisticdiversityswahili,
      title={Benchmarking Sociolinguistic Diversity in Swahili NLP: A Taxonomy-Guided Approach}, 
      author={Kezia Oketch and John P. Lalor and Ahmed Abbasi},
      year={2025},
      eprint={2508.14051},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.14051}, 
}

---
# Contributors

Kezia Oketch  
John Lalor      
Ahmed Abbasi    

# Acknowledgments

This research is supported by the University of Notre Dame's Human-centered Analytics Lab (HAL).

## License

This repository is licensed under the MIT License. See `LICENSE` for details.

