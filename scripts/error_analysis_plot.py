import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches  
import statsmodels.api as sm
import statsmodels.formula.api as smf
import re

# Load data 
df = pd.read_excel("./Data/PredsSwahiliCulturalPsychData.xlsx")
df = df.loc[:, ~df.columns.duplicated()]

# setup
bert_models = ["AfriBERTa", "mBERT", "SwahBERT", "XLM-RoBERTa"]
ling_feats = ['CodeMixed', 'Sheng', 'Tribal', 'Loan']
delta_columns = [col for col in df.columns if col.startswith("Delta_")]

# main effects 
main_results = []

for col in delta_columns:
    match = re.match(r'Delta_(.+?)_(.+)', col)
    if not match:
        continue
    model_name, task_name = match.groups()
    if model_name not in bert_models:
        continue

    y = df[col]
    intercept_added = False

    for feat in ling_feats:
        X = sm.add_constant(df[[feat]])
        model_fit = sm.OLS(y, X).fit()

        coef = model_fit.params[feat]
        stderr = model_fit.bse[feat]
        pval = model_fit.pvalues[feat]
        r2 = model_fit.rsquared

        if pval < 0.001:
            stars = '***'
        elif pval < 0.01:
            stars = '**'
        elif pval < 0.05:
            stars = '*'
        else:
            stars = ''
        pval_str = f"{pval:.3f}{stars}"

        main_results.append({
            'Model': model_name,
            'Task': task_name,
            'Feature': feat,
            'Coef': round(coef, 3),
            'StdErr': round(stderr, 3),
            'p-value': pval_str,
            'R²': round(r2, 3)
        })

        if not intercept_added:
            const_coef = model_fit.params['const']
            const_stderr = model_fit.bse['const']
            const_pval = model_fit.pvalues['const']

            if const_pval < 0.001:
                const_stars = '***'
            elif const_pval < 0.01:
                const_stars = '**'
            elif const_pval < 0.05:
                const_stars = '*'
            else:
                const_stars = ''
            const_pval_str = f"{const_pval:.3f}{const_stars}"

            main_results.append({
                'Model': model_name,
                'Task': task_name,
                'Feature': 'Intercept',
                'Coef': round(const_coef, 3),
                'StdErr': round(const_stderr, 3),
                'p-value': const_pval_str,
                'R²': round(r2, 3)
            })
            intercept_added = True

# Convert to Df and save 
df_main = pd.DataFrame(main_results)
df_main.to_csv("/Users/koketch/Desktop/main_effects_with_significance.csv", index=False)


# Interactions 
long_data = []
for col in delta_columns:
    match = re.match(r'Delta_(.+?)_(.+)', col)
    if not match:
        continue
    model, task = match.groups()
    if model in bert_models:
        temp = df.copy()
        temp["Model"] = model
        temp["Task"] = task
        temp["Delta"] = df[col]
        long_data.append(temp)

df_long = pd.concat(long_data, ignore_index=True)

interaction_results = []
interaction_vars = [
    "CodeMixed", "Sheng", "Tribal", "Loan",
    "CodeMixed:Sheng", "CodeMixed:Tribal", "CodeMixed:Loan",
    "Sheng:Tribal", "Sheng:Loan", "Tribal:Loan"
]

for (model, task), task_df in df_long.groupby(["Model", "Task"]):
    formula = (
        "Delta ~ CodeMixed * Sheng + CodeMixed * Tribal + CodeMixed * Loan + "
        "Sheng * Tribal + Sheng * Loan + Tribal * Loan"
    )
    fit = smf.ols(formula, data=task_df).fit()
    coef_df = fit.summary2().tables[1].reset_index()
    coef_df.columns = ["Variable", "Coef", "StdErr", "t", "p-value", "CI_low", "CI_high"]
    coef_df["Model"] = model
    coef_df["Task"] = task
    interaction_results.append(coef_df[coef_df["Variable"].isin(interaction_vars)])

df_interact = pd.concat(interaction_results, ignore_index=True)

# Label mapping and ordering 
label_map = {
    "Intercept": "Intercept", "CodeMixed": "Code-Mixing", "Sheng": "Sheng Usage",
    "Tribal": "Tribal Lexicon", "Loan": "Loanwords",
    "CodeMixed:Sheng": "Code-Mixing×Sheng", "CodeMixed:Tribal": "Code-Mixing×Tribal",
    "CodeMixed:Loan": "Code-Mixing×Loan", "Sheng:Tribal": "Sheng×Tribal",
    "Sheng:Loan": "Sheng×Loan", "Tribal:Loan": "Tribal×Loan"
}
main_order = ["Intercept", "Code-Mixing", "Sheng Usage", "Tribal Lexicon", "Loanwords"]
interaction_order = [
    "Code-Mixing", "Sheng Usage", "Tribal Lexicon", "Loanwords",
    "Code-Mixing×Sheng", "Code-Mixing×Tribal", "Code-Mixing×Loan",
    "Sheng×Tribal", "Sheng×Loan", "Tribal×Loan"
]

df_main["Feature"] = df_main["Feature"].map(label_map)
df_main["Label"] = pd.Categorical(df_main["Feature"], categories=main_order, ordered=True)
df_interact["Label"] = df_interact["Variable"].map(label_map)
df_interact["Label"] = pd.Categorical(df_interact["Label"], categories=interaction_order, ordered=True)
df_main["Task"] = df_main["Task"].str.replace(r"\.1$", "", regex=True)
df_interact["Task"] = df_interact["Task"].str.replace(r"\.1$", "", regex=True)

# Plotting Setup 
bert_palette = {
    'AfriBERTa': 'darkgreen',
    'mBERT': 'skyblue',
    'SwahBERT': 'darkorange',
    'XLM-RoBERTa': 'darkred'
}
marker_map = {
    "Numeracy": 'o',
    "Trust": 's',
    "Literacy": '^',
    "Anxiety": 'D'
}

fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

# Main Effects
for (model, task), group in df_main.groupby(["Model", "Task"]):
    if task not in marker_map:
        continue
    group = group.drop_duplicates(subset="Label")
    group = group.set_index("Label").reindex(main_order).reset_index()
    axs[0].plot(
        group["Label"], group["Coef"],
        color=bert_palette[model],
        marker=marker_map[task],
        linewidth=1.5, markersize=6, alpha=0.9
    )
axs[0].axhline(0, color='gray', linestyle='--', linewidth=1)
axs[0].set_xlabel("Linguistic Feature", fontsize=12)
axs[0].set_ylabel("Coefficients", fontsize=12)
axs[0].tick_params(axis='x', rotation=45)

# Interaction Effects
for (model, task), group in df_interact.groupby(["Model", "Task"]):
    if task not in marker_map:
        continue
    group = group.drop_duplicates(subset="Label")
    group = group.set_index("Label").reindex(interaction_order).reset_index()
    axs[1].plot(
        group["Label"], group["Coef"],
        color=bert_palette[model],
        marker=marker_map[task],
        linewidth=1.5, markersize=6, alpha=0.9
    )
axs[1].axhline(0, color='gray', linestyle='--', linewidth=1)
axs[1].set_xlabel("Feature or Interaction", fontsize=12)
axs[1].tick_params(axis='x', rotation=45)

# Red patches
# Main Effects plot
axs[0].add_patch(
    patches.Rectangle((1.1, -1.46), 2.0, 2.1, edgecolor='red', facecolor='none', linewidth=2)
)

# Interaction Effects plot
axs[1].add_patch(
    patches.Rectangle((0.8, -1.0), 2.5, 1.6, edgecolor='red', facecolor='none', linewidth=2)
)
axs[1].add_patch(
    patches.Rectangle((6.5, -0.5), 2, 0.7, edgecolor='red', facecolor='none', linewidth=2)
)

# Legend
model_handles = [mlines.Line2D([], [], color=color, label=model, linewidth=3)
                 for model, color in bert_palette.items()]
task_handles = [mlines.Line2D([], [], color='gray', marker=marker, linestyle='None',
                              markersize=8, label=task)
                for task, marker in marker_map.items()]

# pseudo-labels as section headers
model_header = mlines.Line2D([], [], color='none', label=r"$\bf{Model:}$")
task_header = mlines.Line2D([], [], color='none', label=r"$\bf{Task:}$")

legend_handles = [model_header] + model_handles + [task_header] + task_handles

fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.96),  
    ncol=len(legend_handles),
    frameon=False,
    fontsize=12
)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("./Results/error_analysis.png", dpi=1000, bbox_inches='tight')
plt.show()
