# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

# Load dataset
file_path = './Data/FairnessPlotData.csv'
data = pd.read_csv(file_path)

# Define function to plot AUC vs any fairness metric
def plot_auc_vs_fairness(data, tribes, metric_prefix, title_prefix, xlabel, output_path):
    fig, axes = plt.subplots(1, len(tribes), figsize=(14, 6), sharey=True)

    for i, tribe in enumerate(tribes):
        sns.scatterplot(
            ax=axes[i],
            data=data,
            x=f"{metric_prefix}_{tribe}",
            y="AUC",
            hue="Task",
            style="Model",
            s=100,
            legend=(i == 0)
        )
        axes[i].set_title(f"{title_prefix}: {tribe} vs All", fontsize=15)
        axes[i].set_xlabel(xlabel, fontsize=14)
        if i == 0:
            axes[i].set_ylabel("AUC", fontsize=14)
        else:
            axes[i].set_ylabel("")
        axes[i].tick_params(axis='both', labelsize=12)

    # grouped legend
    tasks = data["Task"].unique()
    models = data["Model"].unique()

    palette = sns.color_palette(n_colors=len(tasks))
    available_markers = ['o', 's', 'X', 'D', '^', 'P', '*']
    model_markers = {model: available_markers[i % len(available_markers)] for i, model in enumerate(models)}

    task_handles = [
        Line2D([0], [0], marker='o', color='w', label=task,
               markerfacecolor=palette[i], markersize=10)
        for i, task in enumerate(tasks)
    ]

    model_handles = [
        Line2D([0], [0], marker=model_markers[model], color='black', label=model,
               linestyle='', markersize=10)
        for model in models
    ]

    bold_task = mpatches.Patch(color='none', label=r'$\bf{Task:}$')
    bold_model = mpatches.Patch(color='none', label=r'$\bf{Model:}$')

    legend_handles = [bold_task] + task_handles + [bold_model] + model_handles

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(legend_handles),
        frameon=False,
        prop=FontProperties(size=12),
        handletextpad=0.05
    )

    axes[0].get_legend().remove()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.020)
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    plt.show()


# Run plots

tribes = ['Kamba', 'Kikuyu']

# Figure 6. AUC vs DI
plot_auc_vs_fairness(
    data=data,
    tribes=tribes,
    metric_prefix='DI',
    title_prefix='AUC vs DI',
    xlabel='DI',
    output_path='./Results/DI_Kamba_Kikuyu.png'
)

# Figure 2. AUC vs ∆xAUC
plot_auc_vs_fairness(
    data=data,
    tribes=tribes,
    metric_prefix='∆xAUC',
    title_prefix='AUC vs ∆xAUC',
    xlabel='∆xAUC',
    output_path='./Results/∆xAUC_Kamba_Kikuyu.png'
)