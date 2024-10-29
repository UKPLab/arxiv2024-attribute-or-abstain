from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from analysis.util import (
    load_predictions_with_scores,
    infer_base_data_path,
    load_results_table,
    MODEL_ORDER,
    MODEL_SHORT_NAMES,
    TASK_SHORT_NAMES,
    EXPERIMENT_SHORT_NAMES
)

N_BINS = 5

TASKS = [
    'qasper',
    'natural_questions',
    'evidence_inference',
    'wice',
    'contract_nli',
    'govreport'
]

MODEL_SHORT_NAMES = {
    k: v
    for k, v in MODEL_SHORT_NAMES.items()
    if k in MODEL_ORDER
}

def main():
    """
    For each task, compute the gold evidence distribution and the predicted
    evidence distribution and plot
    :return:
    """
    # Load results table that contains all hashes
    results_table = load_results_table()
    # Use experiment short names
    results_table['description'] = results_table['description'].apply(
        lambda x: EXPERIMENT_SHORT_NAMES[x] if x in EXPERIMENT_SHORT_NAMES else x
    )
    # Use only certain models and experiments
    results_table = results_table.loc[
        results_table['model'].isin(MODEL_SHORT_NAMES)
    ]
    # Use only description citation
    results_table = results_table.loc[
        results_table['description'] == 'citation'
    ]
    # Use only certain tasks
    results_table = results_table.loc[
        results_table['task'].isin(TASKS)
    ]

    # Add new columns to results_table for the results from
    # compare_evidence_position_predicted_gold
    # Do not add columns for distributions
    results_table['mean_predicted_evidence_position'] = None
    results_table['std_predicted_evidence_position'] = None
    results_table['predicted_entropy'] = None
    results_table['mean_gold_evidence_position'] = None
    results_table['std_gold_evidence_position'] = None
    results_table['gold_entropy'] = None
    results_table['kl_divergence'] = None

    # Iterate over results_table for each task separately, and get output
    # from compare_evidence_position_predicted_gold.
    # get gold and predicted evidence distributions and put into
    # a new dataframe (use gold distribution only once).
    # Store the other results in results_table
    dist_results_table = {
        'task': [],
        'model': [],
        'bin_idx': [],
        'prob': [],
        'is_gold': []
    }
    processed_tasks = []
    for row in results_table.iterrows():
        row = row[1]
        task = row['task']
        hash = row['hash']
        metric_name = row['answer_metric_name']
        results = compare_evidence_position_predicted_gold(
            hash,
            metric_name,
            task
        )
        results_table.loc[
            results_table['hash'] == hash,
            'mean_predicted_evidence_position'
        ] = results['mean_predicted_evidence_position']
        results_table.loc[
            results_table['hash'] == hash,
            'std_predicted_evidence_position'
        ] = results['std_predicted_evidence_position']
        results_table.loc[
            results_table['hash'] == hash,
            'predicted_entropy'
        ] = results['predicted_entropy']
        results_table.loc[
            results_table['hash'] == hash,
            'mean_gold_evidence_position'
        ] = results['mean_gold_evidence_position']
        results_table.loc[
            results_table['hash'] == hash,
            'std_gold_evidence_position'
        ] = results['std_gold_evidence_position']
        results_table.loc[
            results_table['hash'] == hash,
            'gold_entropy'
        ] = results['gold_entropy']
        results_table.loc[
            results_table['hash'] == hash,
            'kl_divergence'
        ] = results['kl_divergence']
        # Add to task_table
        for i in range(len(results['predicted_evidence_positions_dist'])):
            dist_results_table['task'].append(task)
            dist_results_table['model'].append(row['model'])
            dist_results_table['bin_idx'].append(i)
            dist_results_table['prob'].append(results['predicted_evidence_positions_dist'][i])
            dist_results_table['is_gold'].append(False)

        if task not in processed_tasks:
            processed_tasks.append(task)
            # Add gold distribution
            for i in range(len(results['gold_evidence_positions_dist'])):
                dist_results_table['task'].append(task)
                dist_results_table['model'].append('Gold')
                dist_results_table['bin_idx'].append(i)
                dist_results_table['prob'].append(results['gold_evidence_positions_dist'][i])
                dist_results_table['is_gold'].append(True)

    dist_results_table = pd.DataFrame(dist_results_table)
    # Get short model names
    dist_results_table['model'] = dist_results_table['model'].apply(
        lambda x: MODEL_SHORT_NAMES[x] if x in MODEL_SHORT_NAMES else x
    )

    make_plots(dist_results_table)

    summary_table = make_summary_table_and_plot(dist_results_table, results_table)



def make_summary_table_and_plot(dist_results_table: pd.DataFrame, results_table: pd.DataFrame) -> pd.DataFrame:
    """For each task, compute the difference between the gold and the predicted
    evidence probability for each bin."""
    summary_table = dist_results_table.copy()

    # Make new column difference_to_gold by subtracting the task- and bin-matched
    # probability for gold from the probability for the model
    summary_table['difference_to_gold'] = summary_table.apply(
        lambda x: x['prob'] - summary_table.loc[
            (summary_table['task'] == x['task'])
            & (summary_table['bin_idx'] == x['bin_idx'])
            & (summary_table['is_gold'])
        ]['prob'].iloc[0],
        axis=1
    )

    # Plot as single small line plot averaging over tasks
    sns.set_context('paper')
    fig, ax = plt.subplots(1, 1, figsize=(3.03, 2))
    sns.lineplot(
        data=summary_table,
        x='bin_idx',
        y='difference_to_gold',
        hue='model',
        errorbar='sd',
        hue_order=MODEL_SHORT_NAMES.values(),
        ax=ax,
        markers=['x' for _ in MODEL_SHORT_NAMES],
        # dashes=[(2,1) for _ in MODEL_SHORT_NAMES]
    )
    # Remove legend
    ax.get_legend().remove()
    # Remove a and y label
    ax.set_xlabel('')
    ax.set_ylabel('')
    # set title
    ax.set_title('P(Evidence): Difference to Gold')
    # Save plot
    plt.tight_layout()
    plt.savefig(infer_base_data_path() / 'plots' / 'evidence_position_gold_vs_predicted_difference_plot.pdf')

    # Aggregate difference_to_gold by averaging
    aggregated_table_mean = summary_table.groupby(['model', 'bin_idx'])['difference_to_gold'].mean()
    aggregated_table_std = summary_table.groupby(['model', 'bin_idx'])['difference_to_gold'].std()

    aggregated_table = aggregated_table_mean.reset_index()
    aggregated_table['std'] = aggregated_table_std.reset_index()['difference_to_gold']
    # Put mean and std into single column formatted as string with 2 decimals and
    # std in brackets
    aggregated_table['mean_std'] = aggregated_table.apply(
        lambda x: f'{x["difference_to_gold"]:.2f} (±{x["std"]:.2f})',
        axis=1
    )

    # Pivot table to get bin_idx as columns
    aggregated_table = aggregated_table.pivot(
        index='model',
        columns='bin_idx',
        values='mean_std'
    )

    aggregated_table = aggregated_table.loc[MODEL_SHORT_NAMES.values()]

    # Compute mean KL divergence from results table for each model
    kl_divergence = results_table.groupby('model')['kl_divergence'].mean()
    # Rename models in index
    kl_divergence.index = kl_divergence.index.map(MODEL_SHORT_NAMES)

    # Add to results aggregated_table
    aggregated_table['Mean KL Divergence'] = kl_divergence

    # Format summary table to latex
    aggregated_table.to_latex(
        infer_base_data_path() / 'tables' / 'evidence_position_gold_vs_predicted_summary_table.tex',
        escape=False,
        float_format='%.2f'
    )
    # Output summary table as csv
    aggregated_table.to_csv(
        infer_base_data_path() / 'tables' / 'evidence_position_gold_vs_predicted_summary_table.csv',
        index=True
    )

    return aggregated_table


def make_plots(dist_results_table):
    """Make one plot for each task with a single legend underneath. All
    plots are in one row and fit on an A4 page"""
    # We make two rows of plots to give space for the legend
    sns.set_context('paper')
    fig, axs = plt.subplots(
        2,
        len(TASKS),
        figsize=(6.3, 1.8),
        gridspec_kw={'height_ratios': [10, 1]}
    )
    for i, task in enumerate(TASKS):
        task_df = dist_results_table.loc[
            dist_results_table['task'] == task
            ]
        # Use the same colors for the same model
        # Use dashed bar for Gold
        sns.barplot(
            data=task_df,
            x='bin_idx',
            y='prob',
            hue='model',
            hue_order=[model for model in MODEL_SHORT_NAMES.values()] + ['Gold'],
            ax=axs[0][i],
            palette='colorblind'
        )
        axs[0][i].set_ylim(0, 1)
        # Hide ytick labels if it is not the first plot in the row
        if i != 0:
            axs[0][i].set_yticklabels([])
        # Make the gold bar hatched
        n_data_series = len(list(MODEL_SHORT_NAMES.keys()) + ['Gold']) * N_BINS
        relevant_bars_start = n_data_series - N_BINS
        for j, bar in enumerate(axs[0][i].patches[relevant_bars_start: relevant_bars_start + N_BINS]):
            bar.set_fill(False)
            bar.set_hatch('///')
        final_bar = axs[0][i].patches[-1]
        final_bar.set_fill(False)
        final_bar.set_hatch('///')

        axs[0][i].set_title(TASK_SHORT_NAMES[task], fontsize=8)
        # Remove x label
        axs[0][i].set_xlabel('')
        if i == 0:
            axs[0][i].set_ylabel('P(Evidence)', fontsize=8)
        else:
            axs[0][i].set_ylabel('')
        # Remove legend
        axs[0][i].get_legend().remove()
        plt.tight_layout()
        # Set xticks to integers with 0, 1, 2, 3, 4
        axs[0][i].set_xticks(range(N_BINS))

        # Set tick fontsize to 6
        axs[0][i].tick_params(axis='both', which='major', labelsize=6)


    # disable unneeded axes
    for ax in axs[1]:
        ax.axis('off')

    fig.text(0.5, 0.15, 'Position', ha='center', fontsize=8)

    # Make a single legend below the plot
    handles, labels = axs[0][0].get_legend_handles_labels()
    handles = [
        handles[i] for i in range(len(handles))
        if labels[i] not in ['model']
    ]
    labels = [
        labels[i] for i in range(len(labels))
        if labels[i] not in ['model']
    ]
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0),
        ncol=len(labels),
        fontsize=6
    )
    # Move legend down
    fig.canvas.draw()
    plt.tight_layout()
    plt.savefig(infer_base_data_path() / 'plots' / 'evidence_position_gold_vs_predicted_plots.pdf')


def compare_evidence_position_predicted_gold(
        hash: str,
        metric_name: str,
        task_name: str
):
    n_bins = N_BINS

    df = load_predictions_with_scores(hash, metric_name)

    predicted_evidence_positions = df['Predicted Extraction Node Idxs']
    gold_evidence_positions = df['Gold Extraction Node Idxs']
    total_n_nodes = df['Total N Nodes']

    if task_name in [
        'qasper',
        'natural_questions',
        'evidence_inference',
        'wice',
        'contract_nli'
    ]:
        answer_has_multiple_statements = False
    elif task_name in ['govreport']:
        answer_has_multiple_statements = True
    else:
        raise ValueError

    if answer_has_multiple_statements:
        # Expand evidence positions to have a single row per position and normalize
        expanded_predicted_evidence_positions = pd.Series([
            idx / total_n_nodes.iloc[i]
            for i, l1 in enumerate(predicted_evidence_positions)
            for l2 in l1
            for idx in l2
        ])
        expanded_gold_evidence_positions = pd.Series([
            idx / total_n_nodes.iloc[i]
            for i, l1 in enumerate(gold_evidence_positions)
            for l2 in l1
            for l3 in l2
            for idx in l3
        ])
    else:
        # Expand evidence positions to have a single row per position and normalize
        expanded_predicted_evidence_positions = pd.Series([
            idx / total_n_nodes.iloc[i]
            for i, l in enumerate(predicted_evidence_positions)
            for idx in l
        ])
        expanded_gold_evidence_positions = pd.Series([
            idx / total_n_nodes.iloc[i]
            for i, l1 in enumerate(gold_evidence_positions)
            for l2 in l1
            for idx in l2
        ])
    # Compute means and stds
    mean_predicted_evidence_position = expanded_predicted_evidence_positions.mean()
    std_predicted_evidence_position = expanded_predicted_evidence_positions.std()
    mean_gold_evidence_position = expanded_gold_evidence_positions.mean()
    std_gold_evidence_position = expanded_gold_evidence_positions.std()

    # Bin
    expanded_predicted_evidence_positions = pd.cut(
        expanded_predicted_evidence_positions,
        n_bins,
        labels=False
    )
    expanded_gold_evidence_positions = pd.cut(
        expanded_gold_evidence_positions,
        n_bins,
        labels=False
    )

    # Count
    predicted_evidence_positions_counts = expanded_predicted_evidence_positions.value_counts()
    gold_evidence_positions_counts = expanded_gold_evidence_positions.value_counts()

    # Normalize
    predicted_evidence_positions_dist = predicted_evidence_positions_counts / predicted_evidence_positions_counts.sum()
    gold_evidence_positions_dist = gold_evidence_positions_counts / gold_evidence_positions_counts.sum()

    # Fill missing values if necessary
    for i in range(n_bins):
        if i not in predicted_evidence_positions_dist.index:
            predicted_evidence_positions_dist[i] = 0
        if i not in gold_evidence_positions_dist.index:
            gold_evidence_positions_dist[i] = 0

    # Sort
    predicted_evidence_positions_dist = predicted_evidence_positions_dist.sort_index()
    gold_evidence_positions_dist = gold_evidence_positions_dist.sort_index()

    # compute entropies
    predicted_entropy = -np.sum(predicted_evidence_positions_dist * np.log(predicted_evidence_positions_dist))
    gold_entropy = -np.sum(gold_evidence_positions_dist * np.log(gold_evidence_positions_dist))

    # Compute KL divergence
    kl_divergence = np.sum(gold_evidence_positions_dist * np.log(gold_evidence_positions_dist / predicted_evidence_positions_dist))

    return {
        'predicted_evidence_positions_dist': predicted_evidence_positions_dist,
        'mean_predicted_evidence_position': mean_predicted_evidence_position,
        'std_predicted_evidence_position': std_predicted_evidence_position,
        'predicted_entropy': predicted_entropy,
        'gold_evidence_positions_dist': gold_evidence_positions_dist,
        'mean_gold_evidence_position': mean_gold_evidence_position,
        'std_gold_evidence_position': std_gold_evidence_position,
        'gold_entropy': gold_entropy,
        'kl_divergence': kl_divergence
    }


if __name__ == '__main__':
    main()