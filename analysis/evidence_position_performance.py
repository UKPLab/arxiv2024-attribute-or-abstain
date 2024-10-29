from typing import List, Tuple
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import seaborn as sns
from scipy.stats import spearmanr

from analysis.util import (
    load_predictions_with_scores,
    load_results_table,
    infer_base_data_path,
    CLASSES,
    MODEL_ORDER,
    MODEL_SHORT_NAMES,
    TASK_SHORT_NAMES,
    METRIC_SHORT_NAMES,
    ANSWER_METRICS, EXPERIMENT_SHORT_NAMES
)

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

EXPERIMENT_ORDER = [
    EXPERIMENT_SHORT_NAMES[experiment_name]
    for experiment_name in ['post-hoc', 'citation']
]

N_BINS = 5

def main():
    """Compute performance dependent on position of annotated gold evidence"""
    
    results_table = load_results_table()
    # Adapt descriptions to short names in results table
    results_table['description'] = results_table['description'].apply(
        lambda x: EXPERIMENT_SHORT_NAMES[x]
    )
    # Use only certain models, experiments and tasks
    results_table = results_table.loc[
        results_table['model'].isin(MODEL_SHORT_NAMES)
    ]
    results_table = results_table.loc[
        results_table['description'].isin(EXPERIMENT_ORDER)
    ]
    results_table = results_table.loc[
        results_table['task'].isin(TASKS)
    ]

    # Extract hashes and metrics to evaluate
    config_table = []
    hash_task_name_metric_name_tuples = []
    for row in results_table.iterrows():
        row = row[1]
        hash_task_name_metric_name_tuples.append((
            row['hash'],
            row['task'],
            row['answer_metric_name'])
        )
        config_table.append(
            (
                row['model'],
                row['description'],
                row['task'],
                row['answer_metric_name'],
            )
        )

    raw_results = []
    for tup in hash_task_name_metric_name_tuples:
        raw_results.append(compute_performance_by_evidence_position_for_hash(*tup))
    
    position_performance_results_table = {
        'model': [],
        'description': [],
        'task': [],
        'bin_idx': [],
        'performance': []   
    }
    correlation_table = {
        'task': [],
        'model': [],
        'description': [],
        'metric_name': [],
        'value': []
    }
    # Add column to store performance on instances with examples only
    results_table['overall_performance'] = None
    for config_tuple, result in zip(config_table, raw_results):
        model, description, task, metric_name = config_tuple
        performance_per_bin = result['performance_per_bin']
        overall_performance = result['overall_performance']
        results_table.loc[
            (results_table['model'] == model)
            & (results_table['description'] == description)
            & (results_table['task'] == task),
            'overall_performance'
        ] = overall_performance
        for bin_idx, performance in performance_per_bin.items():
            position_performance_results_table['model'].append(model)
            position_performance_results_table['description'].append(description)
            position_performance_results_table['task'].append(task)
            position_performance_results_table['bin_idx'].append(bin_idx)
            position_performance_results_table['performance'].append(performance)

        # Compute correlation
        r, p = spearmanr(list(range(len(performance_per_bin))), performance_per_bin)
        correlation_table['task'].append(TASK_SHORT_NAMES[task])
        correlation_table['model'].append(model)
        correlation_table['description'].append(description)
        correlation_table['metric_name'].append('r')
        correlation_table['value'].append(r)
        correlation_table['task'].append(TASK_SHORT_NAMES[task])
        correlation_table['model'].append(model)
        correlation_table['description'].append(description)
        correlation_table['metric_name'].append('p')
        correlation_table['value'].append(p)

    position_performance_results_table = pd.DataFrame(position_performance_results_table)

    # Use short names for model, task and metric name
    position_performance_results_table['model'] = position_performance_results_table['model'].apply(
        lambda x: MODEL_SHORT_NAMES[x]
    )

    make_summary_table_and_plot(position_performance_results_table, results_table)

    # Format numbers to int by multiplying by 100
    position_performance_results_table['performance'] = position_performance_results_table['performance'].apply(
        lambda x: round(x * 100)
    )

    make_lineplots(position_performance_results_table)

    position_performance_results_table['task'] = position_performance_results_table['task'].apply(
        lambda x: TASK_SHORT_NAMES[x] + '-' + METRIC_SHORT_NAMES[ANSWER_METRICS[x]]
    )

    pivot_table = position_performance_results_table.pivot(
        index=['model', 'description'],
        columns=['task', 'bin_idx'],
        values='performance'
    )
    pivot_table = pivot_table.loc[
        [
            (model_short_name, experiment)
            for model_short_name in MODEL_SHORT_NAMES.values()
            for experiment in EXPERIMENT_ORDER
        ],
        :
    ]

    base_data_path = infer_base_data_path()

    pivot_table.to_csv(base_data_path / 'tables' / 'evidence_position_performance.csv')

    latex_table = format_results_to_latex(pivot_table)
    with open(base_data_path / 'tables' / 'evidence_position_performance.tex', 'w') as f:
        f.write(latex_table)

    ########################################################
    # Handle correlation table
    correlation_table = pd.DataFrame(correlation_table)
    # Use model short names
    correlation_table['model'] = correlation_table['model'].apply(
        lambda x: MODEL_SHORT_NAMES[x]
    )
    # Put r and p into separate columns
    correlation_table = correlation_table.pivot(
        index=['model', 'description'],
        columns=['task', 'metric_name'],
        values='value'
    )

    # Add '*' to r if p < 0.05
    for task in TASK_SHORT_NAMES.values():
        for model in MODEL_SHORT_NAMES.values():
            for description in EXPERIMENT_ORDER:
                r_value = correlation_table.loc[(model, description), (task, 'r')]
                p_value = correlation_table.loc[(model, description), (task, 'p')]
                if p_value < 0.05:
                    correlation_table.loc[(model, description), (task, 'r')] = f'{r_value:.2f}*'
                else:
                    correlation_table.loc[(model, description), (task, 'r')] = f'{r_value:.2f}'

    # Order rows
    correlation_table = correlation_table.loc[
        [
            (model_short_name, experiment)
            for model_short_name in MODEL_SHORT_NAMES.values()
            for experiment in EXPERIMENT_ORDER
        ],
        :
    ]

    # Remove p value columns
    correlation_table = correlation_table[[(task, 'r') for task in TASK_SHORT_NAMES.values()]]

    # Write latex
    correlation_table.to_latex(
        base_data_path / 'tables' / 'evidence_position_performance_correlations.tex',
        escape=False,
        column_format='l' + 'c' * len(TASK_SHORT_NAMES)
    )


    pass


def make_summary_table_and_plot(
        position_performance_results_table: pd.DataFrame,
        results_table: pd.DataFrame
):
    """For each dataset, compute the difference of performance on each bin to
    the average performance on the complete dataset"""
    results_table = results_table.copy()

    results_table['model'] = results_table['model'].apply(
        lambda x: MODEL_SHORT_NAMES[x]
    )
    position_performance_results_table = position_performance_results_table.copy()
    # For each bin, get the performance on all bins on that task
    position_performance_results_table['diff_to_mean'] = position_performance_results_table.apply(
        lambda x: x['performance'] - results_table.loc[
            (results_table['model'] == x['model'])
            & (results_table['description'] == x['description'])
            & (results_table['task'] == x['task'])
        ]['overall_performance'].iloc[0],
        axis=1
    )

    # Plot the difference to mean for each bin
    # Plot as single small line plot averaging over all tasks and showing
    # standard deviation
    sns.set_context('paper')
    fig, ax = plt.subplots(1, 1, figsize=(3.03, 2.8))
    sns.lineplot(
        data=position_performance_results_table,
        x='bin_idx',
        y='diff_to_mean',
        hue='model',
        errorbar='sd',
        style='description',
        hue_order=MODEL_SHORT_NAMES.values(),
        ax=ax,
        markers=True,
        palette='colorblind'
        # dashes=[(2,1) for _ in MODEL_SHORT_NAMES]
    )
    # Make yticks every 0.1
    ax.set_yticks(np.arange(-0.3, 0.21, 0.1))
    # Remove a and y label
    ax.set_xlabel('')
    ax.set_ylabel('')
    # Set title
    ax.set_title('Response Quality Difference to Mean')
    # Remove 'model' and 'description' entry from legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [
        handles[i] for i in range(len(handles))
        if labels[i] not in ['model', 'description']
    ]
    labels = [
        labels[i] for i in range(len(labels))
        if labels[i] not in ['model', 'description']
    ]
    # Move legend below plot
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    # Save plot
    plt.tight_layout()
    plt.savefig(infer_base_data_path() / 'plots' / 'evidence_position_performance_summary.pdf')

    # Aggregate difference_to_gold by averaging
    aggregated_table_mean = position_performance_results_table.groupby(['model', 'description', 'bin_idx'])['diff_to_mean'].mean()
    aggregated_table_std = position_performance_results_table.groupby(['model', 'description', 'bin_idx'])['diff_to_mean'].std()

    aggregated_table = aggregated_table_mean.reset_index()
    aggregated_table['std'] = aggregated_table_std.reset_index()['diff_to_mean']
    # Put mean and std into single column formatted as string with 2 decimals and
    # std in brackets
    aggregated_table['mean_std'] = aggregated_table.apply(
        lambda x: f'{x["diff_to_mean"]:.2f} (±{x["std"]:.2f})',
        axis=1
    )

    # Pivot table to get bin_idx as columns
    aggregated_table = aggregated_table.pivot(
        index=['model', 'description'],
        columns='bin_idx',
        values='mean_std'
    )

    # Sort rows
    aggregated_table = aggregated_table.loc[MODEL_SHORT_NAMES.values()]

    # Format summary table to latex
    aggregated_table.to_latex(
        infer_base_data_path() / 'tables' / 'evidence_position_performance_summary_table',
        escape=False,
        float_format='%.2f'
    )
    # Output summary table as csv
    aggregated_table.to_csv(
        infer_base_data_path() / 'tables' / 'evidence_position_performance_summary_table.csv',
        index=True
    )

    return aggregated_table


def format_results_to_latex(selective_prediction_pivot_table):
    # Format numbers as ".90"
    latex_str = selective_prediction_pivot_table.to_latex(
        escape=False,
        multirow=True,
        multicolumn_format='c',
        column_format='l' + 'c' * len(selective_prediction_pivot_table.columns.levels[1]),
        bold_rows=True,
        float_format="%.2f"
    )

    return latex_str


def make_lineplots(position_performance_results_table):
    """Make one lineplot for each task with a single legend underneath. All +
    plots are in one row and fit on an A4 page"""
    # We make two rows of plots to give space for the legend
    sns.set_context('paper')
    fig, axs = plt.subplots(
        2,
        len(TASKS),
        figsize=(6.3, 1.8),
        gridspec_kw={'height_ratios': [10, 1]}
    )
    position_performance_results_table['model_description'] = position_performance_results_table['model'] + ' ' + position_performance_results_table['description']
    for i, task in enumerate(TASKS):
        task_df = position_performance_results_table.loc[
            position_performance_results_table['task'] == task
        ]
        # Use the same colors for the same model and use solid lines for
        # post-hoc and dashed lines for citation
        sns.lineplot(
            data=task_df,
            x='bin_idx',
            y='performance',
            hue='model',
            style='description',
            ax=axs[0][i],
            markers=True,
            palette='colorblind'
        )
        axs[0][i].set_ylim(10, 95)
        # Hide ytick labels if it is not the first plot in the row
        if i != 0:
            axs[0][i].set_yticklabels([])

        axs[0][i].set_title(TASK_SHORT_NAMES[task], fontsize=8)
        axs[0][i].set_xlabel('')
        if i == 0:
            axs[0][i].set_ylabel('Response Quality', fontsize=8)
        else:
            # Remove x label
            axs[0][i].set_ylabel('')
        # Remove legend
        axs[0][i].get_legend().remove()
        plt.tight_layout()
        # Set xticks to integers with 0, 1, 2, 3, 4
        axs[0][i].set_xticks(range(N_BINS))

        # Set tick fontsize to 6
        axs[0][i].tick_params(axis='both', which='major', labelsize=6)
    for ax in axs[1]:
        ax.axis('off')

    fig.text(
        0.5,
        0.15,
        'Gold Evidence Position',
        ha='center',
        fontsize=8
    )
    # Make a single legend below the plot
    handles, labels = axs[0][0].get_legend_handles_labels()
    handles = [
        handles[i] for i in range(len(handles))
        if labels[i] in EXPERIMENT_ORDER
    ]
    labels = [
        labels[i] for i in range(len(labels))
        if labels[i] in EXPERIMENT_ORDER
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
    plt.savefig(infer_base_data_path() / 'plots' / 'evidence_position_performance_lineplots.pdf')


def compute_performance_by_evidence_position_tuple_arg(
        args: Tuple
):
    return compute_performance_by_evidence_position_for_hash(*args)


def compute_performance_by_evidence_position_for_hash(
        hash: str,
        task_name: str,
        metric_name: str
):
    """
    Split instances into n_bins bins based on the mean normalized position 
    of the annotated evidence. 
    E.g. n_bins = 5: Bins = [[0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0)]
    Compute performance on each bin separately.
    :return:
    """
    if task_name in CLASSES:
        classes = CLASSES[task_name]
    elif metric_name == 'unanswerable_f1':
        classes = CLASSES[metric_name]
    else:
        classes = None
        
    n_bins = N_BINS

    df = load_predictions_with_scores(hash, metric_name)
 
    # Remove rows without evidence_positions
    has_evidence_positions = df['Gold Extraction Node Idxs'].apply(
        lambda x: bool(sum(len(l) for l in x))
    )
    df = df.loc[has_evidence_positions]
    
    evidence_positions = df['Gold Extraction Node Idxs']
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
        evidence_positions = evidence_positions.apply(
            lambda x: np.mean([idx for l1 in x for l2 in l1 for idx in l2])
        )
    else:
        evidence_positions = evidence_positions.apply(
            lambda x: np.mean([idx for l in x for idx in l])
        )
    # Normalize evidence positions
    evidence_positions = evidence_positions.div(total_n_nodes)
    # Bin values
    evidence_positions = pd.cut(evidence_positions, n_bins, labels=False)

    # Compute bin sizes
    bin_sizes = evidence_positions.value_counts()
    # sort
    bin_sizes = bin_sizes.sort_index()

    # Compute performance per bin
    if metric_name in ['answer_f1', 'rouge_l']:
        scores = df[metric_name]
        performance_per_bin = scores.groupby(by=evidence_positions).mean()
    elif metric_name == 'classification_f1':
        gold_answers = df['Gold Answer']
        predicted_answers = df['Predicted Answer']
        # Map string class labels to integers
        class_name_int_mapping = {
            class_name: i
            for i, class_name in enumerate(classes)
        }
        classes = list(range(len(classes)))
        class_name_int_mapping['none'] = len(classes)
        # Convert labels to integers
        gold_answers = pd.Series(gold_answers).apply(
            lambda x: class_name_int_mapping[x]
        )
        predicted_answers = pd.Series(predicted_answers).apply(
            lambda x: class_name_int_mapping[x]
        )
        # Compute macro f1 score per bin 
        performance_per_bin = pd.Series(dtype=float)
        for bin_idx in range(n_bins):
            bin_mask = evidence_positions == bin_idx
            bin_gold_answers = gold_answers.loc[bin_mask]
            bin_predicted_answers = predicted_answers.loc[bin_mask]
            score = sklearn.metrics.f1_score(
                bin_gold_answers,
                bin_predicted_answers,
                labels=classes,
                average='macro'
            )
            performance_per_bin[bin_idx] = score
    else:
        raise NotImplementedError

    # Sort
    performance_per_bin = performance_per_bin.sort_index()

    # Compute performance on all rows
    if metric_name in ['answer_f1', 'rouge_l']:
        scores = df[metric_name]
        overall_performance = scores.mean()
    elif metric_name == 'classification_f1':
        gold_answers = df['Gold Answer']
        predicted_answers = df['Predicted Answer']
        # Convert labels to integers
        gold_answers = pd.Series(gold_answers).apply(
            lambda x: class_name_int_mapping[x]
        )
        predicted_answers = pd.Series(predicted_answers).apply(
            lambda x: class_name_int_mapping[x]
        )
        overall_performance = sklearn.metrics.f1_score(
            gold_answers,
            predicted_answers,
            labels=classes,
            average='macro'
        )
    else:
        raise NotImplementedError

    return {
        'performance_per_bin': performance_per_bin,
        'overall_performance': overall_performance
    }


if __name__ == '__main__':
    main()
