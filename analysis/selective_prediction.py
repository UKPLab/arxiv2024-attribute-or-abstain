import json
from pathlib import Path
from typing import Tuple
import multiprocessing

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from analysis.util import (
    selective_prediction,
    load_predictions_with_scores,
    infer_base_data_path,
    load_results_table,
    CLASSES,
    MODEL_ORDER,
    MODEL_SHORT_NAMES,
    TASK_SHORT_NAMES,
    METRIC_SHORT_NAMES,
    EXPERIMENT_SHORT_NAMES,
    ANSWER_METRICS,
    CLASS_SHORT_NAMES
)

EXPERIMENT_ORDER = [
    EXPERIMENT_SHORT_NAMES[experiment_name]
    for experiment_name in ['post-hoc', 'citation']
]

def main():
    """
    For each task, compute selective prediction results as the difference in AUC
    between using attribution to order instances and using random ordering.
    Output csv and latex tables.
    :return:
    """
    # Load results table that contains all hashes
    results_table = load_results_table()
    # Adapt descriptions to short names in results table
    results_table['description'] = results_table['description'].apply(
        lambda x: EXPERIMENT_SHORT_NAMES[x]
    )
    # Use only certain models and experiments
    results_table = results_table.loc[
        results_table['model'].isin(MODEL_ORDER)
    ]
    results_table = results_table.loc[
        results_table['description'].isin(EXPERIMENT_ORDER)
    ]

    # Extract hashes and metrics to evaluate
    selective_prediction_results_table = {
        'model': [],
        'description': [],
        'task': [],
        'metric_name': []
    }
    hash_task_name_metric_name_tuples = []
    for row in results_table.iterrows():
        row = row[1]
        hash_task_name_metric_name_tuples.append((
            row['hash'],
            row['task'],
            row['answer_metric_name'])
        )
        selective_prediction_results_table['model'].append(row['model'])
        selective_prediction_results_table['description'].append(row['description'])
        selective_prediction_results_table['task'].append(row['task'])
        selective_prediction_results_table['metric_name'].append(row['answer_metric_name'])
        if row['has_unanswerable_f1']:
            hash_task_name_metric_name_tuples.append((
                row['hash'],
                row['task'],
                'unanswerable_f1')
            )
            selective_prediction_results_table['model'].append(row['model'])
            selective_prediction_results_table['description'].append(row['description'])
            selective_prediction_results_table['task'].append(row['task'])
            selective_prediction_results_table['metric_name'].append('unanswerable_f1')

    # Get selective prediction results using multiprocessing
    with multiprocessing.Pool(processes=8) as pool:
        selective_prediction_results = pool.map(
            get_selective_prediction_results,
            hash_task_name_metric_name_tuples,
        )

    selective_prediction_results_table = pd.DataFrame(selective_prediction_results_table)

    # Extract selective advantage scores and add as column
    selective_prediction_results_table['Selective AUC Advantage'] = [
        result['Selective Advantage']
        for result in selective_prediction_results
    ]
    # Extract class distributions and add as column (each value is a dict)
    selective_prediction_results_table['Class Distributions'] = [
        result['Class Distributions']
        for result in selective_prediction_results
    ]
    # Extract selective scores and add as column (each value is a list)
    selective_prediction_results_table['Selective Scores'] = [
        result['Selective Scores']
        for result in selective_prediction_results
    ]

    # Use short names for model, task and metric name
    selective_prediction_results_table['model'] = selective_prediction_results_table['model'].apply(
        lambda x: MODEL_SHORT_NAMES[x]
    )
    selective_prediction_results_table['task'] = selective_prediction_results_table['task'].apply(
        lambda x: TASK_SHORT_NAMES[x]
    )
    selective_prediction_results_table['metric_name'] = selective_prediction_results_table['metric_name'].apply(
        lambda x: METRIC_SHORT_NAMES[x]
    )

    plot_class_distributions(selective_prediction_results_table)

    plot_selective_scores(selective_prediction_results_table)

    selective_prediction_results_table['Selective AUC Advantage'] = selective_prediction_results_table[
        'Selective AUC Advantage'].apply(
        lambda x: round(x * 100)
    )

    # Pivot table such that there is one column per task with the
    # Selective Advantage Score
    # Use model and description as index
    results_table_pivot = selective_prediction_results_table.pivot(
        index=['model', 'description'],
        columns=['task', 'metric_name'],
        values='Selective AUC Advantage'
    )

    # Order table
    results_table_pivot = results_table_pivot.loc[
        [
            (model_short_name, experiment)
            for model_short_name in MODEL_SHORT_NAMES.values()
            for experiment in EXPERIMENT_ORDER
        ],
        :
    ]

    base_data_path = infer_base_data_path()

    results_table_pivot.to_csv(base_data_path / 'tables' / 'selective_prediction.csv')

    latex_table = format_results_to_latex(results_table_pivot)
    with open(base_data_path / 'tables' / 'selective_prediction.tex', 'w') as f:
        f.write(latex_table)


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


def plot_class_distributions(
        selective_prediction_results_table: pd.DataFrame
):
    tasks = [
        TASK_SHORT_NAMES[task_name]
        for task_name in [
            'qasper',
            'natural_questions',
            'evidence_inference',
            'wice',
            'contract_nli'
        ]
    ]

    sns.set_context('paper')
    # We make two rows of plots to give space for the legend
    fig, axs = plt.subplots(
        2,
        len(tasks),
        figsize=(6.3, 1.9),
        gridspec_kw={'height_ratios': [10, 1]}
    )
    for i, task in enumerate(tasks):
        task_long_name = [
            k for k, v in TASK_SHORT_NAMES.items()
            if v == task
        ][0]
        if task in [TASK_SHORT_NAMES['qasper'], TASK_SHORT_NAMES['natural_questions']]:
            metric_name = METRIC_SHORT_NAMES['unanswerable_f1']
            classes = CLASSES['unanswerable_f1']
            class_short_names = CLASS_SHORT_NAMES['unanswerable_f1']
        else:
            metric_name = METRIC_SHORT_NAMES['classification_f1']
            classes = CLASSES[task_long_name]
            class_short_names = CLASS_SHORT_NAMES[task_long_name]
        # Get data for GPT 3.5
        class_distribution = selective_prediction_results_table.loc[
            (selective_prediction_results_table['task'] == task)
            & (selective_prediction_results_table['metric_name'] == metric_name)
            & (selective_prediction_results_table['model'] == MODEL_SHORT_NAMES['gpt-35-turbo-0613'])
            & (selective_prediction_results_table['description'] == EXPERIMENT_SHORT_NAMES['citation']),
            'Class Distributions'
        ].iloc[0]
        class_distribution = pd.DataFrame(class_distribution)

        # Get colorblind color palette and reverse the order of colors
        palette = sns.color_palette('colorblind', n_colors=10)
        palette = palette[::-1 * (len(classes))]

        # Plot
        sns.lineplot(
            data=class_distribution,
            x='coverage',
            y='prob',
            hue='class_name',
            hue_order=classes,
            ax=axs[0][i],
            markers=False,
            palette=palette
        )
        axs[0][i].set_ylim(0, 1)
        # Hide ytick labels if it is not the first plot in the row
        if i != 0:
            axs[0][i].set_yticklabels([])

        if i == 0:
            axs[0][i].set_ylabel('P(Class)', fontsize=8)
        else:
            axs[0][i].set_ylabel('')

        if i == 2:
            axs[0][i].set_xlabel('Coverage', fontsize=8)
        else:
            # Remove x label
            axs[0][i].set_xlabel('')

        # Put legend below plot using class short names
        handles, labels = axs[0][i].get_legend_handles_labels()
        labels = [
            int(label) if label in ['0', '1'] else label
            for label in labels
        ]
        labels = [class_short_names[label] for label in labels]
        axs[1][i].legend(
            handles,
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 7),
            ncol=1,
            fontsize=6,
        )

        axs[0][i].set_title(task, fontsize=8)

        # Remove legend
        axs[0][i].get_legend().remove()
        plt.tight_layout()
        # Set xticks to integers with 0, 1, 2, 3, 4
        axs[0][i].set_xticks([0.0, 0.5, 1.0])
        # Set xtick fontsize to 6
        axs[0][i].tick_params(labelsize=6)

    for ax in axs[1]:
        ax.axis('off')

    # Move legend down
    fig.canvas.draw()
    plt.tight_layout()
    plt.savefig(infer_base_data_path() / 'plots' / 'selective_prediction_class_dist_plots.pdf')


def plot_selective_scores(
        selective_prediction_results_table: pd.DataFrame
):
    sns.set_context('paper')
    # We make two rows of plots to give space for the legend
    fig, axs = plt.subplots(
        2,
        len(TASK_SHORT_NAMES),
        figsize=(6.3, 2),
        gridspec_kw={'height_ratios': [10, 1]}
    )
    for i, task in enumerate(TASK_SHORT_NAMES.values()):
        task_long_name = [
            k for k, v in TASK_SHORT_NAMES.items()
            if v == task
        ][0]
        metric_long_name = ANSWER_METRICS[task_long_name]
        metric_name = METRIC_SHORT_NAMES[metric_long_name]
        task_df = selective_prediction_results_table.loc[
            (selective_prediction_results_table['task'] == task)
            & (selective_prediction_results_table['metric_name'] == metric_name)
            ]
        selective_scores_table = {
            metric_name: [],
            'coverage': [],
            'model': [],
            'description': []
        }
        for row in task_df.iterrows():
            row = row[1]
            model = row['model']
            description = row['description']
            selective_scores = row['Selective Scores']
            for j in range(len(selective_scores['coverage'])):
                score = selective_scores[metric_long_name][j]
                coverage = selective_scores['coverage'][j]
                selective_scores_table[metric_name].append(score)
                selective_scores_table['coverage'].append(coverage)
                selective_scores_table['model'].append(model)
                selective_scores_table['description'].append(description)

        selective_scores_table = pd.DataFrame(selective_scores_table)
        selective_scores_table[metric_name] = selective_scores_table[metric_name] * 100

        # Use the same colors for the same model and use solid lines for
        # post-hoc and dashed lines for citation
        sns.lineplot(
            data=selective_scores_table,
            x='coverage',
            y=metric_name,
            hue='model',
            style='description',
            ax=axs[0][i],
            markers=False,
        )
        axs[0][i].set_ylim()
        # Hide ytick labels if it is not the first plot in the row
        if i != 0:
            axs[0][i].set_yticklabels([])

        axs[0][i].set_title(task)
        # Remove x label
        axs[0][i].set_xlabel('')
        axs[0][i].set_ylabel('')
        # Remove legend
        axs[0][i].get_legend().remove()
        plt.tight_layout()
        # Set xticks
        axs[0][i].set_xticks([0.0, 0.5, 1.0])

    for ax in axs[1]:
        ax.axis('off')

    # Make a single legend below the plot
    handles, labels = axs[0][0].get_legend_handles_labels()
    handles = [
        handles[i] for i in range(len(handles))
        if labels[i] not in ['model', 'description']
    ]
    labels = [
        labels[i] for i in range(len(labels))
        if labels[i] not in ['model', 'description']
    ]
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0),
        ncol=len(labels),
        fontsize='small'
    )
    # Move legend down
    fig.canvas.draw()
    plt.tight_layout()
    plt.savefig(infer_base_data_path() / 'plots' / 'selective_prediction_scores_plot.pdf')


def get_selective_prediction_results(
        hash_task_name_metric_name: Tuple[str, str]
):
    """Convenience function for multiprocessing"""
    return get_selective_prediction_results_for_hash(
        hash_task_name_metric_name[0],
        hash_task_name_metric_name[1],
        hash_task_name_metric_name[2]
    )


def get_selective_prediction_results_for_hash(
        hash: str,
        task_name: str,
        metric_name: str,
        eval_idx: str | int = 'latest',
        out_dir: Path = None
):
    """Load the results for a specific hash and metric and compute selective
    prediction results using attribution for confidence."""
    if task_name in CLASSES:
        classes = CLASSES[task_name]
        if task_name in ['wice', 'contract_nli']:
            macro_or_micro = 'micro'
        else:
            macro_or_micro = 'macro'
    elif metric_name == 'unanswerable_f1':
        classes = CLASSES[metric_name]
        macro_or_micro = 'micro'
    else:
        classes = None
        macro_or_micro = None

    df = load_predictions_with_scores(hash, metric_name, eval_idx=eval_idx)

    if 'Attribution' in df.columns:
        # Use answerable predictions only, as attribution is not defined for non-answerable
        df = df.loc[
            df['Predicted Answerability'] == 1
            ]
        if metric_name == 'unanswerable_f1':
            gold_answers = df['Gold Answerability']
            predicted_answers = df['Predicted Answerability']
        else:
            gold_answers = df['Gold Answer']
            predicted_answers = df['Predicted Answer']

        selective_prediction_results = selective_prediction(
            'Attribution',
            df['Attribution'].to_list(),
            score_name=metric_name,
            scores=df[metric_name].to_list(),
            out_dir=out_dir,
            gold_answers=gold_answers,
            predicted_answers=predicted_answers,
            classes=classes,
            macro_or_micro=macro_or_micro
        )
    else:
        print(f'Attribution data missing for hash {hash}')
        selective_prediction_results = {
            'Selective Advantage': 0.0,
            f'Selective AUC {metric_name} - Attribution': 0,
            f'AUC {metric_name} - Random': 0
        }

    return selective_prediction_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hash',
        type=str,
        default='all'
    )
    parser.add_argument(
        '--task_name',
        type=str,
        default=''
    )
    parser.add_argument(
        '--eval_idx',
        type=str,
        default='latest'
    )
    args = parser.parse_args()

    if args.hash == 'all':
        main()
    else:
        eval_idx = args.eval_idx
        if eval_idx != 'latest':
            eval_idx = int(args.eval_idx)
        base_data_path = infer_base_data_path()
        get_selective_prediction_results_for_hash(
            args.hash,
            args.task_name,
            eval_idx,
            out_dir=base_data_path
        )
