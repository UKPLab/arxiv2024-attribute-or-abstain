import json
import os
from pathlib import Path
from typing import Tuple
import sys

import yaml

sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

from analysis.util import (
    load_predictions_with_scores,
    load_results_table,
    infer_base_data_path,
    MODEL_ORDER,
    MODEL_SHORT_NAMES,
    TASK_SHORT_NAMES,
    EXPERIMENT_SHORT_NAMES,
    load_predictions_for_hash
)
from config_lib.base_config import BaseConfig
from evaluation.common import Statistics, BaseTask, BaseInstance
from evaluation.tasks.contract_nli_task import ContractNLITask
from evaluation.tasks.evidence_inference_task import EvidenceInferenceTask
from evaluation.tasks.govreport_task import GovReportTask
from evaluation.tasks.natural_questions_task import NaturalQuestionsTask
from evaluation.tasks.qasper_task import QASPERTask
from evaluation.tasks.wice_task import WiceTask

TASKS = [
    'qasper',
    'natural_questions',
    'evidence_inference',
    'wice',
    'contract_nli',
    'govreport'
]

EXPERIMENTS = [
    'citation',
    'post-hoc'
]

TASK_CLASSES = [
    QASPERTask,
    NaturalQuestionsTask,
    EvidenceInferenceTask,
    GovReportTask,
    WiceTask,
    ContractNLITask
]

N_BINS = 5


def main():
    """Compute performance dependent on position of annotated gold evidence"""

    results_table = load_results_table()
    # Use only certain models, experiments and tasks
    results_table = results_table.loc[
        results_table['model'].isin(MODEL_SHORT_NAMES)
    ]
    results_table = results_table.loc[
        results_table['description'].isin(EXPERIMENTS)
    ]
    results_table = results_table.loc[
        results_table['task'].isin(TASKS)
    ]

    # Make big table that combines predictions from all runs
    table = {
        'model': [],
        'task': [],
        'description': [],
        'attribution_metric_name': [],
        'answer_metric_name': [],
        'evidence_quality': [],
        'response_quality': [],
        'n_words': []
    }
    # Store instances here
    instances_cache: dict[str, list[BaseInstance]] = {}
    # Store number of words for the instances here
    n_words: dict[str, dict[str, int]] = {}
    for row in results_table.iterrows():
        row = row[1]
        hash = row['hash']
        task_name = row['task']
        model_name = row['model']
        description = row['description']
        answer_metric_name = row['answer_metric_name']
        attribution_metric_name = row['attribution_metric_name']
        if attribution_metric_name == 'attribution':
            attribution_metric_name = 'Attribution'
        elif attribution_metric_name == 'evidence_f1':
            attribution_metric_name = 'Evidence F1'

        if task_name not in instances_cache:
            instances_cache[task_name] = load_instances_for_hash(hash)
            n_words[task_name] = get_n_words_per_doc(instances_cache[task_name])

        # Load table of predictions with scores
        predictions_with_scores = load_predictions_with_scores(hash, answer_metric_name)

        # Load predictions (to get example ids)
        predictions = load_predictions_for_hash(
            hash,
            instances_cache[task_name],
            eval_idx='original'
        )

        # Add column with example_ids to predictions_with_scores
        predictions_with_scores['example_id'] = [
            prediction.example_id
            for prediction in predictions
        ]

        # Add document lengths
        predictions_with_scores['n_words'] = predictions_with_scores['example_id'].apply(
            lambda x: n_words[task_name][x]
        )

        # Remove rows where 'Predicted Answer' == 'unanswerable'
        predictions_with_scores = predictions_with_scores.loc[
            predictions_with_scores['Predicted Answer'] != 'unanswerable'
        ]

        # Add to table
        table['model'].extend([model_name for _ in range(len(predictions_with_scores))])
        table['task'].extend([task_name for _ in range(len(predictions_with_scores))])
        table['description'].extend([description for _ in range(len(predictions_with_scores))])
        table['attribution_metric_name'].extend([attribution_metric_name for _ in range(len(predictions_with_scores))])
        table['answer_metric_name'].extend([answer_metric_name for _ in range(len(predictions_with_scores))])
        table['evidence_quality'].extend(predictions_with_scores[attribution_metric_name])
        table['response_quality'].extend(predictions_with_scores[answer_metric_name])
        table['n_words'].extend(predictions_with_scores['n_words'])

    table = pd.DataFrame(table)


    # Plot
    make_lineplots(table, 'evidence_quality', 'Evidence Quality')
    make_lineplots(table, 'response_quality', 'Response Quality')

    # Compute correlation and output table
    make_correlation_table(table, 'evidence_quality')
    make_correlation_table(table, 'response_quality')

    # Make tables with bins
    all_tasks_mean_evidence_quality_df = make_tables_with_bins(table, 'evidence_quality')
    all_tasks_mean_response_quality_df = make_tables_with_bins(table, 'response_quality')

    # Make plots with bins
    make_plots_with_bins(
        all_tasks_mean_evidence_quality_df,
        'evidence_quality',
        'Evidence Quality'
    )
    make_plots_with_bins(
        all_tasks_mean_response_quality_df,
        'response_quality',
        'Response Quality'
    )


def make_correlation_table(
        table: pd.DataFrame,
        performance_col_name: str
):
    """For each combination of description, task and model,
    compute correlation of performance and document length
    Put into one table with tasks in columns and models
    and descriptions in rows."""

    print('Computing correlation table')

    correlation_table = {
        'model': [],
        'description': [],
        'task': [],
        'r': [],
        'p': []
    }

    for task in TASKS:
        task_table = table.loc[
            table['task'] == task
        ]
        for model in MODEL_SHORT_NAMES:
            if model not in MODEL_ORDER:
                continue
            for description in EXPERIMENTS:
                model_description_table = task_table.loc[
                    (task_table['model'] == model) &
                    (task_table['description'] == description)
                ]
                try:
                    r, p = pearsonr(
                        model_description_table[performance_col_name],
                        model_description_table['n_words']
                    )
                except ValueError as e:
                    print(f'ValueError for {model} {description} {task}')
                    raise e
                correlation_table['model'].append(model)
                correlation_table['description'].append(description)
                correlation_table['task'].append(task)
                correlation_table['r'].append(r)
                correlation_table['p'].append(p)

    correlation_table = pd.DataFrame(correlation_table)
    # Rename model, description and task
    correlation_table['model'] = correlation_table['model'].apply(
        lambda x: MODEL_SHORT_NAMES[x]
    )
    correlation_table['description'] = correlation_table['description'].apply(
        lambda x: EXPERIMENT_SHORT_NAMES[x]
    )
    correlation_table['task'] = correlation_table['task'].apply(
        lambda x: TASK_SHORT_NAMES[x]
    )
    # Round 'r' to two decimals and add '*' if 'p' < 0.05
    correlation_table['r'] = correlation_table.apply(
        lambda x: f'{x["r"]:.2f}' if x["p"] > 0.05 else f'{x["r"]:.2f}*',
        axis=1
    )
    # Pivot to get one column per task
    correlation_table = correlation_table.pivot(
        index=['model', 'description'],
        columns='task',
        values=['r']
    )
    # Sort rows
    correlation_table = correlation_table.loc[
        [
            (MODEL_SHORT_NAMES[model_name], EXPERIMENT_SHORT_NAMES[description])
            for model_name in MODEL_ORDER
            for description in EXPERIMENTS
        ],
        :
    ]
    # Sort columns by task order
    correlation_table = correlation_table[
        [('r', TASK_SHORT_NAMES[task_name]) for task_name in TASKS]
    ]
    base_data_path = infer_base_data_path()
    correlation_table.to_csv(base_data_path / 'tables' / f'{performance_col_name}_vs_doc_length_correlation.csv')

    # Save table to tex
    correlation_table.to_latex(
        base_data_path / 'tables' / f'{performance_col_name}_vs_doc_length_correlation.tex',
        escape=False
    )

def make_lineplots(
        table: pd.DataFrame,
        performance_col_name: str,
        y_label: str
):
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
    table['model_description'] = table['model'] + ' ' + table['description']
    for i, task in enumerate(TASKS):
        task_df = table.loc[
            table['task'] == task
        ]
        # Use the same colors for the same model and use solid lines for
        # full-post-hoc and dashed lines for full-citation
        sns.lineplot(
            data=task_df,
            x='n_words',
            y=performance_col_name,
            hue='model',
            style='description',
            ax=axs[0][i]
        )
        # axs[0][i].set_ylim(10, 95)
        # Hide ytick labels if it is not the first plot in the row
        if i != 0:
            axs[0][i].set_yticklabels([])

        axs[0][i].set_title(TASK_SHORT_NAMES[task], fontsize=8)
        axs[0][i].set_xlabel('')
        if i == 0:
            axs[0][i].set_ylabel(y_label, fontsize=8)
        else:
            # Remove x label
            axs[0][i].set_ylabel('')
        # Remove legend
        axs[0][i].get_legend().remove()
        plt.tight_layout()
        # Set xticks to integers with 0, 1, 2, 3, 4
        # axs[0][i].set_xticks(range(N_BINS))

        # Set tick fontsize to 6
        axs[0][i].tick_params(axis='both', which='major', labelsize=6)
    for ax in axs[1]:
        ax.axis('off')

    fig.text(
        0.5,
        0.15,
        'Document Length',
        ha='center',
        fontsize=8
    )
    # Make a single legend below the plot
    handles, labels = axs[0][0].get_legend_handles_labels()
    handles = [
        handles[i] for i in range(len(handles))
        if labels[i] in MODEL_SHORT_NAMES
    ]
    labels = [
        MODEL_SHORT_NAMES[labels[i]] for i in range(len(labels))
        if labels[i] in MODEL_SHORT_NAMES
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
    base_data_path = infer_base_data_path()
    plt.savefig(base_data_path / 'plots' / f'{performance_col_name}_vs_doc_length.pdf')


def make_tables_with_bins(
        table: pd.DataFrame,
        performance_col_name: str
):
    """For each task, make a table showing the evidence quality for
    different ranges of document lengths (measured as the number of
    words)"""
    n_bins = N_BINS

    all_tasks_mean_performance_df = pd.DataFrame(
        columns=[
            'model', 'description', 'bin', performance_col_name, 'task'
        ]
    )

    for task in TASKS:
        task_df = table.loc[
            table['task'] == task
        ]
        # Only use the models from MODEL_ORDER
        task_df = task_df.loc[
            task_df['model'].isin(MODEL_ORDER)
        ]

        # Split rows into n_bins equally sized bins based on n_words
        task_df['bin'] = pd.qcut(task_df['n_words'], n_bins, labels=False)

        # Get string bin labels by getting the max and mean for each bin
        bin_labels = []
        for bin in range(n_bins):
            bin_df = task_df.loc[
                task_df['bin'] == bin
            ]
            bin_labels.append(bin_df["n_words"].min())
        # Update bin labels
        task_df['bin'] = task_df['bin'].apply(lambda x: bin_labels[x])

        # Group by model, description and bin and compute mean performance for each group
        grouped = task_df.groupby(['model', 'description', 'bin'])
        mean_performance = grouped[performance_col_name].mean()

        # Restore index
        mean_performance = mean_performance.reset_index()

        # Rename model and description
        mean_performance['model'] = mean_performance['model'].apply(
            lambda x: MODEL_SHORT_NAMES[x]
        )
        mean_performance['description'] = mean_performance['description'].apply(
            lambda x: EXPERIMENT_SHORT_NAMES[x]
        )

        mean_performance['task'] = task

        all_tasks_mean_performance_df = pd.concat(
            [
                all_tasks_mean_performance_df,
                mean_performance
            ]
        )

        # Make one column per bin
        mean_performance = mean_performance.pivot(
            index=['model', 'description'],
            columns='bin',
            values=performance_col_name
        )
        # Sort by model and description
        mean_performance = mean_performance.loc[
            [
                (MODEL_SHORT_NAMES[model_name], EXPERIMENT_SHORT_NAMES[description])
                for model_name in MODEL_ORDER
                for description in EXPERIMENTS
            ],
            :
        ]
        # Sort columns ascending
        mean_performance = mean_performance.sort_index(axis=1)

        # Store table as csv
        base_data_path = infer_base_data_path()
        mean_performance.to_csv(base_data_path / 'tables' / f'{task}_{performance_col_name}_vs_doc_length.csv')

        # Pretty print table
        print(task)
        print(mean_performance)

    return all_tasks_mean_performance_df


def make_plots_with_bins(
        all_tasks_mean_performance_df: pd.DataFrame,
        performance_col_name: str,
        y_label: str
):
    """
    Make plot with len(TASKS) subplots in one row. Each subplot is a lineplot
    showing performance for each bin
    :param all_tasks_mean_performance_df: dataframe with columns 'model',
        'description', 'bin', performance_col_name, 'task'
    :return:
    """
    all_tasks_mean_performance_df[performance_col_name] = (
        all_tasks_mean_performance_df[performance_col_name] * 100
    )
    # We make two rows of plots to give space for the legend
    sns.set_context('paper')
    fig, axs = plt.subplots(
        2,
        len(TASKS),
        figsize=(6.3, 1.8),
        gridspec_kw={'height_ratios': [10, 1]}
    )
    for i, task in enumerate(TASKS):
        task_df = all_tasks_mean_performance_df.loc[
            all_tasks_mean_performance_df['task'] == task
        ]
        # Use the same colors for the same model and use solid lines for
        # full-post-hoc and dashed lines for full-citation
        sns.lineplot(
            data=task_df,
            x='bin',
            y=performance_col_name,
            hue='model',
            style='description',
            markers=True,
            hue_order=[
                MODEL_SHORT_NAMES[model_name]
                for model_name in MODEL_ORDER
            ],
            ax=axs[0][i],
            palette='colorblind'
        )
        axs[0][i].set_ylim(0, 100)
        # Hide ytick labels if it is not the first plot in the row
        if i != 0:
            axs[0][i].set_yticklabels([])

        axs[0][i].set_title(TASK_SHORT_NAMES[task], fontsize=8)
        axs[0][i].set_xlabel('')
        if i == 0:
            axs[0][i].set_ylabel(y_label, fontsize=8)
        else:
            # Remove x label
            axs[0][i].set_ylabel('')
        # Remove legend
        axs[0][i].get_legend().remove()
        plt.tight_layout()
        # Set xticks to to bins
        bins_sorted = sorted(task_df['bin'].unique().tolist())
        axs[0][i].set_xticks(
            bins_sorted
        )
        axs[0][i].set_xticklabels(axs[0][i].get_xticks(), ha='right')
        # Set tick fontsize to 6
        axs[0][i].tick_params(
            axis='both',
            which='major',
            labelsize=6,
        )
        axs[0][i].tick_params(
            axis='x',
            pad=0,
            rotation=45
        )
    for ax in axs[1]:
        ax.axis('off')

    fig.text(
        0.5,
        0.15,
        'Mean Document Length',
        ha='center',
        fontsize=8
    )
    # Make a single legend below the plot
    handles, labels = axs[0][0].get_legend_handles_labels()
    valid_labels = (
        list(MODEL_SHORT_NAMES.values())
        + list(EXPERIMENT_SHORT_NAMES.values())
    )
    handles = [
        handles[i] for i in range(len(handles))
        if labels[i] in valid_labels
    ]
    labels = [
        labels[i] for i in range(len(labels))
        if labels[i] in valid_labels
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
    base_data_path = infer_base_data_path()
    plt.savefig(base_data_path / 'plots' / f'{performance_col_name}_vs_doc_length_bins.pdf')

def load_instances_for_hash(
        hash: str
) -> list[BaseInstance]:
    """Load dev or test instances depending on config"""
    base_data_path = infer_base_data_path()
    # Load config
    config_path = base_data_path / 'results' / hash / 'config.json'
    with open(config_path, 'r') as f:
        config_json = json.load(f)
    config: BaseConfig = BaseConfig.from_dict(config_json, Path('../config'))
    # Override location config
    if not os.path.exists(config.location.datasets):
        # Use local location config
        local_location_config_path = '../config/location/local.yaml'
        # Load into dict
        with open(local_location_config_path, 'r') as f:
            local_location_config = yaml.safe_load(f)
        # Override and convert to paths
        for key in config.location:
            config.location[key] = Path(local_location_config[key])
    # Load task
    for tc in TASK_CLASSES:
        if config.task.task_name == tc.task_name:
            task_class = tc
        # Initialize Statistics object
    stats = Statistics(
        config.model.model_name,
        config.task.task_name,
        config.description,
        config
    )
    # Load task
    task: BaseTask = task_class(config, stats)
    # Hack: Convert dev instances object to list to load all instances into
    # memory (for speed-up when re-using the dataset)
    if config.use_dev_as_test_data:
        instances = task.dev_instances
    else:
        instances = task.test_instances
    return instances


def get_n_words_per_doc(
        instances: list[BaseInstance]
) -> dict[str, int]:
    """Count the number of words in the document for each instance"""
    n_words_dict = {}
    for instance in instances:
        n_words_for_instance = 0
        for node in instance.document.nodes:
            n_words_for_instance += len(node.content.split())
        n_words_dict[instance.example_id] = n_words_for_instance

    return n_words_dict



if __name__ == '__main__':
    main()
