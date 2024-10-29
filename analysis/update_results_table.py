import json
import os
from typing import Dict
import sys

import numpy as np
import pandas as pd

sys.path.append('..')

from analysis.util import (
    infer_base_data_path,
    load_results_table,
    MODEL_ORDER,
    TASK_ORDER,
    MODEL_SHORT_NAMES,
    TASK_SHORT_NAMES,
    METRIC_SHORT_NAMES
)

MODEL_SHORT_NAMES = {
    k: v
    for k, v in MODEL_SHORT_NAMES.items()
    if k in MODEL_ORDER
}

EXPERIMENT_SHORT_NAMES = {
    'post-hoc': '\\texttt{p-h}',
    'retrieve-then-read': '\\texttt{rtr}',
    'citation': '\\texttt{cit}',
    'reduced-post-hoc': '\\texttt{r-p-h}',
    'reduced-citation': '\\texttt{r-cit}'
}


def get_results(
        results_dict: Dict,
        answer_metric_name: str,
        attribution_metric_name: str,
        has_unanswerable_f1: bool
):
    answer_metric_score = results_dict[answer_metric_name]['score']
    if attribution_metric_name == 'attribution':
        try:
            attribution_metric_score = results_dict['attribution']['score']
        except KeyError:
            attribution_metric_score = 0
    elif attribution_metric_name == 'evidence_f1':
        attribution_metric_score = results_dict[answer_metric_name]['evidence_f1']
    else:
        raise NotImplementedError

    if has_unanswerable_f1:
        unanswerable_f1_score = np.mean([
            results_dict['unanswerable_f1']['score_by_answer_type']['f1']['unanswerable'],
            results_dict['unanswerable_f1']['score_by_answer_type']['f1']['answerable']
        ])
    else:
        unanswerable_f1_score = None

    return {
        answer_metric_name: answer_metric_score,
        attribution_metric_name: attribution_metric_score,
        'unanswerable_f1': unanswerable_f1_score
    }


def main():
    base_data_path = infer_base_data_path()
    results_dir_path = base_data_path / 'results'

    results_table = load_results_table()

    for row in results_table.iterrows():
        row = row[1]
        hash = row['hash']
        if pd.isna(hash):
            continue
        answer_metric_name = row['answer_metric_name']
        attribution_metric_name = row['attribution_metric_name']
        has_unanswerable_f1 = row['has_unanswerable_f1']
        hash_dir_path = results_dir_path / hash

        eval_dir_idcs = [
            int(dirname)
            for dirname in os.listdir(hash_dir_path / 'evaluation')
        ]

        latest_eval_dir_idx = max(eval_dir_idcs)
        while len(os.listdir(hash_dir_path / 'evaluation' / str(latest_eval_dir_idx))) == 0:
            latest_eval_dir_idx -= 1

        latest_eval_dir_idx = str(latest_eval_dir_idx)

        eval_dir_path = hash_dir_path / 'evaluation' / latest_eval_dir_idx

        with open(eval_dir_path / 'results.json') as f:
            results_dict = json.load(f)

        results = get_results(
            results_dict,
            answer_metric_name,
            attribution_metric_name,
            has_unanswerable_f1
        )

        results_table.loc[results_table['hash'] == hash, answer_metric_name] = results[answer_metric_name]
        results_table.loc[results_table['hash'] == hash, attribution_metric_name] = results[attribution_metric_name]
        results_table.loc[results_table['hash'] == hash, 'answer_metric'] = results[answer_metric_name]
        results_table.loc[results_table['hash'] == hash, 'attribution_metric'] = results[attribution_metric_name]

        if has_unanswerable_f1:
            results_table.loc[results_table['hash'] == hash, 'unanswerable_f1'] = results['unanswerable_f1']

    results_table.to_csv(base_data_path / 'results.csv', index=False)

    format_results_table(results_table)



def format_results_table(results_table: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot results table to get
    :param results_table:
    :return:
    """
    results_table = results_table.loc[
        results_table['model'].isin(MODEL_ORDER)
    ]

    task_dfs = []
    answer_score_column_names = []
    attribution_score_column_names = []
    task_metrics = {}

    for task_name in TASK_ORDER:
        task_print_name = TASK_SHORT_NAMES[task_name]
        task_df = results_table.loc[results_table['task'] == task_name]
        answer_metric_name = task_df.iloc[0]['answer_metric_name']
        attribution_metric_name = task_df.iloc[0]['attribution_metric_name']
        has_unanswerable_f1 = task_df.iloc[0]['has_unanswerable_f1']
        if has_unanswerable_f1:
            metrics = [answer_metric_name, attribution_metric_name, 'unanswerable_f1']
        else:
            metrics = [answer_metric_name, attribution_metric_name]
        task_metrics[task_name] = metrics
        for metric_name in metrics:
            task_df[metric_name] = task_df[metric_name].apply(
                lambda x: round(x*100)
            )
        metric_column_names = {
            metric_name: f'{task_print_name}/{METRIC_SHORT_NAMES[metric_name]}'
            for metric_name in metrics
        }
        answer_score_column_names.append(f'{task_print_name}/{METRIC_SHORT_NAMES[answer_metric_name]}')
        attribution_score_column_names.append(f'{task_print_name}/{METRIC_SHORT_NAMES[attribution_metric_name]}')
        task_df = task_df[['model', 'description'] + metrics]
        task_df = task_df.rename(columns=metric_column_names)
        task_dfs.append(task_df)

    df = task_dfs[0]
    for task_df in task_dfs[1:]:
        df = pd.merge(
            df,
            task_df,
            how='left',
            on=['model', 'description']
        )

    df['model'] = df['model'].apply(lambda x: MODEL_SHORT_NAMES[x])
    df['description'] = df['description'].apply(lambda x: EXPERIMENT_SHORT_NAMES[x])

    # Rename columns "model" and "description" to "/model" and "/description"
    df = df.rename(columns={
        'model': 'Model',
        'description': 'Description'
    })

    # Compute average answer score metric and attribution metric
    df['Avg/RQ'] = df[answer_score_column_names].mean(axis=1)
    df['Avg/EQ'] = df[attribution_score_column_names].mean(axis=1)
    task_metrics['Avg'] = ['RQ', 'EQ']

    # Make row multiindex with levels "model", "description"
    df = df.set_index(['Model', 'Description'])

    # Sort rows by model and experiment order
    df = df.loc[
        [
            (model_short_name, experiment_short_name)
            for model_short_name in MODEL_SHORT_NAMES.values()
            for experiment_short_name in EXPERIMENT_SHORT_NAMES.values()
        ],
        :
    ]

    # Make column multiindex by splitting at "/"
    df.columns = pd.MultiIndex.from_tuples(
        [tuple(col.split('/')) for col in df.columns]
    )

    base_data_path = infer_base_data_path()
    df.to_csv(base_data_path / 'tables' / 'main_table.csv')

    # Bold highest value for each combination of task, metric and model
    for task_name, metrics in task_metrics.items():
        if task_name != 'Avg':
            task_name = TASK_SHORT_NAMES[task_name]
        for metric_name in metrics:
            if metric_name not in ['RQ', 'EQ']:
                metric_name = METRIC_SHORT_NAMES[metric_name]
            for model_name in MODEL_SHORT_NAMES.values():
                max_val = df.loc[
                    (model_name, slice(None)),
                    (task_name, metric_name)
                ].max()
                if task_name != 'Avg':
                    df.loc[
                        (model_name, slice(None)),
                        (task_name, metric_name)
                    ] = df.loc[
                        (model_name, slice(None)),
                        (task_name, metric_name)
                    ].apply(lambda x: f'\\textbf{{{x}}}' if x == max_val else f'{x}')
                else:
                    df.loc[
                        (model_name, slice(None)),
                        (task_name, metric_name)
                    ] = df.loc[
                        (model_name, slice(None)),
                        (task_name, metric_name)
                    ].apply(lambda x: f'\\textbf{{{x:.2f}}}' if x == max_val else f'{x:.2f}')

    latex_table = format_results_to_latex(df)
    with open(base_data_path / 'tables' / 'main_table.tex', 'w') as f:
        f.write(latex_table)

    return


def format_results_to_latex(formatted_results_table):
    # Format numbers as ".90"
    latex_str = formatted_results_table.to_latex(
        escape=False,
        multirow=True,
        multicolumn_format='c',
        column_format='l' + 'c' * len(formatted_results_table.columns.levels[1]),
        bold_rows=True,
        float_format="%.2f"
    )

    return latex_str


if __name__ == '__main__':
    main()