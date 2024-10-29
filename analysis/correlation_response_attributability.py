import pandas as pd
from scipy.stats import pearsonr

from analysis.util import load_results_table, MODEL_ORDER, TASK_SHORT_NAMES, infer_base_data_path

def main():
    results_table = load_results_table()

    results_table = results_table.loc[
        (results_table['description'].isin(['citation', 'reduced-citation']))
        & (results_table['model'].isin(MODEL_ORDER))
    ]

    task_correlations = {
        'task': [],
        'metric_name': [],
        'value': []
    }

    for task_name, task_short_name in TASK_SHORT_NAMES.items():
        task_table = results_table.loc[
            results_table['task'] == task_name
        ]
        evidence_scores = task_table.groupby(['model', 'description'])['attribution_metric'].mean()
        response_scores = task_table.groupby(['model', 'description'])['answer_metric'].mean()
        correlation = pearsonr(evidence_scores, response_scores)
        # Store in task_correlations
        task_correlations['task'].append(task_short_name)
        task_correlations['metric_name'].append('r')
        task_correlations['value'].append(correlation[0])
        task_correlations['task'].append(task_short_name)
        task_correlations['metric_name'].append('p')
        task_correlations['value'].append(correlation[1])

    mean_evidence_scores = results_table.groupby(['model', 'description'])['attribution_metric'].mean()
    mean_response_scores = results_table.groupby(['model', 'description'])['answer_metric'].mean()

    mean_correlation = pearsonr(mean_evidence_scores, mean_response_scores)

    # add to task_correlations
    task_correlations['task'].append('mean')
    task_correlations['metric_name'].append('r')
    task_correlations['value'].append(mean_correlation[0])
    task_correlations['task'].append('mean')
    task_correlations['metric_name'].append('p')
    task_correlations['value'].append(mean_correlation[1])

    correlation_table = pd.DataFrame(task_correlations)

    # Pivot to get p and r as separate columns
    correlation_table = correlation_table.pivot(
        index='task',
        columns=['metric_name'],
        values='value'
    )
    correlation_table = correlation_table.reset_index()

    # combine into single column with p in brackets
    # Format r to str with 2 digits
    correlation_table['r'] = correlation_table['r'].apply(lambda x: '${:.2f}$'.format(x))
    # Format p to pretty str ($X.Y\times10^{Z}$
    correlation_table['p'] = correlation_table['p'].apply(lambda x: '{:.1e}'.format(x))
    correlation_table['p'] = correlation_table['p'].apply(lambda x: f' (${x[:3]} \\times 10^{{{int(x[-3:])}}}$)')
    correlation_table['r'] = correlation_table['r'] + correlation_table['p']

    correlation_table = correlation_table[['task','r']].set_index('task')
    row_order = [task_name for task_name in TASK_SHORT_NAMES.values()] + ['mean']
    correlation_table = correlation_table.loc[row_order]

    # Format to latex, format floats with 2 decimals
    correlation_table_latex = correlation_table.to_latex(
        index=True,
        escape=False,
        column_format='l|c'
    )
    # Write out
    base_data_path = infer_base_data_path()
    with open(base_data_path / 'tables' / 'correlation_table.tex', 'w') as f:
        f.write(correlation_table_latex)

    # print table
    print(correlation_table)

if __name__ == '__main__':
    main()