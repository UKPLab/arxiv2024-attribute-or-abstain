import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import sklearn.metrics
from matplotlib import pyplot as plt

from evaluation.common import BaseTask, BaseInstance
from evaluation.metrics import find_label

TASK_ORDER = [
    'qasper',
    'natural_questions',
    'evidence_inference',
    'wice',
    'contract_nli',
    'govreport'
]

MODEL_ORDER = [
    'gpt-35-turbo-0613',
    'gpt-4-turbo-128k',
    'flan-t5-xl',
    'longchat-7b-v1.5-32k',
    'Mistral-7B-Instruct-v0.2'
]

EXPERIMENT_ORDER = [
    'post-hoc',
    'retrieve-then-read',
    'citation',
    'reduced-post-hoc',
    'reduced-citation'
]

ANSWER_METRICS = {
    'qasper': 'answer_f1',
    'natural_questions': 'answer_f1',
    'evidence_inference': 'classification_f1',
    'wice': 'classification_f1',
    'contract_nli': 'classification_f1',
    'govreport': 'rouge_l'
}

ATTRIBUTION_METRICS = {
    'qasper': 'attribution',
    'natural_questions': 'attribution',
    'evidence_inference': 'evidence_f1',
    'wice': 'evidence_f1',
    'contract_nli': 'evidence_f1',
    'govreport': 'attribution'
}

CLASSES = {
    'evidence_inference':
        [
            'significantly increased',
            'significantly decreased',
            'no significant difference'
        ],
    'wice': [
        'not supported',
        'partially supported',
        'supported'
    ],
    'contract_nli': [
        'not mentioned',
        'entailment',
        'contradiction'

    ],
    'unanswerable_f1':
    [
        0,
        1
    ]
}

CLASS_SHORT_NAMES = {
    'evidence_inference': {
        'significantly increased': 'Incr',
        'significantly decreased': 'Decr',
        'no significant difference': 'No diff'
    },
    'wice': {
        'not supported': 'Not sup',
        'partially supported': 'Part sup',
        'supported': 'Sup'
    },
    'contract_nli': {
        'not mentioned': 'Not ment',
        'entailment': 'Ent',
        'contradiction': 'Contr'
    },
    'unanswerable_f1': {
        0: 'Unans',
        1: 'Ans'
    }
}

TASK_PRETTY_NAMES = {
    'qasper': 'QASPER',
    'natural_questions': 'Natural Questions',
    'evidence_inference': 'Evidence Inference',
    'wice': 'WICE',
    'contract_nli': 'Contract NLI',
    'govreport': 'GovReport'
}

TASK_SHORT_NAMES = {
    'qasper': 'QSP',
    'natural_questions': 'NQ',
    'evidence_inference': 'EI',
    'wice': 'WIC',
    'contract_nli': 'CNLI',
    'govreport': 'GR'
}

METRIC_SHORT_NAMES = {
    'answer_f1': 'AF1',
    'classification_f1': 'CF1',
    'rouge_l': 'RL',
    'evidence_f1': 'EF1',
    'attribution': 'ATT',
    'unanswerable_f1': 'UF1'
}

MODEL_SHORT_NAMES = {
    'gpt-35-turbo-0613': 'GPT-3.5',
    'gpt-4-turbo-128k': 'GPT-4',
    'flan-t5-xl': 'Flan-T5',
    'longchat-7b-v1.5-32k': 'Longchat',
    'Mistral-7B-Instruct-v0.2': 'Mistral'
}

EXPERIMENT_SHORT_NAMES = {
    'post-hoc': 'post-hoc',
    'retrieve-then-read': 'retrieve-then-read',
    'citation': 'citation',
    'reduced-post-hoc': 'reduced-post-hoc',
    'reduced-citation': 'reduced-citation'
}

def infer_base_data_path() -> Path:
    path = Path('../../data')

    return path

def load_results_table() -> pd.DataFrame:
    base_data_path = infer_base_data_path()
    table_path = base_data_path / 'results.csv'

    table = pd.read_csv(table_path)

    # Make new column "answer_metric" that is filled depending on the value
    # in column "task" and the metric defined in ANSWER_METRICS
    table['answer_metric'] = table.apply(
        lambda row: row[ANSWER_METRICS[row['task']]],
        axis=1
    )
    # Make new column "attribution_metric" that is filled depending on the value
    # in column "task" and the metric defined in ATTRIBUTION_METRICS
    table['attribution_metric'] = table.apply(
        lambda row: row[ATTRIBUTION_METRICS[row['task']]],
        axis=1
    )

    return table


def find_latest_eval_dir(
        path: Path
) -> int:
    eval_dirnames = os.listdir(path)
    eval_dir_idcs = []
    for dirname in eval_dirnames:
        try:
            int_dirname = int(dirname)
            eval_dir_idcs.append(int_dirname)
        except ValueError:
            continue
    latest_dir_idx = max(eval_dir_idcs)
    while not os.listdir(path / str(latest_dir_idx)):
        # Handle empty directories by jumping to the previous
        latest_dir_idx -= 1
        if latest_dir_idx == 0:
            raise FileNotFoundError

    return latest_dir_idx


def load_predictions_with_scores(
        hash: str,
        score_name: str,
        eval_idx: int | str = 'latest',
) -> pd.DataFrame:
    def load_json(
            s: str
    ):
        try:
            ret = json.loads(s)
        except json.decoder.JSONDecodeError:
            pass
        return ret

    base_data_path = infer_base_data_path() / 'results'

    if eval_idx == 'latest':
        latest_dir_idx = find_latest_eval_dir(
            base_data_path / hash / 'evaluation'
        )
        eval_idx = latest_dir_idx

    table_path = base_data_path / hash / 'evaluation' / str(eval_idx) / f'predictions_{score_name}.csv'

    table = pd.read_csv(table_path)

    # Convert json columns to objects
    json_columns = [
        'Predicted Extraction Node Idxs',
        'Predicted Extraction Node Ixs',
        'Gold Extraction Node Idxs',
        'Gold Extraction Node Ixs'
    ]
    for column in json_columns:
        if column in table.columns:
            # replace ' with "
            table[column] = table[column].str.replace("'", '"')
            table[column] = table[column].apply(load_json)

    return table


def load_predictions_for_hash(
        hash: str,
        instances: list[BaseInstance],
        eval_idx: int | str = 'latest',
):
    base_data_path = infer_base_data_path() / 'results'
    if eval_idx == 'original':
        predictions_path = base_data_path / hash / 'predictions.jsonl'
    else:
        if eval_idx == 'latest':
            latest_dir_idx = find_latest_eval_dir(
                base_data_path / hash / 'evaluation'
            )
            eval_idx = latest_dir_idx
        predictions_path = (
            base_data_path
            / hash
            / 'evaluation'
            / str(eval_idx)
            / 'predictions.jsonl'
        )
    predictions = BaseTask.load_predictions(
        predictions_path,
        instances
    )
    return predictions


def selective_prediction(
        confidence_name: str,
        confidence: List[float] | pd.Series,
        score_name: str = None,
        scores: List[float] | pd.Series = None,
        out_dir: Path = None,
        gold_answers: List[str] | pd.Series = None,
        predicted_answers: List[str] | pd.Series = None,
        classes: List[str] = None,
        macro_or_micro: str = None

):
    """Compute Area under Score - Coverage curve and output plot.
    Sort samples according to confidence. Assuming samples are indexed by i,
    compute mean score for each selection of samples with confidence of sample
    i or higher. Plot against coverage (number of samples for which mean score is
    computed).
    Compute the area under the curve (selective-AUC).
    """

    def compute_mean_scores_on_subsets(scores_) -> List[float]:
        """Given an iterable of scores of length n, compute the mean score for
        each subset [0:k] for all 0 < k <= n."""
        mean_scores = []
        n_samples_seen = 0
        running_avg = 0
        for score in scores_:
            running_avg = running_avg * n_samples_seen
            n_samples_seen += 1
            running_avg = (running_avg + score) / n_samples_seen
            mean_scores.append(running_avg)
        return mean_scores

    def compute_classification_f1_on_subsets(
            gold_answers_: pd.Series,
            predicted_answers_: pd.Series,
            classes_: List[int],
            macro_or_micro_: str
    ) -> List[float]:
        """Given two series of gold answers and predicted answers both of length n,
        compute classification macro f1 for each subset [0:k] for all 0 < k <= n"""
        scores = []
        for i in range(len(gold_answers_)):
            f1 = sklearn.metrics.f1_score(
                gold_answers_[:i+1],
                predicted_answers_[:i+1],
                labels=classes_,
                average=macro_or_micro_
            )
            scores.append(f1)
        return scores

    def compute_class_distribution_on_subsets(
            answers_: pd.Series,
            classes_: List[str]
    ):
        class_distributions_ = {
            'class_name': [],
            'prob': [],
            'coverage': []
        }
        for i in range(len(answers_)):
            class_distribution = answers_[:i + 1].value_counts().to_dict()
            for class_name in classes_:
                class_distributions_['class_name'].append(
                    class_name
                )
                class_distributions_['coverage'].append((i + 1) / len(answers_))
                if class_name in class_distribution:
                    class_distributions_['prob'].append(
                        class_distribution[class_name] / (i+1)
                    )
                else:
                    class_distributions_['prob'].append(0)

        return class_distributions_

    class_distributions = None
    if score_name in ['answer_f1', 'rouge_l']:
        df = pd.DataFrame({
            score_name: scores,
            confidence_name: confidence
        })
        df = df.sort_values(by=confidence_name, ascending=False)
        selective_scores = compute_mean_scores_on_subsets(df[score_name].to_list())
        selective_scores_random = np.zeros((10, len(selective_scores)))
        for i in range(len(selective_scores_random)):
            selective_scores_random[i] = compute_mean_scores_on_subsets(df[score_name].sample(frac=1).to_list())
    elif score_name in ['classification_f1', 'unanswerable_f1']:
        df = pd.DataFrame({
            'gold_answers': gold_answers,
            'predicted_answers': predicted_answers,
            confidence_name: confidence
        })
        df = df.sort_values(by=confidence_name, ascending=False)
        # Get class distributions
        class_distributions = compute_class_distribution_on_subsets(
            df['gold_answers'],
            classes
        )
        if score_name == 'classification_f1':

            # Map string class labels to integers
            class_name_int_mapping = {
                class_name: i
                for i, class_name in enumerate(classes)
            }
            classes = list(range(len(classes)))
            class_name_int_mapping['none'] = len(classes)
            # Convert labels to integers
            df['gold_answers'] = df['gold_answers'].apply(
                lambda x: class_name_int_mapping[x]
            )
            df['predicted_answers'] = df['predicted_answers'].apply(
                lambda x: class_name_int_mapping[x]
            )

        selective_scores = compute_classification_f1_on_subsets(
            df['gold_answers'],
            df['predicted_answers'],
            classes_=classes,
            macro_or_micro_=macro_or_micro
        )
        selective_scores_random = np.zeros((10, len(selective_scores)))
        for i in range(len(selective_scores_random)):
            df = df.sample(frac=1)
            selective_scores_random[i] = compute_classification_f1_on_subsets(
                df['gold_answers'].to_list(),
                df['predicted_answers'].to_list(),
                classes_=[i for i in range(len(classes))],
                macro_or_micro_=macro_or_micro
            )
        df = df.sort_values(by=confidence_name, ascending=False)
    else:
        raise NotImplementedError

    selective_scores_random = np.mean(selective_scores_random, axis=0)
    selective_scores_random_std = np.std(selective_scores_random, axis=0)

    n_samples_total = len(selective_scores)
    coverage = np.arange(n_samples_total) / n_samples_total
    df['Coverage'] = coverage
    df[f'Mean {score_name}'] = selective_scores
    df[f'Mean {score_name} std'] = [0 for _ in range(len(selective_scores))]
    df[f'Mean {score_name} random order'] = selective_scores_random
    df[f'Mean {score_name} random order std'] = selective_scores_random_std
    plot = df.plot.line(
        x='Coverage',
        y=[f'Mean {score_name}', f'Mean {score_name} random order'],
        yerr=[
            df[f'Mean {score_name} std'],
            df[f'Mean {score_name} random order std']
        ]
    )
    auc = sklearn.metrics.auc(df['Coverage'], df[f'Mean {score_name}'])
    random_auc = sklearn.metrics.auc(df['Coverage'], df[f'Mean {score_name} random order'])
    plt.text(
        0,
        0.2,
        f"AUC={auc:.2f}\nRandom AUC={random_auc:.2f}",
        horizontalalignment='left',
        size='medium',
        weight='semibold'
    )
    if out_dir is not None:
        filename = f'selective_prediction_{confidence_name}_{score_name}.png'
        plot.get_figure().savefig(
            out_dir / filename
        )
    plt.close()
    return {
        'Selective Advantage': auc - random_auc,
        f'Selective AUC {score_name} - {confidence_name}': auc,
        f'AUC {score_name} - Random': random_auc,
        f'Class Distributions': class_distributions,
        f'Selective Scores': {
            score_name: selective_scores,
            'coverage': coverage.tolist()
        }
    }
