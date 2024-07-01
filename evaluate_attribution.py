import hashlib
import json
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import git
import numpy as np
import pandas as pd
import torch.cuda

from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from evaluation.common import CustomDataset
from attribution_eval.attribution_model import AttributionBaseModel
from attribution_eval.util import AttributionInstance, load_attribution_dataset_jsonl, load_attribution_dataset_csv

RANDOM_SEED=12345
random.seed(RANDOM_SEED)


DATA_DIR = {
    'local': Path('../data/attribution/'),
}


def predict(
        model,
        dataloader: DataLoader
):
    """Predict attribution for all instances in dataloader."""
    if model.model is not None:
        model.model.eval()
    predictions = []
    with tqdm(dataloader, unit='batch') as bar:
        bar.set_description(f'Prediction')
        for batch in bar:
            predictions.extend(model.predict(batch))

    return predictions


def evaluate(
        predictions: List[int],
        instances: List[AttributionInstance],
        tune_threshold: bool = False
):
    """Evaluate F1 score"""
    y_true = [
        instance.label for instance in instances
    ]

    if tune_threshold:
        pass

    scores = precision_recall_fscore_support(
        y_true[:len(predictions)],
        predictions,
        labels=[0,1]
    )
    macro_f1 = np.mean(scores[2])
    accuracy = np.mean(np.array(predictions) == np.array(y_true[:len(predictions)]))
    # Compute TP, TN, FP, FN
    # TP: predicted 1, true 1
    # TN: predicted 0, true 0
    # FP: predicted 1, true 0
    # FN: predicted 0, true 1
    tp = sum([1 for p, t in zip(predictions, y_true[:len(predictions)]) if p == 1 and t == 1])
    tn = sum([1 for p, t in zip(predictions, y_true[:len(predictions)]) if p == 0 and t == 0])
    fp = sum([1 for p, t in zip(predictions, y_true[:len(predictions)]) if p == 1 and t == 0])
    fn = sum([1 for p, t in zip(predictions, y_true[:len(predictions)]) if p == 0 and t == 1])
    balanced_accuracy = 0.5 * ((tp/(tp+fn)) + (tn/(tn+fp)))

    return {
        'F1': {
            'macro': macro_f1,
            '0': scores[2][0],
            '1': scores[2][1]
        },
        'Accuracy': accuracy,
        'Balanced Accuracy': balanced_accuracy
    }


def make_output(
        predicted_labels: List[int],
        instances: List[AttributionInstance],
        metrics: Dict,
        model_name: str,
        task_name: str,
        concatenate_extraction_nodes: bool,
        description: str
) -> Tuple[Dict, pd.DataFrame]:
    """Create output dict with config, scores and predictions."""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    config = {
        'description': description,
        'hash': None,
        'model_name': model_name,
        'task_name': task_name,
        'commit_hash': sha,
        'concatenate_extraction_nodes': concatenate_extraction_nodes
    }
    config_hash = hashlib.sha256(bytes(f"{dict(config)}", "utf-8")).hexdigest()
    config['hash'] = config_hash[:4]
    result = metrics
    output = {
        'config': config,
        'result': result
    }

    table = {
        'label': [],
        'claim': [],
        'evidence': [],
        'task_name': [],
        'example_id': [],
        'annotation_idx': [],
        'answer_statement_idx': [],
        'answer_type': []
    }
    for predicted_label, instance in zip(predicted_labels, instances):
        table['label'].append(predicted_label)
        table['claim'].append(instance.claim)
        table['evidence'].append(instance.evidence)
        table['task_name'].append(instance.task_name)
        table['example_id'].append(instance.example_id)
        table['annotation_idx'].append(instance.annotation_idx)
        table['answer_statement_idx'].append(instance.sentence_idx)
        table['answer_type'].append(instance.answer_type)

    table = pd.DataFrame(table)

    return output, table


def main(
        description: str,
        model_name: str,
        task_name: str,
        partition: str,
        dataset_is_csv: bool,
        is_three_way_annotation: bool,
        location: str,
        batch_size: int,
        concatenate_extraction_nodes: bool
):
    """
    This function evaluates attribution models on "gold" data.
    The annotated data for specific tasks is used to create attributable and
    non-attributable instances.
    :param model_name: The name of the model that evaluates attribution.
        Available options: "true_nli", "attrscore".
    :param partition: Which partition of the task data to use ("train", "dev"
        or "test").
    :param location: Where the script is running, 'local' or 'shared'
    :param batch_size: The batch size for the model
    :param concatenate_extraction_nodes: If True, all extraction nodes for an
        instance are concatenated and attribution is predicted once. If False,
        attribution is predicted separately for each extraction node and the max
        attribution score is used as the final score.
    """
    # Load data
    dataset_dir_path = DATA_DIR[location]
    if not dataset_is_csv:
        dataset_path = dataset_dir_path / 'datasets' / f'{task_name}-{partition}.jsonl'
        attribution_instances = load_attribution_dataset_jsonl(
            dataset_path
        )
    else:
        dataset_path = dataset_dir_path / 'datasets' / f'{task_name}-{partition}.csv'
        attribution_instances = load_attribution_dataset_csv(
            dataset_path,
            is_three_way_annotation=is_three_way_annotation
        )

    dataset = CustomDataset(attribution_instances)

    # Load model
    print(f'Loading model {model_name}')
    model = AttributionBaseModel.load_model(
        model_name,
        predict_max_in_batch=not(concatenate_extraction_nodes)
    )
    if torch.cuda.is_available() and model.model is not None:
        device = torch.device('cuda:0')
        model = model.to(device)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        collate_fn=model.collate_fn,
        batch_size=batch_size,
        shuffle=False
    )

    # Evaluate model
    predictions = predict(
        model,
        dataloader
    )
    print('Evaluating predictions')
    result = evaluate(
        predictions,
        attribution_instances
    )
    print(json.dumps(result, indent=4))

    # Output result
    output_dict, output_table = make_output(
        predictions,
        attribution_instances,
        result['F1'],
        model_name,
        attribution_instances[0].task_name,
        concatenate_extraction_nodes,
        description
    )
    print(f'Saving output with hash {output_dict["config"]["hash"]}')
    out_dir_path = DATA_DIR[location] / 'results' / f'{output_dict["config"]["hash"]}'
    os.mkdir(out_dir_path)

    with open(out_dir_path / 'results.json', 'w') as f:
        json.dump(output_dict, f, indent=4)

    output_table.to_csv(out_dir_path / 'predictions.csv')

    print('Done')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--description',
        type=str,
        default=''
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='true_nli'
    )
    parser.add_argument(
        '--task_name',
        type=str,
        default='qasper'
    )
    parser.add_argument(
        '--partition',
        type=str,
        default='dev'
    )
    parser.add_argument(
        '--is_csv',
        action='store_true'
    )
    parser.add_argument(
        '--location',
        type=str,
        default='local'
    )
    parser.add_argument(
        '--is_three_way_annotation',
        action='store_true'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1
    )
    parser.add_argument(
        '--no_concatenation',
        action='store_true'
    )
    parser.add_argument(
        '--remote_debug',
        action='store_true'
    )
    args = parser.parse_args()
    if args.remote_debug:
        import pydevd_pycharm
        pydevd_pycharm.settrace('10.167.11.14', port=3851, stdoutToServer=True, stderrToServer=True)

    main(
        args.description,
        args.model_name,
        args.task_name,
        args.partition,
        args.is_csv,
        args.is_three_way_annotation,
        args.location,
        args.batch_size,
        not args.no_concatenation
    )
