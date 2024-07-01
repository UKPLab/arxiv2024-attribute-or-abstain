import copy
import json
from pathlib import Path
from typing import List
import sys
sys.path.append('..')

import nltk
import hydra
import torch
from tqdm import tqdm

from config_lib.base_config import BaseConfig
from evaluation.common import BaseInstance, Statistics, BasePrediction
from evaluation.tasks.evidence_inference_task import EvidenceInferenceTask
from evaluation.tasks.govreport_task import GovReportTask
from evaluation.tasks.natural_questions_task import NaturalQuestionsTask
from evaluation.tasks.qasper_task import QASPERTask
from evaluation.tasks.wice_task import WiceTask
from models.retrieve import BaseRetriever

TASK_CLASSES = {
    'qasper': QASPERTask,
    'natural_questions': NaturalQuestionsTask,
    'evidence_inference': EvidenceInferenceTask,
    'wice': WiceTask,
    'govreport': GovReportTask
}


def make_attributed_instances(
        instances: List[BaseInstance],
        post_hoc_retrieval_model: BaseRetriever
) -> List[BaseInstance]:
    attributed_instances = []

    # use tqdm
    bar = tqdm(instances, unit='instance')
    for instance in bar:
        oracle_predictions = [
            BasePrediction(
                instance.task_name,
                instance.example_id,
                instance.free_text_answer[i],
                instance.extraction_nodes[i],
                instance.answer_type[i]
            )
            for i in range(len(instance.free_text_answer))
        ]
        attributed_instance = copy.deepcopy(instance)
        attributed_instance.extraction_nodes = []
        for oracle_prediction in oracle_predictions:
            post_hoc_prediction = post_hoc_retrieval_model.post_hoc_retrieve_and_update_prediction(
                oracle_prediction,
                instance
            )
            attributed_instance.extraction_nodes.append(post_hoc_prediction.extraction_nodes)
        attributed_instances.append(attributed_instance)

    return attributed_instances


def save_instances(
        instances: List[BaseInstance],
        path: Path
) -> None:
    # Save jsonl file using instance.to_json_dict
    with path.open('w') as f:
        for instance in instances:
            f.write(json.dumps(instance.to_json_dict()) + '\n')


def load_config(
        task_name: str,
        location: str
) -> BaseConfig:
    with hydra.initialize(
            version_base=None,
            config_path='../config',
            job_name=''
    ):
        config = hydra.compose(
            config_name='config',
            overrides=[
                f'task={task_name}',
                f'location={location}'
            ]
        )

    return config


def main(
        task_name: str,
        location: str,
        post_hoc_retrieval_model_name: str,
        k_extraction_nodes: int,
        out_dir_path: Path
):
    config = load_config(task_name, location)

    # Load task instances
    task_class = TASK_CLASSES[task_name]

    stats = Statistics('', task_name, '', config)

    print('Loading task')
    task = task_class(
        config,
        stats
    )

    model = BaseRetriever.load_model(
        'post_hoc',
        post_hoc_retrieval_model_name,
        config.task.answer_has_multiple_statements,
        k_extraction_nodes,
        classes=config.task.classes,
        instances=task.train_instances
    )
    if torch.cuda.is_available() and model.model is not None:
        device = torch.device('cuda:0')
        model = model.to(device)

    attributed_data = {
        'train': [],
        'dev': [],
        'test': []
    }
    for split_name in attributed_data:
        print(f'Making attributed {split_name} data')
        instances = getattr(task, f'{split_name}_instances')
        attributed_instances = make_attributed_instances(
            instances,
            model
        )

        attributed_data[split_name].extend(attributed_instances)

    for split_name in attributed_data:
        print(f'Saving {split_name} data')
        save_instances(attributed_data[split_name], out_dir_path / f'{split_name}.jsonl')


if __name__ == '__main__':
    # Use argparse to get all arguments in main() as command line arguments

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'out_dir_path',
        type=str
    )
    parser.add_argument(
        '--task_name',
        type=str,
        default='govreport'
    )
    parser.add_argument(
        '--location',
        type=str,
        default='shared'
    )
    parser.add_argument(
        '--post_hoc_retrieval_model_name',
        type=str,
        default='bm25'
    )
    parser.add_argument(
        '--k_extraction_nodes',
        type=int,
        default=2
    )

    # call main
    args = parser.parse_args()
    main(
        args.task_name,
        args.location,
        args.post_hoc_retrieval_model_name,
        args.k_extraction_nodes,
        Path(args.out_dir_path)
    )
