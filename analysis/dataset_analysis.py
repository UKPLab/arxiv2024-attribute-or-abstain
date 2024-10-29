import copy
import json
from typing import List
import sys

import nltk

sys.path.append('..')

import hydra
import numpy as np

from evaluation.common import BaseInstance, Statistics, SingleFileDataset
from evaluation.tasks.contract_nli_task import ContractNLITask
from evaluation.tasks.evidence_inference_task import EvidenceInferenceTask
from evaluation.tasks.govreport_task import GovReportTask
from evaluation.tasks.natural_questions_task import NaturalQuestionsTask, NaturalQuestionsDataset
from evaluation.tasks.qasper_task import QASPERTask
from evaluation.tasks.wice_task import WiceTask
from evaluation.util import is_instance_answerable

TASK_CLASSES = {
    'qasper': QASPERTask,
    'natural_questions': NaturalQuestionsTask,
    'evidence_inference': EvidenceInferenceTask,
    'wice': WiceTask,
    'govreport': GovReportTask,
    'contract_nli': ContractNLITask,
}


def count_unanswerable(
        instances: List[BaseInstance],
) -> int:
    '''
    Count the number of unanswerable instances
    :param instances: List of instances for which to count unanswerable
    :return:
    '''
    n_unanswerable = 0
    for instance in instances:
        if not is_instance_answerable(
            instance
        ):
            n_unanswerable += 1

    return n_unanswerable


def get_n_docs(instances: List[BaseInstance]):
    processed_docs = []
    n_docs = 0
    for instance in instances:
        if instance.document.nodes[0].content in processed_docs:
            continue
        n_docs += 1
        processed_docs.append(copy.deepcopy(instance.document.nodes[0].content))

    return n_docs


def get_n_unique_extraction_nodes_per_instance(
        instances: List[BaseInstance]
) -> float:
    '''
    Count the number of unique gold evidence nodes for each instance and compute
    the mean
    :param instances:
    :return:
    '''
    n_evidence_nodes = []
    for instance in instances:
        evidence_nodes_for_instance = set()
        for annotation in instance.extraction_nodes:
            if instance.answer_has_multiple_statements:
                for extraction_nodes_for_statement in annotation:
                    for n in extraction_nodes_for_statement:
                        evidence_nodes_for_instance.add(n)
            else:
                for n in annotation:
                    evidence_nodes_for_instance.add(n)
        n_evidence_nodes.append(len(evidence_nodes_for_instance))

    mean_n_evidence_nodes = np.mean(n_evidence_nodes)
    return mean_n_evidence_nodes


def get_n_extraction_nodes_per_annotation(
        instances: List[BaseInstance]
) -> float:
    n_extraction_nodes = []
    for instance in instances:
        for annotation in instance.extraction_nodes:
            if instance.answer_has_multiple_statements:
                for extraction_nodes_for_statement in annotation:
                    n_extraction_nodes_for_statement = len(
                        extraction_nodes_for_statement
                    )
                    if n_extraction_nodes_for_statement > 0:
                        n_extraction_nodes.append(n_extraction_nodes_for_statement)
            else:
                n_extraction_node_for_annotation = len(annotation)
                if n_extraction_node_for_annotation > 0:
                    n_extraction_nodes.append(n_extraction_node_for_annotation)

    mean_n_extraction_nodes = np.mean(n_extraction_nodes)
    return mean_n_extraction_nodes


def get_n_words_per_doc(
        instances: List[BaseInstance]
) -> float:
    processed_docs = []
    n_words = []
    for instance in instances:
        if instance.document.nodes[0].content in processed_docs:
            continue
        n_words_for_doc = 0
        for n in instance.document.nodes:
            n_words_for_doc += len(n.content.split())

        n_words.append(n_words_for_doc)
        processed_docs.append(copy.deepcopy(instance.document.nodes[0].content))

    mean_n_words = np.mean(n_words)
    return mean_n_words


def get_n_words_per_gold_response(
        instances: List[BaseInstance]
) -> float:
    n_words = []
    for instance in instances:
        for annotation in instance.free_text_answer:
            if instance.answer_has_multiple_statements:
                n_words_for_annotation = 0
                for statement in annotation:
                    n_words_for_annotation += len(statement.split())
                n_words.append(n_words_for_annotation)
            else:
                n_words.append(len(annotation.split()))
    mean_n_words = np.mean(n_words)
    return mean_n_words


def get_n_statements_per_gold_response(
        instances: List[BaseInstance]
) -> float:
    if not instances[0].answer_has_multiple_statements:
        return 1.0

    n_statements = []
    for instance in instances:
        for annotation in instance.free_text_answer:
            n_statements.append(len(annotation))
    mean_n_statements = np.mean(n_statements)
    return mean_n_statements


def main(
        task_name: str,
        location: str
):

    print(f'Computing statistics for {task_name}')
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

    task_class = TASK_CLASSES[task_name]

    stats = Statistics('', task_name, '', config)

    print(f'Loading task {task_name}')
    task = task_class(
        config,
        stats
    )

    print(len(task.train_instances))

    if isinstance(task.train_instances, SingleFileDataset):
        task._train_instances = task.train_instances[:20000]
    if isinstance(task.dev_instances, SingleFileDataset):
        task._dev_instances = task.dev_instances[:]
    if isinstance(task.test_instances, SingleFileDataset):
        task._test_instances = task.test_instances[:]

    statistics = {
        'n_instances': {
            'total': 0,
            'train': 0,
            'dev': 0,
            'test': 0
        },
        'n_docs': {
            'total': 0,
            'train': 0,
            'dev': 0,
            'test': 0
        },
        'n_unanswerable': {
            'total': 0,
            'train': 0,
            'dev': 0,
            'test': 0
        },
        'n_unique_extraction_nodes_per_instance': {
            'total': 0.0,
            'train': 0.0,
            'dev': 0.0,
            'test': 0.0
        },
        'n_extraction_nodes_per_annotation': {
            'total': 0.0,
            'train': 0.0,
            'dev': 0.0,
            'test': 0.0
        },
        'n_words_per_doc': {
            'total': 0.0,
            'train': 0.0,
            'dev': 0.0,
            'test': 0.0
        },
        'n_words_per_gold_response': {
            'total': 0.0,
            'train': 0.0,
            'dev': 0.0,
            'test': 0.0
        },
        'n_statements_per_gold_response': {
            'total': 0.0,
            'train': 0.0,
            'dev': 0.0,
            'test': 0.0
        }
    }
    print('Computing n_instances')
    statistics['n_instances']['train'] = len(task.train_instances)
    statistics['n_instances']['dev'] = len(task.dev_instances)
    statistics['n_instances']['test'] = len(task.test_instances)
    statistics['n_instances']['total'] = (
        statistics['n_instances']['train']
        + statistics['n_instances']['dev']
        + statistics['n_instances']['test']
    )
    print('Computing n_docs')
    statistics['n_docs']['train'] = get_n_docs(task.train_instances)
    statistics['n_docs']['dev'] = get_n_docs(task.dev_instances)
    statistics['n_docs']['test'] = get_n_docs(task.test_instances)
    statistics['n_docs']['total'] = (
        statistics['n_docs']['train']
        + statistics['n_docs']['dev']
        + statistics['n_docs']['test']
    )

    print('Computing n_unanswerable')
    statistics['n_unanswerable']['train'] = count_unanswerable(task.train_instances)
    statistics['n_unanswerable']['dev'] = count_unanswerable(task.dev_instances)
    statistics['n_unanswerable']['test'] = count_unanswerable(task.test_instances)
    statistics['n_unanswerable']['total'] = (
        statistics['n_unanswerable']['train']
        + statistics['n_unanswerable']['dev']
        + statistics['n_unanswerable']['test']
    )

    print('Computing n_unique_extraction_nodes_per_instance')
    statistics['n_unique_extraction_nodes_per_instance']['train'] = get_n_unique_extraction_nodes_per_instance(
        task.train_instances
    )
    statistics['n_unique_extraction_nodes_per_instance']['dev'] = get_n_unique_extraction_nodes_per_instance(
        task.dev_instances
    )
    statistics['n_unique_extraction_nodes_per_instance']['test'] = get_n_unique_extraction_nodes_per_instance(
        task.test_instances
    )
    statistics['n_unique_extraction_nodes_per_instance']['total'] = (
        statistics['n_instances']['train'] * statistics['n_unique_extraction_nodes_per_instance']['train']
        + statistics['n_instances']['dev'] * statistics['n_unique_extraction_nodes_per_instance']['dev']
        + statistics['n_instances']['test'] * statistics['n_unique_extraction_nodes_per_instance']['test']
    ) / statistics['n_instances']['total']

    print('Computing n_extraction_nodes_per_annotation')
    statistics['n_extraction_nodes_per_annotation']['train'] = get_n_extraction_nodes_per_annotation(
        task.train_instances
    )
    statistics['n_extraction_nodes_per_annotation']['dev'] = get_n_extraction_nodes_per_annotation(
        task.dev_instances
    )
    statistics['n_extraction_nodes_per_annotation']['test'] = get_n_extraction_nodes_per_annotation(
        task.test_instances
    )
    statistics['n_extraction_nodes_per_annotation']['total'] = (
        statistics['n_instances']['train'] * statistics['n_extraction_nodes_per_annotation']['train']
        + statistics['n_instances']['dev'] * statistics['n_extraction_nodes_per_annotation']['dev']
        + statistics['n_instances']['test'] * statistics['n_extraction_nodes_per_annotation']['test']
    ) / statistics['n_instances']['total']

    print('Computing n_words_per_doc')
    statistics['n_words_per_doc']['train'] = get_n_words_per_doc(
        task.train_instances
    )
    statistics['n_words_per_doc']['dev'] = get_n_words_per_doc(
        task.dev_instances
    )
    statistics['n_words_per_doc']['test'] = get_n_words_per_doc(
        task.test_instances
    )
    statistics['n_words_per_doc']['total'] = (
        statistics['n_docs']['train'] * statistics['n_words_per_doc']['train']
        + statistics['n_docs']['dev'] * statistics['n_words_per_doc']['dev']
        + statistics['n_docs']['test'] * statistics['n_words_per_doc']['test']
    ) / statistics['n_docs']['total']

    print('Computing n_words_per_gold_response')
    statistics['n_words_per_gold_response']['train'] = get_n_words_per_gold_response(
        task.train_instances
    )
    statistics['n_words_per_gold_response']['dev'] = get_n_words_per_gold_response(
        task.dev_instances
    )
    statistics['n_words_per_gold_response']['test'] = get_n_words_per_gold_response(
        task.test_instances
    )
    statistics['n_words_per_gold_response']['total'] = (
        statistics['n_instances']['train'] * statistics['n_words_per_gold_response']['train']
        + statistics['n_instances']['dev'] * statistics['n_words_per_gold_response']['dev']
        + statistics['n_instances']['test'] * statistics['n_words_per_gold_response']['test']
    ) / statistics['n_instances']['total']

    print('Computing n_statements_per_gold_response')
    statistics['n_statements_per_gold_response']['train'] = get_n_statements_per_gold_response(
        task.train_instances
    )
    statistics['n_statements_per_gold_response']['dev'] = get_n_statements_per_gold_response(
        task.dev_instances
    )
    statistics['n_statements_per_gold_response']['test'] = get_n_statements_per_gold_response(
        task.test_instances
    )
    statistics['n_statements_per_gold_response']['total'] = (
        statistics['n_instances']['train'] * statistics['n_statements_per_gold_response']['train']
        + statistics['n_instances']['dev'] * statistics['n_statements_per_gold_response']['dev']
        + statistics['n_instances']['test'] * statistics['n_statements_per_gold_response']['test']
    ) / statistics['n_instances']['total']

    print(f'Statistics for {task_name}')
    print(json.dumps(statistics, indent=4))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_name',
        type=str,
        required=True
    )
    parser.add_argument(
        '--location',
        type=str,
        required=True
    )
    args = parser.parse_args()
    main(args.task_name, args.location)