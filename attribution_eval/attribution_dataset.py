# Implement creating attribution datasets from task datasets
import json
import random
from pathlib import Path
from typing import List, Dict
import sys
sys.path.append('..')

import hydra

from attribution_eval.util import AttributionInstance, save_attribution_dataset_jsonl, save_attribution_dataset_csv
from evaluation.common import BaseInstance, BasePrediction, Statistics
from evaluation.tasks.evidence_inference_task import EvidenceInferenceInstance, EvidenceInferenceTask
from evaluation.util import find_label
from evaluation.tasks.govreport_task import GovReportTask
from evaluation.tasks.natural_questions_task import NaturalQuestionsTask
from evaluation.tasks.qasper_task import QASPERTask
from evaluation.tasks.wice_task import WiceTask
from evaluation.tasks.contract_nli_task import ContractNLITask

CONTRACT_NLI_CLAIMS = {
    "nda-11": {
        "short_description": "No reverse engineering",
        "hypothesis": "Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information.",
        "inverse_hypothesis": "Receiving Party may reverse engineer any objects which embody Disclosing Party's Confidential Information."
    },
    "nda-16": {
        "short_description": "Return of confidential information",
        "hypothesis": "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.",
        "inverse_hypothesis": "Receiving Party shall not destroy or return some Confidential Information upon the termination of Agreement."
    },
    "nda-15": {
        "short_description": "No licensing",
        "hypothesis": "Agreement shall not grant Receiving Party any right to Confidential Information.",
        "inverse_hypothesis": "Agreement may grant Receiving Party any right to Confidential Information.",
    },
    "nda-10": {
        "short_description": "Confidentiality of Agreement",
        "hypothesis": "Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.",
        "inverse_hypothesis": "Receiving Party may disclose the fact that Agreement was agreed or negotiated."
    },
    "nda-2": {
        "short_description": "None-inclusion of non-technical information",
        "hypothesis": "Confidential Information shall only include technical information.",
        "inverse_hypothesis": "Confidential Information may include other information besides technical information."
},
    "nda-1": {
        "short_description": "Explicit identification",
        "hypothesis": "All Confidential Information shall be expressly identified by the Disclosing Party.",
        "inverse_hypothesis": "All Confidential Information may be identified by the Disclosing Party at any time."
},
    "nda-19": {
        "short_description": "Survival of obligations",
        "hypothesis": "Some obligations of Agreement may survive termination of Agreement.",
        "inverse_hypothesis": "No obligations of Agreement may survive termination of Agreement."
    },
    "nda-12": {
        "short_description": "Permissible development of similar information",
        "hypothesis": "Receiving Party may independently develop information similar to Confidential Information.",
        "inverse_hypothesis": "Receiving Party shall not independently develop information similar to Confidential Information."
    },
    "nda-20": {
        "short_description": "Permissible post-agreement possession",
        "hypothesis": "Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.",
        "inverse_hypothesis": "Receiving Party shall not retain any Confidential Information even after the return or destruction of Confidential Information."
    },
    "nda-3": {
        "short_description": "Inclusion of verbally conveyed information",
        "hypothesis": "Confidential Information may include verbally conveyed information.",
        "inverse_hypothesis": "Confidential Information shall not include verbally conveyed information."
    },
    "nda-18": {
        "short_description": "No solicitation",
        "hypothesis": "Receiving Party shall not solicit some of Disclosing Party's representatives.",
        "inverse_hypothesis": "Receiving Party may solicit some of Disclosing Party's representatives."
    },
    "nda-7": {
        "short_description": "Sharing with third-parties",
        "hypothesis": "Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).",
        "inverse_hypothesis": "Receiving Party shall not share any Confidential Information with any third-parties (including consultants, agents and professional advisors)."
    },
    "nda-17": {
        "short_description": "Permissible copy",
        "hypothesis": "Receiving Party may create a copy of some Confidential Information in some circumstances.",
        "inverse_hypothesis": "Receiving Party shall not create a copy of any Confidential Information in any circumstances."
    },
    "nda-8": {
        "short_description": "Notice on compelled disclosure",
        "hypothesis": "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.",
        "inverse_hypothesis": "Receiving Party is not obliged to notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information."
    },
    "nda-13": {
        "short_description": "Permissible acquirement of similar information",
        "hypothesis": "Receiving Party may acquire information similar to Confidential Information from a third party.",
        "inverse_hypothesis": "Receiving Party shall not acquire information similar to Confidential Information from a third party."
    },
    "nda-5": {
        "short_description": "Sharing with employees",
        "hypothesis": "Receiving Party may share some Confidential Information with some of Receiving Party's employees.",
        "inverse_hypothesis": "Receiving Party shall not share any Confidential Information with any of Receiving Party's employees."
    },
    "nda-4": {
        "short_description": "Limited use",
        "hypothesis": "Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.",
        "inverse_hypothesis": "Receiving Party may use some Confidential Information for purposes other than the purposes stated in Agreement."
    }
}

CONTRACT_NLI_INVERSE_CLAIM_MAPPING = {
    d['hypothesis']: d['inverse_hypothesis']
    for d in CONTRACT_NLI_CLAIMS.values()
}

CONTRACT_NLI_LABELS = [
    'entailment',
    'contradiction',
    'not mentioned'
]

WICE_LABELS = [
    'not supported',
    'partially supported',
    'supported'
]


def make_attribution_dataset(
        task_name: str,
        answer_has_multiple_statements: bool,
        instances: List[BaseInstance],
        invert_contract_nli_contradiction_claim: bool = True
) -> List[AttributionInstance]:
    """
    Create a dataset to evaluate attribution models from a dataset with annotated
    evidence. We pair the original annotated answer / class / summary sentence / ...
    with the original annotated evidence to get a "positive" (attributable)
    instance. We create negative instances by pairing the original answer with
    random evidence.
    :param task_name:
    :param answer_has_multiple_statements:
    :param instances:
    :return:
    """
    attribution_instances = []

    for instance in instances:
        # Make positive instances
        attribution_instances.extend(
            make_attribution_instances_from_base_instance(
                instance,
                answer_has_multiple_statements,
                prediction=None,
                invert_contract_nli_contradiction_claim=invert_contract_nli_contradiction_claim
            )
        )
        # Make negative instances with random evidence
        attribution_instances.extend(
            make_negative_attribution_instances_from_base_instance(
                instance,
                answer_has_multiple_statements,
                invert_contract_nli_contradiction_claim=invert_contract_nli_contradiction_claim
            )
        )

    return attribution_instances


def format_evidence(
        doc_title: str,
        evidence: str
) -> str:
    return 'Document title: ' + doc_title + '\n' + evidence


def make_attribution_instances_from_base_instance(
        instance: BaseInstance | EvidenceInferenceInstance,
        answer_has_multiple_statements: bool,
        prediction: BasePrediction = None,
        answer_type: str = None,
        skip_empty: bool = True,
        invert_contract_nli_contradiction_claim: bool = True
) -> List[AttributionInstance]:
    """
    Create attribution instances by converting answers to claims and pairing them
    with the evidence. Answer-to-claim conversion is dataset-specific.
    When a prediction is given, we use the answer and extraction nodes from the
    prediction.
    :param instance:
    :param answer_has_multiple_statements:
    :param prediction:
    :param answer_type: When prediction is passed, use this answer type
    :param skip_empty: If true, do not create attribution instances for answers
        / statements without evidence
    :return:
    """
    annotation_idxs = []
    sentence_idxs = []
    claims = []
    evidences = []
    labels = []
    all_answer_types = []

    if prediction is not None:
        free_text_answers = [prediction.free_text_answer]
        extraction_nodes = [prediction.extraction_nodes]
        if answer_type is not None:
            answer_types = [answer_type for _ in free_text_answers]
        else:
            answer_types = ['none' for _ in free_text_answers]
    else:
        free_text_answers = instance.free_text_answer
        extraction_nodes = instance.extraction_nodes
        answer_types = instance.answer_type

    for i, (free_text_answer, extraction_node_list, answer_type) in enumerate(zip(
        free_text_answers, extraction_nodes, answer_types
    )):
        annotation_idx = i
        if answer_has_multiple_statements:
            for j, (answer_sentence, extraction_nodes_for_sentence) in enumerate(zip(
                free_text_answer, extraction_node_list
            )):
                if (
                        (not extraction_nodes_for_sentence or not answer_sentence)
                         and skip_empty
                ):
                    continue
                # Sort extraction nodes
                sorted_extraction_nodes_for_sentence = sorted(
                    extraction_nodes_for_sentence,
                    key=lambda x: int(x.ix.split('_')[-1])
                )
                evidence = '\n'.join(n.content for n in sorted_extraction_nodes_for_sentence)
                evidence = format_evidence(instance.document.nodes[0].content, evidence)
                if instance.task_name in ['govreport']:
                    claim = make_claim_from_summarization(answer_sentence)
                    label = 1
                else:
                    raise NotImplementedError
                annotation_idxs.append(annotation_idx)
                sentence_idxs.append(j)
                claims.append(claim)
                evidences.append(evidence)
                labels.append(label)
                all_answer_types.append(answer_type)
        else:
            if (
                    (not extraction_node_list or not free_text_answer)
                    and skip_empty
            ):
                continue
            sentence_idx = 0
            sorted_extraction_node_list = sorted(
                extraction_node_list,
                key=lambda x: int(x.ix.split('_')[-1])
            )
            evidence = '\n'.join(n.content for n in sorted_extraction_node_list)
            evidence = format_evidence(instance.document.nodes[0].content, evidence)
            if instance.task_name in ['qasper', 'natural_questions']:
                claim = make_claim_from_qa(
                    instance.question,
                    free_text_answer
                )
                label = 1
            elif instance.task_name == 'contract_nli':
                free_text_answer = find_label(
                    CONTRACT_NLI_LABELS,
                    free_text_answer
                )
                claim = make_claim_from_contract_nli(
                    instance.statement,
                    free_text_answer,
                    invert_contradiction_claim=invert_contract_nli_contradiction_claim
                )
                if free_text_answer in ['contradiction', 'entailment']:
                    label = 1
                else:
                    label = 0
            elif instance.task_name == 'wice':
                free_text_answer = find_label(
                    WICE_LABELS,
                    free_text_answer
                )
                claim = make_claim_from_wice(
                    instance.statement,
                    free_text_answer,
                    instance.additional_info
                )
                if free_text_answer == 'not supported':
                    label = 0
                else:
                    label = 1
            elif instance.task_name == 'evidence_inference':
                claim = make_claim_from_evidence_inference(
                    free_text_answer,
                    instance.outcome,
                    instance.comparator,
                    instance.intervention
                )
                label = 1
            else:
                raise NotImplementedError

            annotation_idxs.append(annotation_idx)
            sentence_idxs.append(sentence_idx)
            claims.append(claim)
            evidences.append(evidence)
            labels.append(label)
            all_answer_types.append(answer_type)

    attribution_instances = []
    for (
        annotation_idx,
        sentence_idx,
        claim,
        evidence,
        label,
        answer_type
    ) in zip(
        annotation_idxs,
        sentence_idxs,
        claims,
        evidences,
        labels,
        all_answer_types
    ):
        attribution_instance = AttributionInstance(
            instance.task_name,
            instance.example_id,
            annotation_idx,
            sentence_idx,
            claim,
            evidence,
            label,
            answer_type
        )
        attribution_instances.append(attribution_instance)

    return attribution_instances


def make_negative_attribution_instances_from_base_instance(
        instance: BaseInstance | EvidenceInferenceInstance,
        answer_has_multiple_statements: bool,
        k_when_evidence_is_empty: int = 2,
        invert_contract_nli_contradiction_claim: bool = True
) -> List[AttributionInstance]:
    """
    Create negative attribution instances (label = 0 -> not attributable) by
    sampling random non-evidence nodes and pairing them with the original
    claims / answers
    :param instance:
    :param answer_has_multiple_statements:
    :return:
    """
    annotation_idxs = []
    sentence_idxs = []
    claims = []
    evidences = []
    labels = []
    all_answer_types = []

    free_text_answers = instance.free_text_answer
    extraction_nodes = instance.extraction_nodes
    all_extraction_nodes = list(set(
        n for node_list in extraction_nodes for n in node_list
    ))
    answer_types = instance.answer_type

    paragraphs_for_sampling = [
        n for n in instance.document.nodes
        if (
            (n.ntype == 'p')
            and (n not in all_extraction_nodes)
            and (len(n.content) > 100)
        )
    ]

    n_nodes_for_sampling = len(paragraphs_for_sampling)

    for i, (free_text_answer, extraction_node_list, answer_type) in enumerate(zip(
            free_text_answers, extraction_nodes, answer_types
    )):
        annotation_idx = i
        if answer_has_multiple_statements:
            raise NotImplementedError
        else:
            k_nodes_to_sample = len(extraction_node_list)
            if k_nodes_to_sample == 0:
                k_nodes_to_sample = k_when_evidence_is_empty
            if (
                    (k_nodes_to_sample == 0)
                    or (n_nodes_for_sampling / k_nodes_to_sample < 10)
            ):
                # Ignore instances without extraction nodes and instances
                # where there the ratio of nodes for sampling to sample size is
                # < 10
                continue

            sentence_idx = 0
            # Sample random paragraphs as "negative evidence"
            neg_extraction_nodes = random.sample(
                paragraphs_for_sampling,
                k_nodes_to_sample
            )
            # Sort
            neg_extraction_nodes = sorted(
                neg_extraction_nodes,
                key=lambda x: int(x.ix.split('_')[-1])
            )
            evidence = '\n'.join(n.content for n in neg_extraction_nodes)
            evidence = format_evidence(instance.document.nodes[0].content, evidence)
            if instance.task_name in ['qasper', 'natural_questions']:
                claim = make_claim_from_qa(
                    instance.question,
                    free_text_answer
                )
                label = 0
            elif instance.task_name == 'contract_nli':
                free_text_answer = find_label(
                    CONTRACT_NLI_LABELS,
                    free_text_answer
                )
                claim = make_claim_from_contract_nli(
                    instance.statement,
                    free_text_answer,
                    invert_contradiction_claim=invert_contract_nli_contradiction_claim
                )
                label = 0

            elif instance.task_name == 'wice':
                free_text_answer = find_label(
                    WICE_LABELS,
                    free_text_answer
                )
                claim = make_claim_from_wice(
                    instance.statement,
                    free_text_answer,
                    instance.additional_info
                )
                label = 0
            elif instance.task_name == 'evidence_inference':
                claim = make_claim_from_evidence_inference(
                    free_text_answer,
                    instance.outcome,
                    instance.comparator,
                    instance.intervention
                )
                label = 0
            else:
                raise NotImplementedError

            annotation_idxs.append(annotation_idx)
            sentence_idxs.append(sentence_idx)
            claims.append(claim)
            evidences.append(evidence)
            labels.append(label)
            all_answer_types.append(answer_type)

    attribution_instances = []
    for (
            annotation_idx,
            sentence_idx,
            claim,
            evidence,
            label,
            answer_type
    ) in zip(
        annotation_idxs,
        sentence_idxs,
        claims,
        evidences,
        labels,
        all_answer_types
    ):
        attribution_instance = AttributionInstance(
            instance.task_name,
            instance.example_id,
            annotation_idx,
            sentence_idx,
            claim,
            evidence,
            label,
            answer_type
        )
        attribution_instances.append(attribution_instance)

    return attribution_instances


def make_claim_from_qa(
        question: str,
        answer: str
) -> str:
    # If prediction is given, use answer and extraction nodes from prediction
    claim_template = 'The answer to the question "{question}" is "{answer}"'
    claim = claim_template.format(
        question=question,
        answer=answer
    )
    return claim


def make_claim_from_evidence_inference(
        answer: str,
        outcome: str,
        comparator: str,
        intervention: str
) -> str:
    # If significantly increased or decreased
    if answer in ['significantly decreased', 'significantly increased']:
        claim = f'The {intervention} {answer} {outcome} in comparison to {comparator}'
    # If no significant difference
    else:
        claim = f'There was {answer} between the effect of {intervention} and the effect of {comparator} on {outcome}.'

    return claim


def make_claim_from_contract_nli(
        statement: str,
        answer: str,
        invert_contradiction_claim: bool = True
) -> str:
    if answer == 'contradiction':
        if invert_contradiction_claim:
            claim = make_inverse_claim_from_contract_nli(statement)
        else:
            claim = f'The inverse of the statement "{statement}" is accurate.'
    else:
        claim = statement
    return claim


def make_inverse_claim_from_contract_nli(
        statement: str
) -> str:
    negative_claim = CONTRACT_NLI_INVERSE_CLAIM_MAPPING[statement]
    return negative_claim


def make_claim_from_wice(
        statement: str,
        answer: str,
        additional_info: str
) -> str:
    if answer in ['supported', 'not supported']:
        claim = f'{statement}'
    else:
        # Partially supported
        claim = f'The claim "{statement}" is partially true.'

    return claim


def make_claim_from_summarization(
        summary_sentence: str
) -> str:
    claim = summary_sentence
    return claim


def compute_dataset_statistics(
        dataset: List[AttributionInstance]
) -> Dict:
    n_positive_instances = 0
    for instance in dataset:
        if instance.label == 1:
            n_positive_instances += 1
    return {
        'n_instances': len(dataset),
        'n_positive_instances': n_positive_instances,
        'n_negative_instances': len(dataset) - n_positive_instances
    }


if __name__ == '__main__':
    import argparse

    random.seed(42)

    TASK_CLASSES = {
        'qasper': QASPERTask,
        'natural_questions': NaturalQuestionsTask,
        'evidence_inference': EvidenceInferenceTask,
        'wice': WiceTask,
        'govreport': GovReportTask,
        'contract_nli': ContractNLITask,
    }

    OUT_PATHS = {
        'local': Path('../data/attribution/datasets'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_name',
        type=str,
        default='qasper'
    )
    parser.add_argument(
        '--location',
        type=str,
        default='shared'
    )
    parser.add_argument(
        '--partition',
        type=str,
        default='dev'
    )
    parser.add_argument(
        '--csv',
        action='store_true'
    )
    parser.add_argument(
        '--no_invert_contract_nli_contradiction',
        action='store_false'
    )

    args = parser.parse_args()

    task_name = args.task_name
    location = args.location
    partition_name = args.partition

    print('Program start')
    print(f'Creating attribution dataset from {task_name}-{partition_name}')
    print('Loading config')

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

    print('Loading task')
    task = task_class(
        config,
        stats
    )

    if partition_name == 'dev':
        instances = task.dev_instances
    else:
        instances = task.test_instances

    print('Converting task instances to attribution instances')
    dataset = make_attribution_dataset(
        task_name,
        False,
        instances,
        invert_contract_nli_contradiction_claim=args.no_invert_contract_nli_contradiction
    )
    dataset_statistics = compute_dataset_statistics(dataset)
    print(f'Created attribution dataset from {task_name}-{partition_name}')
    print('Dataset statistics')
    print(json.dumps(dataset_statistics, indent=4))

    if args.csv:
        print('Saving dataset to csv')
        save_attribution_dataset_csv(
            dataset,
            OUT_PATHS[location] / f'{task_name}-{partition_name}.csv'
        )
    else:
        print('Saving dataset to json')
        save_attribution_dataset_jsonl(
            dataset,
            OUT_PATHS[location] / f'{task_name}-{partition_name}.jsonl'
        )
