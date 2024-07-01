import copy
import json
import os
from pathlib import Path
import sys

from evaluation.tasks.qasa_task import QASATask

sys.path.append('')
from typing import List, Dict, Tuple

import nltk
import torch.cuda
import yaml
import pandas as pd
from intertext_graph import Node
from matplotlib import pyplot as plt
import scipy

from attribution_eval.attribution_model import AttributionBaseModel
from evaluation.tasks.contract_nli_task import ContractNLITask
from evaluation.tasks.govreport_task import GovReportTask
from evaluation.tasks.wice_task import WiceTask
from models.retrieve import BaseRetriever
from structformer.input_preparation import map_node_ids_to_nodes
from evaluation.metrics import evaluate_attribution, get_descriptive_stats, get_reference_based_metric
from evaluation.common import BaseTask, Statistics, BasePrediction, BaseInstance, SingleFileDataset, CustomDataset
from evaluation.tasks.qasper_task import QASPERTask
from evaluation.tasks.natural_questions_task import NaturalQuestionsTask
from evaluation.tasks.evidence_inference_task import EvidenceInferenceTask
from config_lib.base_config import BaseConfig
from evaluation.util import parse_answer, is_instance_answerable, is_prediction_answerable, \
    should_prediction_be_considered_for_attribution
from attribution_eval.attribution_dataset import make_attribution_instances_from_base_instance


TASK_CLASSES = [
    QASPERTask,
    NaturalQuestionsTask,
    EvidenceInferenceTask,
    GovReportTask,
    WiceTask,
    ContractNLITask,
    QASATask
]


AUTO_METRICS = {
    'qasper': {
        'metrics': [
            'answer_f1',
            'attribution',
            'unanswerable_f1'
        ],
        'metrics_for_pasting': [
            'answer_f1',
            'attribution',
            'unanswerable_f1'
        ]
    },
    'qasa': {
        'metrics': [
            'rouge_l',
            'attribution',
            'unanswerable_f1'
        ],
        'metrics_for_pasting': [
            'rouge_l',
            'attribution',
            'unanswerable_f1'
        ]
    },
    'natural_questions': {
        'metrics': [
            'answer_f1',
            'attribution',
            'unanswerable_f1'
        ],
        'metrics_for_pasting': [
            'answer_f1',
            'attribution',
            'unanswerable_f1'
        ]
    },
    'evidence_inference': {
        'metrics': [
            'classification_f1',
            'attribution'
        ],
        'metrics_for_pasting': [
            'classification_f1',
            'evidence_f1'
        ]
    },
    'wice': {
        'metrics': [
            'classification_f1',
            'attribution'
        ],
        'metrics_for_pasting': [
            'classification_f1',
            'evidence_f1'
        ]
    },
    'contract_nli': {
        'metrics': [
            'classification_f1',
            'attribution'
        ],
        'metrics_for_pasting': [
            'classification_f1',
            'evidence_f1'
        ]
    },
    'govreport': {
        'metrics': [
            'rouge_l',
            'attribution'
        ],
        'metrics_for_pasting': [
            'rouge_l',
            'attribution'
        ]
    }
}


def get_auto_metrics(task_name: str):
    metrics = AUTO_METRICS[task_name]['metrics']
    metrics_for_pasting = AUTO_METRICS[task_name]['metrics_for_pasting']
    return metrics, metrics_for_pasting


def re_extract_from_raw_generations(
        predictions: List[BasePrediction],
        instances: List[BaseInstance],
        node_id_template: str,
        extraction_mode: str,
        answer_format: str,
        required_aspects: str,
        unanswerable_keywords: List[str],
        answer_has_multiple_statements: bool,
        classes: List[str] = None,
        return_all_extraction_candidates: bool = False
) -> List[BasePrediction]:
    instance_mapping = {
        instance.example_id: instance
        for instance in instances
    }
    new_predictions = []
    for prediction in predictions:
        instance = instance_mapping[prediction.example_id]
        node_id_to_node_mapping = map_node_ids_to_nodes(
            instance.document.nodes,
            node_id_template
        )
        free_text_answer, extraction_nodes = parse_answer(
            prediction.raw_generation,
            instance.extraction_candidates,
            extraction_mode,
            answer_format,
            required_aspects,
            prediction.task_name,
            unanswerable_keywords=unanswerable_keywords,
            answer_has_multiple_statements=answer_has_multiple_statements,
            node_id_to_node_mapping=node_id_to_node_mapping,
            classes=classes,
            return_all_extraction_candidates=return_all_extraction_candidates
        )
        new_prediction = BasePrediction(
            task_name=prediction.task_name,
            example_id=prediction.example_id,
            free_text_answer=free_text_answer,
            extraction_nodes=extraction_nodes,
            raw_generation=prediction.raw_generation
        )
        new_predictions.append(new_prediction)
    return new_predictions


def make_output_for_attribution_annotation(
        instances: List[BaseInstance],
        predictions: List[BasePrediction],
        answer_has_multiple_statements: bool,
        answer_types: List[str]
) -> pd.DataFrame:
    columns = {
        'label': [],
        'claim': [],
        'evidence': [],
        'task_name': [],
        'example_id': [],
        'annotation_idx': [],
        'sentence_idx': [],
        'answer_type': []
    }
    for instance, prediction, answer_type in zip(
            instances, predictions, answer_types
    ):
        attribution_instances = make_attribution_instances_from_base_instance(
            instance,
            answer_has_multiple_statements,
            prediction=prediction,
            answer_type=answer_type,
            skip_empty=False
        )
        for attribution_instance in attribution_instances:
            for column_name in columns:
                columns[column_name].append(getattr(attribution_instance, column_name))
    return pd.DataFrame(columns)


def make_output_for_evidence_position_analysis(
        predictions: List[BasePrediction],
        instances: List[BaseInstance],
        answer_has_multiple_statements: bool
) -> Tuple:
    """

    For each prediction and instance:
    - Get the total number of nodes in the document
    - extract the indices of extraction nodes
      in the complete list of nodes and the ix attributes.
    :param predictions:
    :param instances:
    :param answer_has_multiple_statements:
    :return:
    """
    total_n_nodes = []
    gold_extraction_node_idxs = []
    gold_extraction_node_ixs = []
    predicted_extraction_node_idxs = []
    predicted_extraction_node_ixs = []

    for prediction, instance in zip(predictions, instances):
        total_n_nodes_for_instance = len(instance.document.nodes)
        if answer_has_multiple_statements:
            gold_extraction_node_idxs_for_instance = [
                [
                    [] for _ in extraction_nodes_for_annotation
                ] for extraction_nodes_for_annotation in instance.extraction_nodes
            ]
            gold_extraction_node_ixs_for_instance = [
                [
                    [] for _ in extraction_nodes_for_annotation
                ] for extraction_nodes_for_annotation in instance.extraction_nodes
            ]
            predicted_extraction_node_idxs_for_prediction = [
                [] for _ in prediction.extraction_nodes
            ]
            predicted_extraction_node_ixs_for_prediction = [
                [] for _ in prediction.extraction_nodes
            ]
            for i, n in enumerate(instance.document.nodes):
                # Go over all extraction nodes from instance and check if
                # n is in there
                for j, extraction_nodes_for_annotation in enumerate(instance.extraction_nodes):
                    for k, extraction_nodes_for_annotation in enumerate(extraction_nodes_for_annotation):
                        for extraction_node in extraction_nodes_for_annotation:
                            if n == extraction_node:
                                gold_extraction_node_idxs_for_instance[j][k].append(i)
                                gold_extraction_node_ixs_for_instance[j][k].append(n.ix)
                # Go over all extraction nodes from prediction and check if
                # n is in there
                for j, extraction_nodes_for_annotation in enumerate(prediction.extraction_nodes):
                    for extraction_node in extraction_nodes_for_annotation:
                        if n == extraction_node:
                            predicted_extraction_node_idxs_for_prediction[j].append(i)
                            predicted_extraction_node_ixs_for_prediction[j].append(n.ix)

        else:
            gold_extraction_node_idxs_for_instance = [
                [] for _ in instance.extraction_nodes
            ]
            gold_extraction_node_ixs_for_instance = [
                [] for _ in instance.extraction_nodes
            ]
            predicted_extraction_node_idxs_for_prediction = []
            predicted_extraction_node_ixs_for_prediction = []
            for i, n in enumerate(instance.document.nodes):
                # Go over all extraction nodes from instance and check if
                # n is in there
                for j, extraction_nodes_for_annotation in enumerate(instance.extraction_nodes):
                    for extraction_node in extraction_nodes_for_annotation:
                        if n == extraction_node:
                            gold_extraction_node_idxs_for_instance[j].append(i)
                            gold_extraction_node_ixs_for_instance[j].append(n.ix)
                # Go over all extraction nodes from prediction and check if
                # n is in there
                for extraction_node in prediction.extraction_nodes:
                    if n == extraction_node:
                        predicted_extraction_node_idxs_for_prediction.append(i)
                        predicted_extraction_node_ixs_for_prediction.append(n.ix)
        total_n_nodes.append(total_n_nodes_for_instance)
        gold_extraction_node_idxs.append(gold_extraction_node_idxs_for_instance)
        gold_extraction_node_ixs.append(gold_extraction_node_ixs_for_instance)
        predicted_extraction_node_idxs.append(predicted_extraction_node_idxs_for_prediction)
        predicted_extraction_node_ixs.append(predicted_extraction_node_ixs_for_prediction)

    return (
        total_n_nodes,
        predicted_extraction_node_idxs,
        predicted_extraction_node_ixs,
        gold_extraction_node_idxs,
        gold_extraction_node_ixs
    )


def format_scores_for_pasting(
        results: Dict,
        metric_names: List[str]
) -> str:
    """Format scores to conveniently paste them into a spreadsheet"""
    scores = []
    for metric_name in metric_names:
        if metric_name == 'evidence_f1':
            # Use evidence F1 belonging to first metric
            scores.append(str(results[metric_names[0]]['evidence_f1']))
        else:
            scores.append(str(results[metric_name]['score']))

    return ','.join(scores)


def analyze_predictions(
        predictions: List[BasePrediction],
        instances: List[BaseInstance] | SingleFileDataset,
        metrics: List[str],
        re_extract_from_raw_generation: bool,
        do_post_hoc_extract: bool,
        do_retrieve_then_read: bool,
        do_write_output: bool,
        answer_has_multiple_statements: bool,
        out_dir: Path = None,
        attribution_model: AttributionBaseModel = None,
        attribution_batch_size: int = 2,
        attribution_concatenate_extraction_nodes: bool = True,
        classes: List[str] = None,
        use_all_annotations_for_evidence_f1: bool = False,
        train_instances: List[BaseInstance] | CustomDataset = None,
        post_hoc_retrieval_model: str = 'bm25',
        post_hoc_retrieval_k: int = 3,
        post_hoc_retrieval_threshold: float = None,
        post_hoc_sbert_model_name: str = 'all-mpnet-base-v2',
        retrieve_then_read_model: str = 'bm25',
        retrieve_then_read_k: int = 10,
        retrieve_then_read_threshold: float = None,
        retrieve_then_read_sbert_model_name: str = 'all-mpnet-base-v2',
        metrics_for_pasting: List = None,
        re_extract_node_id_template: str = None,
        re_extract_extraction_mode: str = None,
        re_extract_answer_format: str = None,
        re_extract_required_aspects: str = None,
        re_extract_unanswerable_keywords: List[str] = None,
        re_extract_return_all_extraction_candidates: bool = False
):

    def make_scatter_with_correlation(
            df_: pd.DataFrame,
            x_name_: str,
            y_name_: str,
            do_write_output: bool,
            out_dir: Path = None
    ):
        """Make a scatter plot and print correlation in the plot"""
        # Plot Score vs evidence F1
        plot = df_.plot.scatter(x=x_name_, y=y_name_)
        # Compute correlation
        try:
            r, p = scipy.stats.pearsonr(
                df_[x_name_],
                df_[y_name_]
            )
        except ValueError:
            r, p = 0, 1
        plt.text(0, 0.2, f"R={r:.2f}, p={p:.2f}", horizontalalignment='left', size='medium',
                 weight='semibold')
        x_name_clean = x_name_.replace(" ", "_").lower()
        y_name_clean = y_name_.replace(" ", "_").lower()
        if do_write_output:
            filename = f'{x_name_clean}_vs_{y_name_clean}.png'
            plot.get_figure().savefig(
                out_dir / filename
            )
        plt.close()
        return {'r': r, 'p': p}

    def make_analyses_and_output_table(
            questions_: List[str],
            predicted_answers_: List[str],
            predicted_evidences_: List[List[str]],
            gold_answers_: List[str],
            gold_evidences_: List[List[str]],
            answer_scores_: List[float],
            evidence_f1s_: List[float],
            answer_score_name_: str,
            out_dir_: Path,
            do_write_output_: bool,
            answer_types_: List[str] = None,
            attribution_: List[float| int] = None,
            gold_answerability_: List[int] = None,
            predicted_answerability_: List[int] = None,
            total_n_nodes_: List[int] = None,
            predicted_extraction_node_idxs_: List = None,
            predicted_extraction_node_ixs_: List = None,
            gold_extraction_node_idxs_: List = None,
            gold_extraction_node_ixs_: List = None
    ) -> Dict:

        """Output several plots for analysis and output raw data in table"""
        result_dict_ = {}
#        import pdb
#        pdb.set_trace()
        table = pd.DataFrame({
            'Question': questions_,
            'Predicted Answer': predicted_answers_,
            'Gold Answer': gold_answers_,
            answer_score_name_ : answer_scores_,
            'Predicted Evidence': predicted_evidences_,
            'Gold Evidence': gold_evidences_,
            'Evidence F1': evidence_f1s_
        })
        if attribution_ is not None:
            table['Attribution'] = attribution_
        if answer_types_ is not None:
            table['Answer Type'] = answer_types_
        if gold_answerability_ is not None:
            table['Gold Answerability'] = gold_answerability_
        if predicted_answerability_ is not None:
            table['Predicted Answerability'] = predicted_answerability_
        if total_n_nodes_ is not None:
            table['Total N Nodes'] = total_n_nodes_
        if predicted_extraction_node_idxs_ is not None:
            table['Predicted Extraction Node Idxs'] = predicted_extraction_node_idxs_
        if predicted_extraction_node_ixs_ is not None:
            table['Predicted Extraction Node Ixs'] = predicted_extraction_node_ixs_
        if gold_extraction_node_idxs_ is not None:
            table['Gold Extraction Node Idxs'] = gold_extraction_node_idxs_
        if gold_extraction_node_ixs_ is not None:
            table['Gold Extraction Node Ixs'] = gold_extraction_node_ixs_

        df = pd.DataFrame(table)

        if 'Answer Type' in df.columns:
            # Plot score by answer type
            score_by_answer_type = df.groupby('Answer Type')[answer_score_name_].mean()
            score_by_answer_type.plot.bar().get_figure().savefig(
                out_dir / f'{answer_score_name_.replace(" ", "_").lower()}_by_answer_type.png'
            )

        if do_write_output_:
            # Plot score distribution
            df[answer_score_name_].hist().get_figure().savefig(
                out_dir / f'{answer_score_name_.replace(" ", "_").lower()}_dist.png'
            )
        plt.close()

        # Plot Score vs evidence F1
        result_dict_[f'Correlation Evidence F1 - {answer_score_name_}'] = make_scatter_with_correlation(
            df,
            'Evidence F1',
            answer_score_name_,
            do_write_output_,
            out_dir_
        )

        if attribution_ is not None:
            # Remove all 'unanswerable' predictions.
            df_without_predicted_unanswerable = df.loc[
                df['Predicted Answerability'] == 1
            ]
            # Check number of unique values in attribution
            if len(df_without_predicted_unanswerable['Attribution'].unique()) > 2:
                # If there are more than 2 values, make a scatterplot
                result_dict_[f'Correlation Attribution - {answer_score_name_}'] = \
                    make_scatter_with_correlation(
                        df_without_predicted_unanswerable,
                        'Attribution',
                        metric_name,
                        do_write_output,
                        out_dir_
                    )
                result_dict_[f'Correlation Attribution - Evidence F1'] = \
                    make_scatter_with_correlation(
                        df_without_predicted_unanswerable,
                        'Attribution',
                        'Evidence F1',
                        do_write_output,
                        out_dir_
                    )
            else:
                score_by_attribution = df_without_predicted_unanswerable.groupby('Attribution')[[answer_score_name_, 'Evidence F1']].mean()
                # Make barplot for answer score
                result_dict_[f'Attribution - {answer_score_name_} distribution'] = \
                    score_by_attribution[answer_score_name_].to_dict()
                score_by_attribution[answer_score_name_].plot.bar().get_figure().savefig(
                    out_dir / f'{answer_score_name_.replace(" ", "_").lower()}_by_attribution.png'
                )
                # Make barplot for Evidence F1
                result_dict_[f'Attribution - {answer_score_name_} distribution'] = \
                    score_by_attribution['Evidence F1'].to_dict()
                if do_write_output_:
                    score_by_attribution['Evidence F1'].plot.bar().get_figure().savefig(
                        out_dir / f'{answer_score_name_.replace(" ", "_").lower()}_by_attribution.png'
                    )

            # Compare attribution for unanswerable vs answerable instances
            if 'Gold Answerability' in df_without_predicted_unanswerable.columns:
                answerable_attribution_dist = df_without_predicted_unanswerable.groupby('Gold Answerability')['Attribution'].mean()
                result_dict_['Answerable Attribution Dist'] = answerable_attribution_dist.to_dict()
                # Make boxplot
                boxplot = df_without_predicted_unanswerable.boxplot(
                    column='Attribution',
                    by='Gold Answerability'
                )
                boxplot.get_figure().savefig(
                    out_dir / f'gold_answerability_attribution_dist.png'
                )

        if do_write_output_:
            df.to_csv(out_dir / f'predictions_{answer_score_name_.replace(" ", "_").lower()}.csv', index=False)

        return result_dict_

    if metrics is None:
        metrics = []

    # Keep only instances for which there are predictions
    predicted_example_ids = [prediction.example_id for prediction in predictions]
    if type(instances) is list:
        instances = [
            instance for instance in instances
            if instance.example_id in predicted_example_ids
        ]
    else:
        # If instances are in a lazy-loading dataset, load them now
        instances = instances.get_examples_from_ids(predicted_example_ids)

    # Sort instances according to predictions
    instances = sorted(
        instances,
        key=lambda x: predicted_example_ids.index(x.example_id)
    )

    if do_retrieve_then_read:
        print('Doing retrieve then read')
        retrieve_then_read_model = BaseRetriever.load_model(
            'retrieve_and_reduce',
            retrieve_then_read_model,
            answer_has_multiple_statements,
            retrieve_then_read_k,
            retrieve_then_read_threshold,
            instances=instances,
            sbert_model_name=retrieve_then_read_sbert_model_name,
            classes=classes
        )
        instances = [
            retrieve_then_read_model.retrieve_and_reduce(instance)
            for instance in instances
        ]

    # Optionally re-extract evidence from raw generations
    if re_extract_from_raw_generation:
        print('Extracting from raw generations')
        predictions = re_extract_from_raw_generations(
            predictions,
            instances,
            re_extract_node_id_template,
            re_extract_extraction_mode,
            re_extract_answer_format,
            re_extract_required_aspects,
            re_extract_unanswerable_keywords,
            answer_has_multiple_statements,
            classes=classes,
            return_all_extraction_candidates=re_extract_return_all_extraction_candidates
        )

    if do_post_hoc_extract:
        print('Doing post hoc extraction')
        post_hoc_extraction_model = BaseRetriever.load_model(
            'post_hoc',
            post_hoc_retrieval_model,
            answer_has_multiple_statements,
            post_hoc_retrieval_k,
            post_hoc_retrieval_threshold,
            instances=instances,
            sbert_model_name=post_hoc_sbert_model_name,
            classes=classes
        )
        predictions = [
            post_hoc_extraction_model.post_hoc_retrieve_and_update_prediction(
                prediction,
                instance
            ) for prediction, instance in zip(predictions, instances)
        ]

#    import pdb
#    pdb.set_trace()
    # Extract questions, predicted answers and predicted evidence
    questions = [instance.question for instance in instances]
    predicted_answers = [prediction.free_text_answer for prediction in predictions]
    predicted_evidences = []

    metrics = copy.deepcopy(metrics)

    # Determine instance answerability
    gold_answerability = []
    for instance in instances:
        if is_instance_answerable(instance):
            gold_answerability.append(1)
        else:
            gold_answerability.append(0)

    predicted_answerability = []
    for prediction in predictions:
        if should_prediction_be_considered_for_attribution(
            prediction,
            classes=classes
        ):
            predicted_answerability.append(1)
        else:
            predicted_answerability.append(0)

    for prediction in predictions:
        if len(prediction.extraction_nodes) > 0:
            extraction_nodes = prediction.extraction_nodes
            if isinstance(extraction_nodes[0], list):
                predicted_evidence = [
                    n.content
                    for node_list in extraction_nodes
                    for n in node_list
                ]
            else:
                predicted_evidence = [
                    n.content for n in prediction.extraction_nodes
                ]
        else:
            predicted_evidence = []
        predicted_evidences.append(predicted_evidence)

    # Make output for evidence position analysis
    (
        total_n_nodes,
        predicted_extraction_node_idxs,
        predicted_extraction_node_ixs,
        gold_extraction_node_idxs,
        gold_extraction_node_ixs
    ) = make_output_for_evidence_position_analysis(
        predictions,
        instances,
        answer_has_multiple_statements
    )

    # Make dict for output
    result_dict = {
        'N Predictions': len(predictions)
    }

    # Evaluate attribution
    attribution_labels = None
    if 'attribution' in metrics:
        metrics.remove('attribution')
        attribution_data = evaluate_attribution(
            predictions,
            instances,
            predictions[0].task_name,
            attribution_model,
            attribution_concatenate_extraction_nodes,
            attribution_batch_size,
            answer_has_multiple_statements,
            classes=classes
        )
        result_dict['attribution'] = {
            'metric_name': 'attribution',
            'score': attribution_data['score']
        }
        attribution_labels = attribution_data['all_scores']

    answer_types = None
    # Evaluate other metrics
    for metric_name in metrics:
        raw_results = get_reference_based_metric(
            predictions,
            instances,
            metric_name,
            classes=classes,
            use_all_annotations_for_evidence_f1=use_all_annotations_for_evidence_f1,
            answer_has_multiple_statements=answer_has_multiple_statements
        )
        if answer_types is None:
            answer_types = raw_results['answer_types']

        if answer_has_multiple_statements:
            gold_evidences = [[] for _ in raw_results['gold_evidences']]
        else:
            gold_evidences = [[n.content for n in node_list] for node_list in raw_results['gold_evidences']]

#        import pdb
#        pdb.set_trace()
        results = make_analyses_and_output_table(
            questions,
            predicted_answers,
            predicted_evidences,
            raw_results['gold_answers'],
            gold_evidences,
            raw_results['all_scores'],
            raw_results['all_evidence_f1s'],
            raw_results['metric_name'],
            out_dir,
            do_write_output,
            raw_results['answer_types'],
            attribution_labels,
            gold_answerability_=gold_answerability,
            predicted_answerability_=predicted_answerability,
            total_n_nodes_=total_n_nodes,
            predicted_extraction_node_idxs_=predicted_extraction_node_idxs,
            predicted_extraction_node_ixs_=predicted_extraction_node_ixs,
            gold_extraction_node_idxs_=gold_extraction_node_idxs,
            gold_extraction_node_ixs_=gold_extraction_node_ixs
        )
        results['metric_name'] = metric_name
        results['score'] = raw_results['score']
        results['evidence_f1'] = raw_results['evidence_f1']
        results['statistics'] = raw_results['statistics']
        results['score_by_answer_type'] = raw_results['score_by_answer_type']
        result_dict[metric_name] = results

    # Count the number of instances that are predicted as unanswerable or
    # that were parsing errors
    n_unanswerable_predicted = 0
    n_unparsable = 0
    for prediction in predictions:
        if isinstance(prediction.free_text_answer, list):
            free_text_answer = prediction.free_text_answer[0]
        else:
            free_text_answer = prediction.free_text_answer
        if free_text_answer.lower().strip() == 'parsing error':
            n_unparsable += 1
    result_dict['Predicted N Parsing Errors'] = n_unparsable
    result_dict['Predicted Proportion Parsing Errors'] = n_unparsable / len(predictions)

    # Predictive statistics
    prediction_descriptive_stats = get_descriptive_stats(
        predictions=predictions
    )
    result_dict['Predictions Descriptive Statistics'] = \
        prediction_descriptive_stats.to_json_dict()
    instances_descriptive_stats = get_descriptive_stats(
        instances=instances
    )
    result_dict['Instances Descriptive Statistics'] = \
        instances_descriptive_stats.to_json_dict()

    # Output result_dict
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(result_dict, f, indent=4)

    if attribution_model is None:
        attribution_model_name = 'None'
    else:
        attribution_model_name = attribution_model.model_name
    # Output evaluation configuration
    evaluation_config = {
        'metrics': list(metrics),
        're_extract_from_raw_generation': re_extract_from_raw_generation,
        'do_post_hoc_extract': do_post_hoc_extract,
        'answer_has_multiple_statements': answer_has_multiple_statements,
        'attribution_model_name': attribution_model_name,
        'attribution_batch_size': attribution_batch_size,
        'attribution_concatenate_extraction_nodes': attribution_concatenate_extraction_nodes,
        'classes': list(classes) if classes is not None else None,
        'post_hoc_retrieval_model': post_hoc_retrieval_model,
        'post_hoc_retrieval_k': post_hoc_retrieval_k,
        'post_hoc_retrieval_threshold': post_hoc_retrieval_threshold,
        'post_hoc_sbert_model_name': post_hoc_sbert_model_name
    }
    with open(out_dir / 'evaluation_config.json', 'w') as f:
        json.dump(evaluation_config, f, indent=4)

    if do_post_hoc_extract or re_extract_from_raw_generation:
        BaseTask.save_predictions(predictions, out_dir / 'predictions.jsonl')
#    import pdb
#    pdb.set_trace()
    # Output for annotation
    table_for_annotation = make_output_for_attribution_annotation(
        instances,
        predictions,
        answer_has_multiple_statements,
        answer_types
    )
    table_for_annotation.to_csv(
        out_dir / 'output_for_annotation.csv',
        index=False
    )

    if metrics_for_pasting:
        formatted_scores = format_scores_for_pasting(result_dict, metrics_for_pasting)
        result_dict['metrics_for_pasting'] = {
            'metric_names': metrics_for_pasting,
            'scores': formatted_scores
        }

    return result_dict


def load_config(
        hash_to_load: str,
        results_dir: Path,
        all_configs_dir: Path,
        location: str
) -> BaseConfig:
    """This implements two ways to load the config as the placement changed"""
    try:
        # Current placement
        with open(results_dir / 'config.json') as f:
            config_dict = json.load(f)
    except FileNotFoundError:
        # legacy placement
        config_filename = None
        for filename in os.listdir(all_configs_dir):
            if hash_to_load in filename:
                config_filename = filename
                break
        if config_filename is None:
            raise ValueError(f'Did not find config for hash {hash_to_load} in dir {all_configs_dir}')
        with open(all_configs_dir / config_filename) as f:
            config_dict = json.load(f)


    # Adjust location config
    with open(f'config/location/{location}.yaml') as f:
        location_config = yaml.load(f, Loader=yaml.FullLoader)
    config_dict['location'] = location_config

    # Load config json into config object
    config: BaseConfig = BaseConfig.from_dict(config_dict, Path('../config'))

    # Quick fix: Override post hoc retrieval model keys because some runs had
    # faulty configs
    with open(f'config/task/{config.task.task_name}.yaml') as f:
        task_config = yaml.load(f, Loader=yaml.FullLoader)
    config.task.post_hoc_extraction_k = task_config['post_hoc_extraction_k']
    config.task.post_hoc_extraction_model = task_config['post_hoc_extraction_model']
    config.task.post_hoc_extraction_sbert_model_name = task_config['post_hoc_extraction_sbert_model_name']

    return config


def load_predictions(
        hash_to_load: str,
        results_dir: Path,
        all_predictions_dir: Path,
        instances
) -> List[BasePrediction]:
    """This implements two ways to find the predictions file and load it, as
    the file placement changed"""
    # Current placement
    predictions_path = results_dir / 'predictions.jsonl'
    if not os.path.exists(predictions_path):
        # Legacy placement
        predictions_filename = None
        for filename in os.listdir(all_predictions_dir):
            if hash_to_load in filename:
                predictions_filename = filename
                break
        if predictions_filename is None:
            raise ValueError(f'Did not find predictions file for hash {hash_to_load} in dir {all_predictions_dir}')
        predictions_path = all_predictions_dir / predictions_filename

    predictions = BaseTask.load_predictions(predictions_path, instances)

    return predictions


def main(
        hashes: str,
        auto_mode: bool,
        location: str,
        config_dir: Path,
        results_dir: Path,
        out_dir: Path,
        use_first_n_predictions: int,
        re_extract_from_raw_generation: bool,
        do_post_hoc_extract: bool,
        do_retrieve_then_read: bool,
        do_evaluate_answer_f1: bool,
        do_evaluate_rouge_l: bool,
        do_evaluate_bertscore: bool,
        do_evaluate_unanswerable_f1: bool,
        do_evaluate_attribution: bool,
        attribution_model_name: str = 'attrscore',
        concatenate_extraction_nodes_in_attribution: bool = True,
        attribution_predict_binary: bool = True,
        post_hoc_retrieval_model: str = 'bm25',
        post_hoc_retrieval_k: int = 3,
        post_hoc_retrieval_threshold: float = None,
        post_hoc_sbert_model_name: str = 'all-mpnet-base-v2',
        retrieve_then_read_model: str = 'bm25',
        retrieve_then_read_k: int = 10,
        retrieve_then_read_threshold: float = None,
        retrieve_then_read_sbert_model_name: str = 'all-mpnet-base-v2',
        use_all_annotations_for_evidence_f1: str | bool = 'auto',
        metrics_for_pasting: List[str] = None,
        re_extract_return_all_extraction_candidates: bool = False
):
    print('Starting script')
    # Load attribution evaluation model
    attribution_model = None

    task_cache = {}

    for hash_to_load in hashes:
        print('*'*80)
        print(f'Doing evaluation for hash {hash_to_load}')
        # Load the config json of the run to evaluate
        results_dir_for_hash = results_dir / hash_to_load

        config = load_config(hash_to_load, results_dir_for_hash, config_dir, location)

        print(f'Description: {config.description}')

        print(f'Model name: {config.model.model_name}')

        if auto_mode:
            print('Using Auto Mode')
            metrics, metrics_for_pasting = get_auto_metrics(config.task.task_name)
            print('Setting')
            print(f'- metrics to {metrics}')
            print(f'- metrics_for_pasting to {metrics_for_pasting}')
            if do_post_hoc_extract:
                post_hoc_retrieval_model = config.task.post_hoc_extraction_model
                post_hoc_retrieval_k = config.task.post_hoc_extraction_k
                post_hoc_sbert_model_name = config.task.post_hoc_extraction_sbert_model_name
                print(f'- post_hoc_retrieval_model to {post_hoc_retrieval_model}')
                print(f'- post_hoc_retrieval_k to {post_hoc_retrieval_k}')
                print(f'- post_hoc_sbert_model_name to {post_hoc_sbert_model_name}')

        else:
            metrics = config.task.metrics
            if do_evaluate_answer_f1 and 'answer_f1' not in metrics:
                metrics.append('answer_f1')
            if do_evaluate_rouge_l and 'rouge_l' not in metrics:
                metrics.append('rouge_l')
            if do_evaluate_bertscore and 'bertscore' not in metrics:
                metrics.append('bertscore')
            if do_evaluate_unanswerable_f1 and 'unanswerable_f1' not in metrics:
                metrics.append('unanswerable_f1')
            if do_evaluate_attribution and 'attribution' not in metrics:
                metrics.append('attribution')

        if 'attribution' in metrics and attribution_model is None:
            print(f'Loading attribution model {attribution_model_name}')
            attribution_model = AttributionBaseModel.load_model(
                attribution_model_name,
                predict_max_in_batch=not (concatenate_extraction_nodes_in_attribution),
                predict_binary=attribution_predict_binary
            )
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                attribution_model.to(device)

        if use_all_annotations_for_evidence_f1 == 'auto':
            use_all_annotations_for_evidence_f1 = config.task.use_all_annotations_for_evidence_f1

        # Load the task and dataset
        stats = Statistics(
            config.model.model_name,
            config.task.task_name,
            config.description,
            config
        )
        task_class = None
        for tc in TASK_CLASSES:
            if config.task.task_name == tc.task_name:
                task_class = tc
        if task_class is None:
            raise ValueError(f'Did not find task {config.task.task_name}')
        print(f'Loading task {task_class.task_name}')

        if config.task.task_name not in task_cache:
            task: BaseTask = task_class(config, stats)
            task_cache[task.task_name] = task
            # Hack: Convert dev instances object to list to load all instances into
            # memory (for speed-up when re-using the dataset)
            if not isinstance(task.dev_instances, list):
                task._dev_instances = [
                    instance for instance in task.dev_instances
                ]
            # convert test instances object to list
            if not isinstance(task.test_instances, list):
                task._test_instances = [
                    instance for instance in task.test_instances
                ]
        else:
            task = task_cache[config.task.task_name]

        if config.use_dev_as_test_data:
            instances = task.dev_instances
        else:
            instances = task.test_instances

        print('Loading predictions')
        # Load predictions
        predictions = load_predictions(
            hash_to_load,
            results_dir_for_hash,
            config.location.predictions,
            instances,
        )
        if use_first_n_predictions != -1:
            predictions = predictions[:use_first_n_predictions]

        if out_dir == Path('auto'):
            print('Creating new output directory')
            # Use original results directory
            out_dir_to_use = config.location.results / hash_to_load / 'evaluation'
            # Make new subdirectory for new evaluation results
            idx = 1
            dir_exists = os.path.exists(out_dir_to_use / str(idx))
            while dir_exists:
                idx += 1
                dir_exists = os.path.exists(out_dir_to_use / str(idx))
            out_dir_to_use = out_dir_to_use / str(idx)
            if not(os.path.exists(config.location.results / hash_to_load)):
                os.mkdir(config.location.results / hash_to_load)
            if not(os.path.exists(config.location.results / hash_to_load / 'evaluation')):
                os.mkdir(config.location.results / hash_to_load / 'evaluation')
            os.mkdir(out_dir_to_use)
        else:
            out_dir_to_use = out_dir

        # Analyze
        results = analyze_predictions(
            predictions,
            instances,
            metrics,
            re_extract_from_raw_generation,
            do_post_hoc_extract,
            do_retrieve_then_read,
            True,
            config.task.answer_has_multiple_statements,
            out_dir=out_dir_to_use,
            attribution_model=attribution_model,
            attribution_concatenate_extraction_nodes=concatenate_extraction_nodes_in_attribution,
            classes=config.task.classes,
            use_all_annotations_for_evidence_f1=use_all_annotations_for_evidence_f1,
            train_instances=task.train_instances,
            post_hoc_retrieval_model=post_hoc_retrieval_model,
            post_hoc_retrieval_k=post_hoc_retrieval_k,
            post_hoc_retrieval_threshold=post_hoc_retrieval_threshold,
            post_hoc_sbert_model_name=post_hoc_sbert_model_name,
            retrieve_then_read_model=retrieve_then_read_model,
            retrieve_then_read_k=retrieve_then_read_k,
            retrieve_then_read_threshold=retrieve_then_read_threshold,
            metrics_for_pasting=metrics_for_pasting,
            re_extract_node_id_template=config.node_id_template,
            re_extract_extraction_mode=config.extraction_mode,
            re_extract_answer_format=config.answer_format,
            re_extract_required_aspects=config.required_aspects,
            re_extract_unanswerable_keywords=config.unanswerable_keywords,
            re_extract_return_all_extraction_candidates=re_extract_return_all_extraction_candidates
        )
        print('Results:')
        print(json.dumps(results, indent=4))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'hashes',
        type=str,
        nargs='+'
    )
    parser.add_argument(
        '--no_auto_mode',
        action='store_true'
    )
    parser.add_argument(
        '--location',
        type=str,
        default='shared'
    )
    parser.add_argument(
        '--config_dir_path',
        type=Path,
        default=Path('/mnt/beegfs/shared/extraction_benchmark/configs')
    )
    parser.add_argument(
        '--results_dir_path',
        type=Path,
        default=Path('/mnt/beegfs/shared/extraction_benchmark/results')
    )
    parser.add_argument(
        '--out_dir_path',
        type=Path,
        default=Path('auto')
    )
    parser.add_argument(
        '--use_first_n_predictions',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--re_extract',
        action='store_true'
    )
    parser.add_argument(
        '--re_extract_return_all_extraction_candidates',
        action='store_true'
    )
    parser.add_argument(
        '--answer_f1',
        action='store_true'
    )
    parser.add_argument(
        '--rouge_l',
        action='store_true'
    )
    parser.add_argument(
        '--bertscore',
        action='store_true'
    )
    parser.add_argument(
        '--unanswerable_f1',
        action='store_true'
    )
    parser.add_argument(
        '--attribution',
        action='store_true'
    )
    parser.add_argument(
        '--attribution_model_name',
        type=str,
        default='true_nli'
    )
    parser.add_argument(
        '--attribution_no_concatenation',
        action='store_true'
    )
    parser.add_argument(
        '--attribution_predict_probs',
        action='store_true'
    )
    parser.add_argument(
        '--post_hoc_extract',
        action='store_true'
    )
    parser.add_argument(
        '--post_hoc_retrieval_model',
        type=str,
        default='bm25'
    )
    parser.add_argument(
        '--post_hoc_retrieval_k',
        type=int,
        default=3
    )
    parser.add_argument(
        '--post_hoc_retrieval_threshold',
        type=float,
        default=None
    )
    parser.add_argument(
        '--post_hoc_sbert_model_name',
        type=str,
        default='all-mpnet-base-v2'
    )
    parser.add_argument(
        '--retrieve_then_read',
        action='store_true'
    )
    # Same arguments for retrieve then read as for post_hoc
    parser.add_argument(
        '--retrieve_then_read_model',
        type=str,
        default='bm25'
    )
    parser.add_argument(
        '--retrieve_then_read_k',
        type=int,
        default=10
    )
    parser.add_argument(
        '--retrieve_then_read_threshold',
        type=float,
        default=None
    )
    parser.add_argument(
        '--retrieve_then_read_sbert_model_name',
        type=str,
        default='all-mpnet-base-v2'
    )
    parser.add_argument(
        '--use_all_annotations_for_evidence_f1',
        type=str,
        default='auto'
    )
    parser.add_argument(
        '--metrics_for_pasting',
        type=str,
        nargs='+'
    )
    parser.add_argument(
        '--remote_debug',
        action='store_true'
    )
    args = parser.parse_args()

    if args.remote_debug:
        import pydevd_pycharm
        pydevd_pycharm.settrace('10.167.11.14', port=3851, stdoutToServer=True, stderrToServer=True)

    use_all_annotations_for_evidence_f1 = args.use_all_annotations_for_evidence_f1
    if use_all_annotations_for_evidence_f1 != 'auto':
        # convert string to boolean
        use_all_annotations_for_evidence_f1 = use_all_annotations_for_evidence_f1.lower() == 'true'

    main(
        args.hashes,
        not args.no_auto_mode,
        args.location,
        args.config_dir_path,
        args.results_dir_path,
        args.out_dir_path,
        args.use_first_n_predictions,
        args.re_extract,
        args.post_hoc_extract,
        args.retrieve_then_read,
        args.answer_f1,
        args.rouge_l,
        args.bertscore,
        args.unanswerable_f1,
        args.attribution,
        args.attribution_model_name,
        not args.attribution_no_concatenation,
        not args.attribution_predict_probs,
        args.post_hoc_retrieval_model,
        args.post_hoc_retrieval_k,
        args.post_hoc_retrieval_threshold,
        args.post_hoc_sbert_model_name,
        args.retrieve_then_read_model,
        args.retrieve_then_read_k,
        args.retrieve_then_read_threshold,
        args.retrieve_then_read_sbert_model_name,
        use_all_annotations_for_evidence_f1=use_all_annotations_for_evidence_f1,
        metrics_for_pasting=args.metrics_for_pasting,
        re_extract_return_all_extraction_candidates=args.re_extract_return_all_extraction_candidates
    )
