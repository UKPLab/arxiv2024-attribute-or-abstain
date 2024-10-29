import collections
import string
from typing import Dict, List
import re

import numpy as np
from bert_score import BERTScorer
from intertext_graph import Node
from rouge_score.rouge_scorer import RougeScorer

from attribution_eval.attribution_dataset import make_attribution_instances_from_base_instance
from evaluation.common import BaseInstance, BasePrediction, BaseDescriptiveStats
from attribution_eval.attribution_model import TRUEModel, AttrScoreModel, AttributionBaseModel
from evaluation.util import find_label, is_instance_answerable, should_prediction_be_considered_for_attribution


def precision(confusion: Dict[str, int]) -> float:
    if confusion["TP"] == 0 and confusion["FP"] == 0:
        return 0  # TODO: is this correct?
    else:
        return confusion["TP"] / (confusion["TP"] + confusion["FP"])


def recall(confusion: Dict[str, int]) -> float:
    if confusion["TP"] == 0 and confusion["FN"] == 0:
        return 0  # TODO: is this correct?
    else:
        return confusion["TP"] / (confusion["TP"] + confusion["FN"])


def f1_score(confusion: Dict[str, int]) -> float:
    prec = precision(confusion)
    reca = recall(confusion)
    if prec + reca == 0:
        return 0
    else:
        return 2 * prec * reca / (prec + reca)


################################################################################
# General evaluation
################################################################################

def get_reference_based_metric(
        predictions,
        instances,
        metric_name,
        classes: List[str] = None,
        use_all_annotations_for_evidence_f1: bool = False,
        task_name: str = None,
        attribution_model_name: str = None,
        attribution_concatenate_extraction_nodes: bool = None,
        attribution_batch_size: int = None,
        answer_has_multiple_statements: bool = None,

):
    # Instances and predictions should be in the same order

    if metric_name == 'classification_f1':
        if classes == None:
            raise ValueError('No classes given')
        return get_classification_f1(
            predictions,
            instances,
            classes,
            use_all_annotations_for_evidence_f1=use_all_annotations_for_evidence_f1
        )
    elif metric_name == 'unanswerable_f1':
        return get_unanswerable_f1(
            predictions,
            instances,
            answer_has_multiple_statements
        )
    elif metric_name == 'attribution':
        # Check if all needed args are given
        missing_attribution_args = [
            arg for arg in [
                task_name,
                attribution_model_name,
                attribution_concatenate_extraction_nodes,
                attribution_batch_size,
                answer_has_multiple_statements
            ] if arg is None
        ]
        if len(missing_attribution_args) > 0:
            raise ValueError(
                f'Missing arguments {missing_attribution_args} to evaluate attribution'
            )
        return evaluate_attribution(
            predictions,
            instances,
            task_name,
            attribution_model_name,
            attribution_concatenate_extraction_nodes,
            attribution_batch_size,
            answer_has_multiple_statements
        )

    data_dict = {}
    # Make dictionary from predictions and instances as common data structure
    for prediction, instance in zip(predictions, instances):
        prediction_dict = {
            'free_text_answer': prediction.free_text_answer,
            'extraction_nodes': prediction.extraction_nodes
        }
        annotation_dicts = []
        for free_text_answer, extraction_nodes, answer_type in zip(
                instance.free_text_answer,
                instance.extraction_nodes,
                instance.answer_type
        ):
            annotation_dict = {
                'free_text_answer': free_text_answer,
                'extraction_nodes': extraction_nodes,
                'answer_type': answer_type
            }
            annotation_dicts.append(annotation_dict)

        data_dict[instance.example_id] = {
            'prediction': prediction_dict,
            'annotations': annotation_dicts
        }

    if metric_name in ['answer_f1']:
        score_data_dict_simple(
            data_dict,
            metric_name
        )
    elif metric_name in ['bertscore', 'rouge_1', 'rouge_l']:
        score_data_dict_batch(data_dict, metric_name)
    else:
        raise ValueError(f'Unknown metric {metric_name}')

    return_dict = {
        'metric_name': metric_name,
        'score': 0,
        'evidence_f1': 0,
        'score_by_answer_type': {},
        'all_scores': [],
        'all_evidence_f1s': [],
        'gold_answers': [],
        'gold_evidences': [],
        'answer_types': [],
        'statistics': {}
    }
    scores_by_answer_type = {}
    # Choose best annotation for each prediction and store score, gold answer,
    # gold evidence, answer type, evidence F1
    for example_id in data_dict:
        annotations = data_dict[example_id]['annotations']
        scores = [
            annotation['score'] for annotation in annotations
        ]
        argmax_score = np.argmax(scores)
        max_score = annotations[argmax_score]['score']
        best_gold_answer = annotations[argmax_score]['free_text_answer']
        best_gold_evidence = annotations[argmax_score]['extraction_nodes']
        best_answer_type = annotations[argmax_score]['answer_type']
        if best_answer_type in scores_by_answer_type:
            scores_by_answer_type[best_answer_type].append(max_score)
        else:
            scores_by_answer_type[best_answer_type] = [max_score]
        if use_all_annotations_for_evidence_f1:
            evidence_f1 = max(
                paragraph_f1_score(
                    data_dict[example_id]['prediction']['extraction_nodes'],
                    annotation_dict['extraction_nodes']
                ) for annotation_dict in annotations
            )
        else:
            evidence_f1 = paragraph_f1_score(
                data_dict[example_id]['prediction']['extraction_nodes'],
                best_gold_evidence
            )
        return_dict['all_scores'].append(max_score)
        return_dict['all_evidence_f1s'].append(evidence_f1)
        return_dict['gold_answers'].append(best_gold_answer)
        return_dict['gold_evidences'].append(best_gold_evidence)
        return_dict['answer_types'].append(best_answer_type)

    return_dict['score'] = np.mean(return_dict['all_scores'])
    return_dict['evidence_f1'] = np.mean(return_dict['all_evidence_f1s'])
    return_dict['score_by_answer_type'] = {
        answer_type: np.mean(scores)
        for answer_type, scores in scores_by_answer_type.items()
    }

    return return_dict


def score_data_dict_simple(
        data_dict: Dict[str, Dict],
        metric_name
):
    METRIC_MAP = {
        'answer_f1': token_f1_score
    }
    score_func = METRIC_MAP[metric_name]
    for example_id in data_dict:
        prediction = data_dict[example_id]['prediction']['free_text_answer']
        if isinstance(prediction, list):
            prediction = ' '.join(prediction)
        for i, annotation_dict in enumerate(data_dict[example_id]['annotations']):
            annotation = annotation_dict['free_text_answer']
            if isinstance(annotation, list):
                annotation = ' '.join(annotation)
            score = score_func(prediction, annotation)
            data_dict[example_id]['annotations'][i]['score'] = score
    return


def score_data_dict_batch(
        data_dict,
        metric_name
):

    METRIC_MAP = {
        'bertscore': bert_scores,
        'rouge_1': rouge_1_scores,
        'rouge_l': rouge_l_scores
    }
    score_func = METRIC_MAP[metric_name]

    # Make lists of equal length containing the predicted answers,
    # gold answers and
    # We need this to increase efficiency (all BERTscores are computed
    # in a single batch)
    predictions = []
    annotations = []
    # Go over all instances
    for example_id in data_dict:
        prediction = data_dict[example_id]['prediction']['free_text_answer']
        if isinstance(prediction, list):
            prediction = ' '.join(prediction)
        for i, annotation_dict in enumerate(data_dict[example_id]['annotations']):
            annotation = annotation_dict['free_text_answer']
            if isinstance(annotation, list):
                annotation = ' '.join(annotation)

            annotations.append(annotation)
            predictions.append(prediction)

    scores = score_func(predictions, annotations)

    idx = 0
    for example_id in data_dict:
        for i in range(len(data_dict[example_id]['annotations'])):
            data_dict[example_id]['annotations'][i]['score'] = float(scores[idx])
            idx += 1

    assert idx == len(scores)

    return


def bert_scores(
        predictions: List[str],
        annotations: List[str]
):
    # Initialize BERT Scorer
    scorer = BERTScorer(
        lang='en',
        rescale_with_baseline=True
    )
    # Get scores for each prediction-reference pair
    _, _, f1 = scorer.score(predictions, annotations)

    return f1


def rouge_1_scores(
        predictions: List[str],
        annotations: List[str]
):
    return rouge_scores(
        predictions,
        annotations,
        'rouge1'
    )


def rouge_l_scores(
        predictions: List[str],
        annotations: List[str]
):
    return rouge_scores(
        predictions,
        annotations,
        'rougeL'
    )


def rouge_scores(
        predictions: List[str],
        annotations: List[str],
        metric_name: str
):
    scores = []
    scorer = RougeScorer([metric_name])
    for prediction, annotation in zip(predictions, annotations):
        scores.append(scorer.score(prediction, annotation)[metric_name].fmeasure)
    return scores


################################################################################
# Classification F1
################################################################################


def get_classification_f1(
        predictions: List[BasePrediction],
        instances: List[BaseInstance],
        classes: List[str],
        use_all_annotations_for_evidence_f1: bool = False
):
    """
    Compute classification f1 and class confusions.
    If count_missing_predictions is True, for each missing prediction a random
    label is chosen.
    It is assumed that the predicted label is in prediction.free_text_answer and
    the ground truth labels are in instance.free_text_answer.
    :param predictions: List of predictions
    :param instances: List of instances
    :param count_missing_predictions: Whether missing predictions should count
        towards the overall score.
    :param classes: The list of possible labels.
    :return:
    """
    classes = [class_name.lower().strip() for class_name in classes]
    # Ensure that earlier classes in list are not substrings of later classes
    for i in range(len(classes[:-1])):
        for j in range(len(classes[i+1:])):
            j = i + j + 1
            assert not classes[i] in classes[j]

    class_confusion = {
        label: collections.Counter() for label in classes
    }

    return_dict = {
        'metric_name': 'classification_f1',
        'score': 0,
        'evidence_f1': 0,
        'score_by_answer_type': {},
        'answer_types': [],
        'all_scores': [],
        'all_evidence_f1s': [],
        'gold_answers': [],
        'gold_evidences': [],
        'statistics': {
            'gold_class_distribution': {
                label: 0 for label in classes
            },
            'predicted_class_distribution': {
                label: 0 for label in classes
            }
        }
    }
    return_dict['statistics']['predicted_class_distribution']['none'] = 0

    for prediction, instance in zip(predictions, instances):
        predicted_label = find_label(
            classes,
            prediction.free_text_answer
        )
        if predicted_label is None:
            predicted_label = 'none'
        # classification: evaluate for each ground truth and choose the best one
        class_confusions_for_prediction = []
        scores_here = []

        for gold_label in instance.free_text_answer:
            gold_label = gold_label.lower().strip()
            class_confusion_for_annotation = {
                label: collections.Counter() for label in classes
            }
            for label in classes:
                # We check for each possible class label
                if gold_label == label and predicted_label == label:
                    class_confusion_for_annotation[label]["TP"] += 1
                elif gold_label == label and predicted_label != label:
                    class_confusion_for_annotation[label]["FN"] += 1
                elif gold_label != label and predicted_label == label:
                    class_confusion_for_annotation[label]["FP"] += 1
                else:
                    class_confusion_for_annotation[label]["TN"] += 1
            score_here = sum(
                class_confusion_for_annotation[label]["TP"] + class_confusion_for_annotation[label]["TN"]
                for label in classes
            )

            class_confusions_for_prediction.append(class_confusion_for_annotation)
            scores_here.append(score_here)

        # choose the 'best-fitting' ground truth/evaluation
        best_idx = scores_here.index(max(scores_here))
        best_gold_label = instance.free_text_answer[best_idx].lower().strip()

        best_gold_evidence = instance.extraction_nodes[best_idx]
        # compute accuracy score
        score = 1.0 if predicted_label == best_gold_label else 0.0
        best_class_confusion_here = class_confusions_for_prediction[best_idx]
        for label in classes:
            class_confusion[label] += best_class_confusion_here[label]
        if use_all_annotations_for_evidence_f1:
            evidence_f1 = max(
                paragraph_f1_score(
                    prediction.extraction_nodes,
                    gold_evidence
                ) for gold_evidence in instance.extraction_nodes
            )
        else:
            evidence_f1 = paragraph_f1_score(
                prediction.extraction_nodes,
                best_gold_evidence
            )
        return_dict['all_scores'].append(score)
        return_dict['all_evidence_f1s'].append(evidence_f1)
        return_dict['gold_evidences'].append(best_gold_evidence)
        return_dict['gold_answers'].append(best_gold_label)
        return_dict['answer_types'].append(best_gold_label)

        return_dict['statistics']['gold_class_distribution'][best_gold_label] += 1
        return_dict['statistics']['predicted_class_distribution'][predicted_label] += 1

    # compute precisions, recalls, and F1 scores for classification
    class_precisions = {}
    class_recalls = {}
    class_f1_scores = {}
    for label in classes:
        class_precisions[label] = precision(class_confusion[label])
        class_recalls[label] = recall(class_confusion[label])
        class_f1_scores[label] = f1_score(class_confusion[label])
    macro_f1_score = float(np.mean(np.array(
        [class_f1_scores[label] for label in classes]
    )))
    return_dict['score'] = macro_f1_score
    return_dict['evidence_f1'] = np.mean(return_dict['all_evidence_f1s'])
    return_dict['score_by_answer_type'] = {
        'recall': class_recalls,
        'precision': class_precisions,
        'f1': class_f1_scores
    }

    return return_dict


################################################################################
# Unanswerable classification
################################################################################


def get_unanswerable_f1(
        predictions: List[BasePrediction],
        instances: List[BaseInstance],
        answer_has_multiple_statements: bool
) -> Dict:
    """"""
    # Find predictions that contain one of the unanswerable keywords and replace
    # free text answer with 'unanswerable', else keep original prediction
    # Unanswerable keywords (taken from Slobodkin et al, 2023)

    # free text answer with 'unanswerable', else
    new_predictions = []
    for prediction in predictions:
        free_text_answer = prediction.free_text_answer
        if answer_has_multiple_statements:
            free_text_answer = ' '.join(free_text_answer).strip().lower()
        if not free_text_answer == 'unanswerable':
            free_text_answer = 'answerable'
        new_predictions.append(
            BasePrediction(
                prediction.task_name,
                prediction.example_id,
                free_text_answer,
                prediction.extraction_nodes,
                prediction.raw_generation
            )
        )

    # Make new list of instances:
    # When all annotators annotated "unanswerable": Set instance as unanswerable
    # Else: Set instance as answerable
    new_instances = []
    for instance in instances:
        if is_instance_answerable(instance):
            free_text_answer = ['answerable']
        else:
            free_text_answer = ['unanswerable']

        new_instances.append(BaseInstance(
            instance.task_name,
            instance.example_id,
            instance.document,
            instance.prompt,
            instance.question,
            instance.statement,
            instance.extraction_level,
            instance.extraction_candidates,
            free_text_answer,
            instance.answer_type,
            instance.extraction_nodes
        ))

    return_dict = get_classification_f1(
        new_predictions,
        new_instances,
        ['unanswerable', 'answerable']
    )
    return_dict['metric_name'] = 'unanswerable_f1'
    # Compute macro f1
    return_dict['score'] = (
        (
            return_dict['score_by_answer_type']['f1']['unanswerable']
            + return_dict['score_by_answer_type']['f1']['answerable']
        ) / 2
    )

    return return_dict


################################################################################
# Basic Statistics
################################################################################


def get_mean_n_words(
        texts: List[str] | List[List[str]]
):
    """
    Calculate the mean number of words in a list of texts
    If each text consists of a list of texts, join them
    """
    if len(texts) > 0:
        if isinstance(texts[0], List):
            texts = [
                ' '.join([t for t in text_list]) for text_list in texts
            ]
    return np.mean([len(text.split(' ')) for text in texts])


def get_mean_free_text_answer_length(
        instances: List[BaseInstance] = None,
        predictions: List[BasePrediction] = None
) -> float:
    """Calculate the mean number of words in the instances or predictions
    If there are multiple annotations take all into account"""
    if (instances is None) and (predictions is None):
        raise ValueError('Both instances and predictions are None')
    elif (instances is not None) and (predictions is not None):
        raise ValueError('Only instances or only predictions should be passed')
    elif instances is not None:
        free_text_answers = [
            answer for instance in instances for answer in instance.free_text_answer
        ]
    else:
        free_text_answers = [
            prediction.free_text_answer
            for prediction in predictions
        ]
    return get_mean_n_words(free_text_answers)


def get_mean_n_extraction_nodes(
        instances: List[BaseInstance] = None,
        predictions: List[BasePrediction] = None
) -> float:
    """Calculate the mean number of extraction nodes in the instances or predictions."""
    if (instances is None) and (predictions is None):
        raise ValueError('Both instances and predictions are None')
    elif (instances is not None) and (predictions is not None):
        raise ValueError('Only instances or only predictions should be passed')
    elif instances is not None:
        n_extraction_nodes = []
        for instance in instances:
            for annotated_extraction_nodes in instance.extraction_nodes:
                if len(annotated_extraction_nodes) > 0:
                    if isinstance(annotated_extraction_nodes[0], list):
                        n_extraction_nodes.extend([
                            len(l) for l in annotated_extraction_nodes
                        ])
                    else:
                        n_extraction_nodes.append(len(annotated_extraction_nodes))
                else:
                    n_extraction_nodes.append(0)

    else:
        n_extraction_nodes = []
        for prediction in predictions:
            if len(prediction.extraction_nodes) > 0:
                if isinstance(prediction.extraction_nodes[0], list):
                    n_extraction_nodes.extend([len(l) for l in prediction.extraction_nodes])
                else:
                    n_extraction_nodes.append(len(prediction.extraction_nodes))
            else:
                n_extraction_nodes.append(0)
    mean_n_extraction_nodes = np.mean(n_extraction_nodes)
    return mean_n_extraction_nodes


def get_descriptive_stats(
        instances: List[BaseInstance] = None,
        predictions: List[BasePrediction] = None
) -> BaseDescriptiveStats:
    """
    Calculate the descriptive statistics for the instances or predictions.
    Currently, we compute the mean number of words in free_text_answer and
    the mean number of extraction nodes.
    """
    if (instances is None) and (predictions is None):
        raise ValueError('Both instances and predictions are None')
    elif (instances is not None) and (predictions is not None):
        raise ValueError('Only instances or only predictions should be passed')
    elif instances is not None:
        mean_free_text_answer_length = get_mean_free_text_answer_length(
            instances=instances
        )
        mean_n_extraction_nodes = get_mean_n_extraction_nodes(
            instances=instances
        )
    else:
        mean_free_text_answer_length = get_mean_free_text_answer_length(
            predictions=predictions
        )
        mean_n_extraction_nodes = get_mean_n_extraction_nodes(
            predictions=predictions
        )
    descriptive_stats = BaseDescriptiveStats(
        mean_n_extraction_nodes=mean_n_extraction_nodes,
        mean_free_text_answer_length=mean_free_text_answer_length
    )
    return descriptive_stats


################################################################################
# Evaluation utils taken from QASPER
# see https://github.com/allenai/qasper-led-baseline/blob/afd0fb96bf78ce8cd8157639c6f6a6995e4f9089/scripts/evaluator.py
################################################################################


def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1_score(prediction, ground_truth):
    """
    Compute token F1 score
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def paragraph_f1_score(prediction: List, ground_truth: List):
    """
    Compute paragraph F1 score for an instance
    Score is 1 if both prediction and ground truth are empty
    """
    if not ground_truth and not prediction:
        # The question is unanswerable and the prediction is empty.
        return 1.0
    if len(prediction) > 0:
        if isinstance(prediction[0], list):
            # Do recursive call if prediction consists of multiple lists)
            # Requires equal lengths of prediction and ground truth
            return np.mean([
                paragraph_f1_score(sub_prediction, sub_ground_truth)
                for sub_prediction, sub_ground_truth
                in zip(prediction, ground_truth)
            ])
        if isinstance(prediction[0], Node):
            # There seem to be node comparison problems when doing multiprocessing
            # So we convert to string
            prediction = [n.content for n in prediction]
    if len(ground_truth) > 0:
        if isinstance(ground_truth[0], Node):
            # There seem to be node comparison problems when doing multiprocessing
            # So we convert to string
            ground_truth = [n.content for n in ground_truth]
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


################################################################################
# Selective Prediction
################################################################################


def select_answerable_predictions(
        predictions: List[BasePrediction],
        unanswerable_keywords: List[str]
) -> List[BasePrediction]:
    """Remove unanswerable predictions from the list of predictions."""
    filtered_predictions = [
        prediction for prediction in predictions
        if prediction.free_text_answer.lower().strip() not in unanswerable_keywords
    ]
    return filtered_predictions


################################################################################
# Attribution evaluation
################################################################################

ATTRIBUTION_MODEL_CLASSES = [
    TRUEModel,
    AttrScoreModel
]


def evaluate_attribution(
        predictions: List[BasePrediction],
        instances: List[BaseInstance],
        task_name: str,
        attribution_model: AttributionBaseModel,
        concatenate_extraction_nodes: bool,
        batch_size: int,
        answer_has_multiple_statements: bool,
        classes: List[str] = None
):
    """
    Given a list of predictions and instances and an attribution model,
    compute individual attribution scores and an average attribution score.
    To compute the average attribution score, the labels combing from the attribution
    model are binarized (score <= 0.5 -> 0, score > 0.5: 1).
    :param predictions:
    :param instances:
    :param task_name:
    :param attribution_model:
    :param concatenate_extraction_nodes:
    :param batch_size:
    :param answer_has_multiple_statements:
    :param classes:
    :return:
    """
    if not concatenate_extraction_nodes:
        batch_size = 1

    attribution_instances = []
    # Make dict of raw attribution scores for each prediction
    # There can be multiple scores when answer has multiple statements
    raw_attribution_scores = {
        prediction.example_id: []
        for prediction in predictions
    }

    for instance, prediction in zip(
        instances, predictions
    ):
        if answer_has_multiple_statements:
            for _ in prediction.extraction_nodes:
                raw_attribution_scores[prediction.example_id].append(0)
        else:
            raw_attribution_scores[prediction.example_id].append(0)
        # Make attribution instance (i.e. make claim from prediction)
        attribution_instances_for_prediction = make_attribution_instances_from_base_instance(
            instance,
            answer_has_multiple_statements,
            prediction=prediction,
            skip_empty=True
        )
        attribution_instances.extend(attribution_instances_for_prediction)

    # Predict attribution scores
    for i in range(0, len(attribution_instances), batch_size):
        batch = attribution_instances[i: i + batch_size]
        collated_batch = attribution_model.collate_fn(batch)
        predicted_labels = attribution_model.predict(collated_batch)
        for attribution_instance, label in zip(
            batch, predicted_labels
        ):
            # Overwrite default 0 with computed attribution score
            raw_attribution_scores[attribution_instance.example_id][attribution_instance.sentence_idx] = label

    # List of one attribution score per prediction
    aggregated_attribution_scores = []
    # List of attribution scores for only answerable predictions
    # This is used to compute the final average attribution score over all predictions
    aggregated_binarized_attribution_scores_answerable_only = []
    raw_attribution_scores_list = []
    # Aggregate attribution scores for each prediction to get a single score per
    # prediction
    for prediction in predictions:
        example_id = prediction.example_id
        scores = raw_attribution_scores[example_id]
        raw_attribution_scores_list.append(scores)
        mean_score_for_prediction = np.mean(scores)
        aggregated_attribution_scores.append(mean_score_for_prediction)
        if should_prediction_be_considered_for_attribution(
                prediction,
                classes
        ):
            # Compute attrbutability score only for answerable predictions
            # Binarize scores
            binary_scores = scores
            if not attribution_model.predict_binary:
                binary_scores = [
                    1 if score > 0.5 else 0
                    for score in scores
                ]
            mean_binary_score_for_prediction = np.mean(binary_scores)
            aggregated_binarized_attribution_scores_answerable_only.append(mean_binary_score_for_prediction)

    # Compute final score as mean for all answerable predictions
    attribution_score = np.mean(aggregated_binarized_attribution_scores_answerable_only)

    return {
        'score_name': 'attribution',
        'score': attribution_score,
        'all_scores': aggregated_attribution_scores,
        'raw_scores': raw_attribution_scores_list
    }
