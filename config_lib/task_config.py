from dataclasses import dataclass
from typing import List, Optional

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from config_lib.config_container import ConfigContainer


@dataclass
class BaseTaskConfig(ConfigContainer):
    """
    Base Configuration for all tasks

    :param task_name: Name of the task

    :param train_filenames_path: Name of file with names of files to use for training
        If given, only examples from these files will be used for training.
        The file should be stored in the same directory as the folders with the
        files for train / dev / test
    :param dev_filenames_path: Name of file with names of files to use for validation
        If given, only examples from these files will be used for training.
        The file should be stored in the same directory as the folders with the
        files for train / dev / test
    :param test_filenames_path: Name of file with names of files to use for testing
        If given, only examples from these files will be used for training.
        The file should be stored in the same directory as the folders with the
        files for train / dev / test


    :param deep_or_shallow: Whether to use deep or shallow version of the dataset

    :param max_new_tokens: The maximum number of tokens that need to be generated
        for the task. Used in generate functions.

    :param is_include_node_types: Which node types to include. This overrides
        whatever is set in is_include_node_types in the main config

    :param answer_has_multiple_statements: Whether expected answer consists of
        one or multiple statements (sentences).

    # Prompt
    :param has_statement: Whether to use the prompt template with or without statement.
    :param null_answer_string: String that represents a null answer.
    :param example_ids: Ids of instances to use as examples. Ids can be int or str
    :param task_input_template: Used to compose prompt

    :param metrics: List of metrics to evaluate for the task. Options:
        - "answer_f1"
        - "bertscore"
        - "rouge_1"
        - "rouge_l"
        - "classification_f1"
        - "unanswerable_f1"
    :param classes: If the task is a classification task, list all classes
    :param use_all_annotations_for_evidence_f1: If True, we use the annotation
        that results in the highest evidence F1 to compute evidence F1. If False,
        we use the annotation that resulted in the highest answer F1 / classification
        F1 ...

    # QASPER
    :param qasper_text_evidence_only: Whether to use only text evidence or also other evidence
    :param count_missing_predictions: In cases of missing questions in the predictions, whether to
        count them into the final result. When true, they count as F1=0

    # Natural Questions
    :param natural_questions_use_first_non_null_answer: Whether to use the first non-null answer as
        gold answer. If false, all gold answers are used.

    # GovReport
    :param govreport_part: Which parts of the govreport dataset to use. Options are
        'gao', 'crs' or 'full' (both gao and crs)
    :param govreport_or_fastfact: If the full govreport summaries are used as
        target or only the short fastfact summaries. Options: 'govreport', 'fastfact'

    # WiCE
    :param wice_claim_or_subclaim: Whether to use the "claim" or the "subclaim"
        part of the wice dataset.

    # QASA
    :param qasa_example_ids_in_test_set: The ids of instances from the
        test set that are added to the (empty) training set to be used as examples

    # Document Pruning
    :param keep_k_random_nodes_when_pruning: When doing pruning, keep this number
        of random paragraph nodes when there are no annotated extraction nodes.

    # Retrieve then read and post hoc
    :param retrieve_then_read_short_k: The number of paragraphs to retrieve
        when using all retrieved paragraphs as evidence.
    :param retrieve_then_read_short_model: The retrieval model to use when
        using all retrieved paragraphs as evidence.
    :param retrieve_then_read_short_sbert_model_name: When using a sentence transformer,
        the specific model to use.
    :param retrieve_then_read_long_k: The number of paragraphs to retrieve when
        not using all retrieved paragraphs as evidence.
    :para retrieve_then_read_long_model: The retrieval model to use when
        using all retrieved paragraphs as evidence.
    :param retrieve_then_read_long_sbert_model_name: When using a sentence transformer,
        the specific model to use.
    :param post_hoc_extraction_k: The number of paragraphs to extract in post hoc
        evidence extraction.
    post_hoc_extraction_model: The retriever to use in post hoc extraction.
    post_hoc_extraction_sbert_model_name: When using a sentence transformer,
        the specific model to use.
    """

    task_name: str = MISSING

    train_filenames: Optional[str] = None
    dev_filenames: Optional[str] = None
    test_filenames: Optional[str] = None

    metrics: List[str] = MISSING
    classes: Optional[List[str]] = None
    use_all_annotations_for_evidence_f1: Optional[bool] = False

    max_new_tokens: int = MISSING

    is_include_node_types: List[str] = MISSING
    deep_or_shallow: Optional[str] = None

    answer_has_multiple_statements: bool = MISSING

    # prompt
    has_statement: bool = MISSING
    null_answer_string: Optional[str] = None
    example_ids: Optional[List] = None
    task_input_template: Optional[str] = None

    # QASPER
    qasper_text_evidence_only: Optional[bool] = None
    count_missing_predictions: Optional[bool] = None

    # Natural Questions
    natural_questions_use_first_non_null_answer: Optional[bool] = None

    # GovReport
    govreport_part: Optional[str] = None

    # WiCE
    wice_claim_or_subclaim: Optional[str] = None

    # QASA
    qasa_example_ids_in_test_set: Optional[List[str]] = None

    # Pruning
    keep_k_random_nodes_when_pruning: Optional[int] = None

    # Retrieve then read and post hoc
    retrieve_then_read_short_k: Optional[int] = None
    retrieve_then_read_short_model: Optional[str] = None
    retrieve_then_read_short_sbert_model_name: Optional[str] = None
    retrieve_then_read_long_k: Optional[int] = None
    retrieve_then_read_long_model: Optional[str] = None
    retrieve_then_read_long_sbert_model_name: Optional[str] = None
    post_hoc_extraction_k: Optional[int] = None
    post_hoc_extraction_model: Optional[str] = None
    post_hoc_extraction_sbert_model_name: Optional[str] = None

cs = ConfigStore.instance()
cs.store(
    name="base_task_config",
    group='task',
    node=BaseTaskConfig
)