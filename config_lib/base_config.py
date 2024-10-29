import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Optional, Dict
import logging

from hydra import initialize, compose
from omegaconf import MISSING, DictConfig
from hydra.core.config_store import ConfigStore
import git

from config_lib.config_container import ConfigContainer
from config_lib.location_config import BaseLocationConfig
from config_lib.model_config import BaseModelConfig
from config_lib.task_config import BaseTaskConfig

logger = logging.getLogger(__name__)


###############################################################################

@dataclass
class BaseConfig(ConfigContainer):
    """
    This class defines the base config for the project.
    :param location: Paths to the data and the models.

    :param slurm_job_id: The id of the slurm job.
    :param description: A description of the run.
    :param commit_hash: The commit hash of the code used for the run.

    :param instances_path: Path to folder containing pre-processed instances
        as "train.jsonl", "dev.jsonl", "test.jsonl"

    :param model_name: The name of the model. E.g. "longt5".
    :param remote_debug: Whether to enable remote debugging.
    :param do_train: Whether to run training.
    :param use_dev_as_test_data: Whether to use the dev data as test data.
    :param use_first_n_test_instances: Use only the first n instances in testing.
        When given -1, all test instances are used.
    :param load_model: Whether to load a model from the models directory using the hash_to_load.
    :param hash_to_load: The hash of the model to load.
    :param checkpoint_step_idx: The step at which the checkpoint was saved. If
        not given, using the final checkpoint saved after training.
    :param save_predictions: Whether to save the predictions.
    :param log_test_predictions: Whether to log individual predictions when testing
    :param log_val_predictions: Whether to log individual predictions when validating
    :param log_loss_every_n_batches: How often to log the loss to tensorboard.
    :param use_cache: Whether to use the cache.
    :param lr_scheduler_type: Type of Learning rate scheduler to use

    :param max_depth: The maximum depth of the tree that is considered for node depth infusion.
    :param node_types: The node types that are considered for node type infusion.

    :param is_mode: The mode for the input sequence. Can be "vanilla", "text_with_node_boundaries", "text_with_node_types", "text_with_node_depths".
    :param is_replace_newlines: Whether to replace newlines with spaces.
    :param is_do_close: Whether to add closing tags.
    :param is_node_separator: The separator between nodes.
    :param is_include_node_types: The node types to include.
        This can be overridden by the tasks
    :param is_use_core_node_types_only: Whether to use only the core node types ("p", "title", "abstract", "article-title")
    :param is_use_bos_eos_token: Whether to use the default BOS and EOS tokens of the tokenizer when mode is "text_with_node_boundaries".
    :param is_bos_token: The BOS token to use when mode is "text_with_node_boundaries" and is_use_bos_eos_token is True.
    :param is_eos_token: The EOS token to use when mode is "text_with_node_boundaries" and is_use_bos_eos_token is True.

    :param do_prune_to_extraction_nodes: If True, input documents are pruned to
        annotated extraction nodes only. If there are no annotated extraction
        nodes, k random nodes are kept (determined in task config).
    :param keep_structure_when_pruning: If True, all title and section title nodes
        are kept alongside the annotated extraction nodes.

    :param precision: The precision to use (from pytorch lightning).
    :param dataloader_num_workers: The number of processes to use for the dataloader.
    :param use_vllm: Whether to use vllm for inference. Will throw an error
        when trying to use with load_model or training

    :param random_seed: The random seed to use.
    :param deterministic_trainer: Whether to use a deterministic trainer.

    :param required_aspects: str, one of
        "answer_and_segments"
        "answer_only"
        "segments_only"
    :param extraction_mode: str, one of
        "text": The model should quote the text of relevant nodes
        "node_id": The model should generate the ids of relevant nodes
    :param answer_format: str, the format of the answer, one of
        "json": The model should generate a json dict with keys "answer" and
            "segments" (depending on the required aspects)
        "text": The model should generate the answer in a standardized textual
            format (controlled by evaluation/util/make_format_explanation()).
        "structured_text": The model should generate an answer and a markdown
            list of extracted segments

    :param do_post_hoc_extract: Whether to do post-hoc extraction.
    :param post_hoc_retrieval_model: The model to use for post-hoc retrieval.
    :param post_hoc_retrieval_k: The number of nodes to retrieve for post-hoc extraction.
    :param post_hoc_retrieval_threshold: The threshold score for post-hoc extraction.
    :param post_hoc_sbert_model_name: The name of the sentence-transformers model to use for post-hoc extraction.

    :param do_retrieve_then_read: Whether to do retrieve-then-read.
    :param retrieve_then_read_model: The model to use for retrieve-then-read.
    :param retrieve_then_read_k: The number of nodes to retrieve for retrieve-then-read.
    :param retrieve_then_read_threshold: The threshold score for retrieve-then-read.
    :param retrieve_then_read_sbert_model_name: The name of the sentence-transformers model to use for retrieve-then-read.
    :param return_all_extraction_candidates: Whether to return all extraction candidates as extraction nodes

    # Prompt
    system_prompt, task_explanation, format_explanation and example are
    automatically set in model initialization (BaseModel.set_prompt_elements()).
    They are part of the config to document them after evaluation.
    :param system_prompt: The system prompt for the model.
        Set in BaseModel.set_prompt_elements()
    :param system_prompt_idx: The identifier of the system prompt.
        See config/prompts.yaml for available system prompts.
    :param task_explanation: The explanation for the task. Set in
        BaseModel.set_prompt_elements().
    :param task_explanation_idx: The identifier of the task explanation. See
        config/prompts.yaml for available task explanations.
    :param format_explanation: The explanation of the output format. Set in
        BaseModel.set_prompt_elements().
    :param format_explanation_idx: The identifier of the format explanation. When
        set to "auto", the explanation is automatically chosen according to
        extraction_mode and answer_format. See config/prompts.yaml for available
        format explanations.
    :param example: The examples that are shown to the model. These are automatically
        generated from training instances in BaseModel.set_prompt_elements().
    :param n_examples: How many examples to show to the model.
    :param shorten_example_docs: Whether documents in examples should be shortened
        to save space.
    :param example_doc_max_n_chars_per_node: If documents in examples should be shortened,
        determines the maximum number of characters each node in an example doc
        can have
    :param example_use_n_surrounding_nodes: When shorten_example_docs is
        'to_extraction_nodes', this parameter determines the number of nodes in
        the vicinity of extraction nodes to keep.
    :param node_id_template: How to format the node id (e.g. "<<Segment-{node_id}>>
    :param format_explanation_n_dummy_segments: How many dummy segments should be
        shown to the model in the format explanation. See
        evaluation/util/make_format_explanation().
    :param prompt_input_variables: The variables that can be used in the prompt.
        'task_input' is further defined by task_input_variables. Each model
        defines their own prompt template.
    :param task_input_variables: The variables that can be used in task input.
        Each task defines its own task input template.

    # OpenAI
    :param openai_api_version: The date of the OpenAI API version as a "YYYY-MM-DD" string
    :param openai_api_type: The type of the OpenAI API, typically "azure"

    # Unanswerable keywords
    :param unanswerable_keywords: Strings that a model can use to signify
        unanswerability.
    """
    location: BaseLocationConfig = MISSING
    model: Optional[BaseModelConfig] = None
    task: BaseTaskConfig = MISSING

    slurm_job_id: str = MISSING
    description: str = MISSING
    commit_hash: str = MISSING

    instances_path: Optional[Path] = None

    # Extraction
    required_aspects: str = MISSING  # Options: answer_and_segments, answer_only, segments_only
    extraction_mode: str = MISSING  # Options: text, node_id
    answer_format: str = MISSING  # Options: text, structured_text, json

    do_post_hoc_extract: Optional[bool] = None
    post_hoc_retrieval_model: Optional[str] = None
    post_hoc_retrieval_k: Optional[int] = None
    post_hoc_retrieval_threshold: Optional[float] = None
    post_hoc_sbert_model_name: Optional[str] = None

    do_retrieve_then_read: Optional[Union[str, bool]] = None
    retrieve_then_read_model: Optional[str] = None
    retrieve_then_read_k: Optional[int] = None
    retrieve_then_read_threshold: Optional[float] = None
    retrieve_then_read_sbert_model_name: Optional[str] = None
    return_all_extraction_candidates: Optional[bool] = None

    remote_debug: bool = MISSING
    do_train: bool = MISSING
    use_dev_as_test_data: bool = MISSING
    use_first_n_test_instances: Optional[int] = None
    load_model: bool = MISSING
    hash_to_load: Optional[str] = None
    checkpoint_step_idx: Optional[int] = None
    save_predictions: bool = MISSING
    log_test_predictions: bool = MISSING
    log_val_predictions: bool = MISSING
    log_loss_every_n_batches: int = MISSING
    use_cache: bool = MISSING
    lr_scheduler_type: str = MISSING

    # input sequence
    is_mode: str = MISSING
    is_replace_newlines: bool = MISSING
    is_node_separator: str = MISSING
    is_include_node_types: List[str] = MISSING

    # Document Pruning
    do_prune_to_extraction_nodes: Optional[bool] = None
    keep_structure_when_pruning: Optional[bool] = None

    # engineering
    precision: Union[str, int] = MISSING
    dataloader_num_workers: int = MISSING
    use_vllm: Optional[bool] = None

    # Randomness
    random_seed: int = MISSING
    deterministic_trainer: bool = MISSING

    system_prompt: Optional[str] = None
    system_prompt_idx: Optional[str] = None
    task_explanation: Optional[str] = None
    task_explanation_idx: Optional[str] = None
    format_explanation: Optional[str] = None
    format_explanation_idx: Optional[str] = None
    example: Optional[str] = None
    n_examples: Optional[int] = None
    shorten_example_docs: Optional[str] = None
    example_doc_max_n_chars_per_node: Optional[int] = None
    example_use_n_surrounding_nodes: Optional[int] = None
    node_id_template: Optional[str] = None
    format_explanation_n_dummy_segments: Optional[int] = None
    prompt_input_variables: Optional[List[str]] = None
    task_input_variables: Optional[List[str]] = None

    # OpenAI
    openai_api_version: Optional[str] = None
    openai_api_type: Optional[str] = None

    # Unanswerable keywords
    unanswerable_keywords: Optional[List[str]] = None

    @classmethod
    def from_dict(
            cls,
            config_dict: Dict,
            config_path: Path
    ):
        """Get a config object for the provided config dict (which is typically
        loaded from a json file).
        A default config is loaded, and then all keys provided in the config_dict
        are overridded."""
        def override_config(
                config,
                d: Dict
        ):
            """For each key in the provided dictionary d, check if there is a
            corresponding field in the config object. If yes, override it."""
            for k in d:
                try:
                    setattr(config, k, d[k])
                except AttributeError:
                    print(k)
                    pass

        # Load default config
        with initialize(version_base=None, config_path=str(config_path)):
            cfg = compose(
                config_name="config",
                overrides=[
                    f'task={config_dict["task"]["task_name"]}',
                ]
            )

        # Go over all keys and override
        model_config = config_dict.pop('model')
        task_config = config_dict.pop('task')
        location_config = config_dict.pop('location')
        override_config(cfg, config_dict)
        override_config(cfg.model, model_config)
        override_config(cfg.task, task_config)
        override_config(cfg.location, location_config)

        return cfg


cs = ConfigStore.instance()
cs.store(
    name="base_config",
    node=BaseConfig
)


def init_config(
        config: DictConfig | BaseConfig,
):
    # Set commit hash
    _set_commit_hash(config)

    # Set input sequence mode (is_mode) depending on extraction mode
    if (
        (config.extraction_mode == 'node_id')
        and config.required_aspects in ['answer_and_segments', 'segments_only']
    ):
        logger.info('Setting is_mode to "text_with_node_ids"')
        config.is_mode = 'text_with_node_ids'

    # Override is_include_node_types by task-specific value if set in task
    if config.task.is_include_node_types:
        config.is_include_node_types = config.task.is_include_node_types

    # Set the number of instances to a maximum of 100 when using GPT 4
    if config.model.model_name == 'gpt4-turbo-128k' and config.use_first_n_test_instances > 100:
        logger.info('Setting use_first_n_test_instances to 100 for GPT-4')
        config.use_first_n_test_instances = 100

    # Use vllm when not training or loading pretrained model
    if not config.do_train and not config.load_model:
        try:
            from vllm import LLM
            config.use_vllm = True
            logger.info('Using VLLM for inference')
        except ImportError:
            config.use_vllm = False
            logger.info('VLLM not available, using regular hf inference')

    # Use batch size 10 with vllm
    if config.use_vllm:
        logger.info('Setting batch size to 10 for VLLM')
        config.model.batch_size = 10

    # Set example config specific to dataset
    if config.task.task_name == 'govreport':
        logger.info('Setting example config for GovReport')
        logger.info('Setting example_doc_max_n_chars_per_node to 100')
        logger.info('Setting example_use_n_surrounding_nodes to 0')
        config.example_doc_max_n_chars_per_node = 100
        config.example_use_n_surrounding_nodes = 0
        if str(os.getcwd()).startswith('/mnt'):
            logger.info('Setting instances_path for GovReport')
            config.instances_path = '/mnt/beegfs/shared/extraction_benchmark/datasets/govreport/attributed_bm25'
    else:
        logger.info('Setting example config for other datasets')
        logger.info('Setting example_doc_max_n_chars_per_node to 10000')
        logger.info('Setting example_use_n_surrounding_nodes to 2')
        config.example_doc_max_n_chars_per_node = 10000
        config.example_use_n_surrounding_nodes = 2

    if config.do_retrieve_then_read:
        # Return all extraction candidates when doing retrieve then read with k<=5
        # for wice or k<=2 for other datasets
        if config.do_retrieve_then_read == 'short':
            logger.info('Setting retrieve_then_read config for short')
            logger.info('Setting return_all_extraction_candidates to True')
            logger.info('Setting required_aspects to "answer_only"')
            logger.info(f'Setting retrieve_then_read_k to {config.task.retrieve_then_read_short_k}')
            logger.info(f'Setting retrieve_then_read_model to {config.task.retrieve_then_read_short_model}')
            logger.info(f'Setting retrieve_then_read_sbert_model_name to {config.task.retrieve_then_read_short_sbert_model_name}')
            config.retrieve_then_read_k = config.task.retrieve_then_read_short_k
            config.retrieve_then_read_model = config.task.retrieve_then_read_short_model
            config.retrieve_then_read_sbert_model_name = config.task.retrieve_then_read_short_sbert_model_name
            config.return_all_extraction_candidates = True
            config.required_aspects = 'answer_only'
        elif config.do_retrieve_then_read == 'long':
            logger.info('Setting retrieve_then_read config for long')
            logger.info(f'Setting retrieve_then_read_k to {config.task.retrieve_then_read_long_k}')
            logger.info(f'Setting retrieve_then_read_model to {config.task.retrieve_then_read_long_model}')
            logger.info(f'Setting retrieve_then_read_sbert_model_name to {config.task.retrieve_then_read_long_sbert_model_name}')
            config.retrieve_then_read_k = config.task.retrieve_then_read_long_k
            config.retrieve_then_read_model = config.task.retrieve_then_read_long_model
            config.retrieve_then_read_sbert_model_name = config.task.retrieve_then_read_long_sbert_model_name
        else:
            pass

        # Do not do multiprocessing when using sbert for retrieve then read as this
        # will lead to errors
        if config.retrieve_then_read_model in ['sbert', 'contriever']:
            logger.info(f'Setting dataloader_num_workers to 0 for retrieve then read with {config.retrieve_then_read_model}')
            config.dataloader_num_workers = 0

def _set_commit_hash(
        config: DictConfig | BaseConfig
):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    config.commit_hash = sha