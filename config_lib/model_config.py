from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import MISSING, DictConfig
from hydra.core.config_store import ConfigStore

from config_lib.config_container import ConfigContainer


@dataclass
class BaseModelConfig(ConfigContainer):
    """
    :param max_input_length: The maximum length of the input sequence.

    :param pe_init_std: The standard deviation for the initialization of the position embeddings.

    :param extraction_weight: The weight for the extraction loss.
    :param use_extraction_loss_weights: Whether to use extraction loss weights. Can be useful when
    there are much more irrelevant nodes than relevant nodes.

    :param max_steps: The maximum number of steps to train.
    :param min_steps: The minimum number of steps to train.
    :param val_check_interval: The number of instances after which to run validation.
    :param batch_size: The batch size.
    :param accumulate_grad_batches: The number of batches to accumulate gradients over.
    :param learning_rate: The learning rate.

    # OpenAI
    :param deployment_name: The name of the model as deployed at OpenAI
    :param azure_openai_endpoint: URL of the API endpoint
    :param api_key: Secret keyphrase to access the API
    
    # Prompt
    :param prompt_template_with_example: Prompt template when using example
    :param prompt_template_without_example: Prompt template when not using example
    :param prompt_template_training: Prompt template when training
    :param example_template: template for examples

    :param system_message_template: How the system message should be formatted

    :param use_lora: Whether to use LoRA finetuning.
    :param lora_r: r parameter for LoRA.
    :param lora_alpha: alpha parameter for LoRA.
    :param lora_dropout: Dropout in LoRA layers.
    :param lora_target_modules: Which modules should receive lora adapters

    :param oracle_return_random_non_gold_evidence: Whether the oracle should return
        random nodes as evidence that are not annotated as true evidence.
    """

    model_name: str = MISSING
    model_class: str = MISSING
    hf_model_id: Optional[str] = ''
    load_from_shared: Optional[bool] = None
    shared_checkpoint_path: Optional[Path] = None

    # Model parameters
    max_input_length: Optional[int] = None

    d_model_name: Optional[str] = None
    final_layer_idx: Optional[int] = None

    # Structure infusion
    # Position embeddings
    pe_init_std: Optional[float] = None

    # Extraction
    extraction_weight: Optional[float] = None

    use_extraction_loss_weights: Optional[bool] = None

    # Training
    max_steps: Optional[int] = -1
    min_steps: Optional[int] = -1
    val_check_interval: Optional[int] = 0
    batch_size: Optional[int] = None
    accumulate_grad_batches: Optional[int] = None
    learning_rate: Optional[float] = None

    # Prompt (not needed in LongT5)
    prompt_template_with_example: Optional[str] = None
    prompt_template_without_example: Optional[str] = None
    prompt_template_training: Optional[str] = None
    example_template: Optional[str] = None
    system_message_template: Optional[str] = None

    # OpenAI
    deployment_name: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    api_key: Optional[str] = None

    # LORA
    use_lora: Optional[bool] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    lora_target_modules: Optional[List[str]] = None

    # Oracle
    oracle_return_random_non_gold_evidence: Optional[bool] = None

cs = ConfigStore.instance()
cs.store(
    name="base_model_config",
    group='model',
    node=BaseModelConfig
)