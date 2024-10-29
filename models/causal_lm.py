import logging
from typing import Any, ClassVar, List

from config_lib.base_config import BaseConfig
from evaluation.common import (
    Statistics,
    BaseInstance,
    CustomDataset,
)
from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class CausalLMForExtractionModel(BaseModel):
    """
    Class that should be able to load any model from huggingface transformers that
    is compatible to the AutoModelForCausalLM class.
    """
    model_class_name: ClassVar[str] = "causal_lm"

    def __init__(
            self,
            config: BaseConfig,
            stats: Statistics,
            train_dataset: CustomDataset
    ):
        # FIXME: Add padding token to tokenizer and model embedding matrix
        #   when using batch size > 1. Currently not using pad tokens because
        #   this will make model saving and loading with LORA more complicated.
        #   Padding tokens were removed after commit 3243be6e9fa651a538462c6263d491795d212f07
        super(CausalLMForExtractionModel, self).__init__(config, stats, train_dataset)


    def training_collate_fn(self, instances: List[BaseInstance]) -> Any:

        (
            node_spans,  # node_spans map ITG nodes to character sequences in the input text
            offsets,  # The first character of the document in the input
            input_texts,  # The complete prompt
            answer_texts,  # The gold answer texts
            document_lengths,  # Character length of input document
            node_id_to_node_mappings  # Maps node id strings to nodes
        ) = self._prepare_input(
            instances,
            is_training=True
        )

        tokenized_input = self.tokenizer(
            input_texts[0] + answer_texts[0] + self.tokenizer.eos_token,
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized_input['input_ids'],
            'labels': tokenized_input['input_ids'],
            'attention_mask': tokenized_input['attention_mask']
        }