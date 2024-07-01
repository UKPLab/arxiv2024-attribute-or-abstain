import logging
from typing import Any, ClassVar, List

from config_lib.base_config import BaseConfig
from evaluation.common import Statistics, BaseInstance, CustomDataset
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class Seq2SeqForExtractionModel(BaseModel):
    """General class for enccoder-decoder models."""
    model_class_name: ClassVar[str] = "seq2seq_lm"

    def __init__(
            self,
            config: BaseConfig,
            stats: Statistics,
            train_dataset: CustomDataset
    ):
        super(Seq2SeqForExtractionModel, self).__init__(config, stats, train_dataset)

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
            text=input_texts,
            text_target=answer_texts,
            return_tensors='pt'
        )

        return tokenized_input
