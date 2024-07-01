import logging
from typing import Any, ClassVar, List

import openai
import tenacity
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from config_lib.base_config import BaseConfig
from evaluation.common import (
    Statistics,
    BaseInstance,
    BasePrediction, CustomDataset,
)
from models.base_model import BaseModel
from evaluation.util import (
    parse_answer
)
logger = logging.getLogger(__name__)


class APILMForExtractionModel(BaseModel):
    """General class for API served LMs"""
    model_class_name: ClassVar[str] = "api_lm"

    def __init__(
            self,
            config: BaseConfig,
            stats: Statistics,
            train_dataset: CustomDataset
    ):
        super(APILMForExtractionModel, self).__init__(config, stats, train_dataset)

        # Initialize OpenAI API
        self.client = openai.AzureOpenAI(
            azure_endpoint=self.hydra_config.model.azure_openai_endpoint,
            api_key=self.hydra_config.model.api_key,
            api_version=self.hydra_config.openai_api_version
        )
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def training_step(self, batch, batch_idx):
        raise NotImplementedError(
            'Training is not possible with API-based LMs'
        )

    def generate(
            self,
            input_text: str,
            **kwargs
    ) -> str:
        if self.hydra_config.model.model_name in [
            'gpt-35-turbo-0301',
            'gpt-35-turbo-0613',
            'gpt-4-turbo-128k'
        ]:
            generated_text = self.openai_chat_complete(
                input_text
            )

        else:
            raise NotImplementedError(
                f'Did not find suitable generation function for model {self.model_name}'
            )

        return generated_text

    def openai_chat_complete(
            self,
            input_text: str,
    ) -> str:
        """Compile messages and make call to OpenAI API"""
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        if self.example:
            # Examples already come in message format
            messages += self.example
        # The actual instance is added here
        messages.append({
            'role': 'user', 'content': input_text
        })
        if self.hydra_config.answer_format == 'json':
            response_format = { "type": "json_object" }
        else:
            response_format = None
        try:
            response = self._openai_completion_with_backoff(
                model=self.hydra_config.model.deployment_name,
                response_format=response_format,
                seed=self.hydra_config.random_seed,
                max_tokens=self.hydra_config.task.max_new_tokens,
                temperature=0,
                messages=messages
            )
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
            return response.choices[0].message.content
        except tenacity.RetryError:
            return ''

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _openai_completion_with_backoff(
            self, **kwargs
    ):
        return self.client.chat.completions.create(**kwargs)

    def validation_step(self, batch, batch_idx):

        # Generate free text answer
        generated_sequences = [
            self.generate(seq) for seq in batch['input_texts']
        ]

        predictions = []
        for batch_ix, (sequence, instance) in enumerate(zip(
                generated_sequences,
                batch["instances"]
        )):
            # Get the free text answer
            generated_text = sequence

            if generated_text is None:
                generated_text = ''

            # extraction
            # Get extraction nodes from free text answer
            free_text_answer, extraction_nodes = parse_answer(
                generated_text,
                instance.extraction_candidates,
                self.extraction_mode,
                self.hydra_config.answer_format,
                self.hydra_config.required_aspects,
                instance.task_name,
                self.hydra_config.unanswerable_keywords,
                answer_has_multiple_statements=self.hydra_config.task.answer_has_multiple_statements,
                node_id_to_node_mapping=batch["node_id_to_node_mappings"][batch_ix],
                classes=self.hydra_config.task.classes,
                return_all_extraction_candidates=self.hydra_config.return_all_extraction_candidates
            )

            predictions.append(BasePrediction(
                task_name=instance.task_name,
                example_id=instance.example_id,
                free_text_answer=free_text_answer,
                extraction_nodes=extraction_nodes,
                raw_generation=generated_text
            ))
        return predictions

    def validation_collate_fn(self, instances: List[BaseInstance]) -> Any:
        if self.hydra_config.do_retrieve_then_read:
            instances = [
                self.retrieve_then_read_model.retrieve_and_reduce(instance)
                for instance in instances
            ]

        (
            node_spans,  # node_spans map ITG nodes to character sequences in the input text
            offsets,  # The first character of the document in the input
            input_texts,  # The complete prompt
            answer_texts,  # The gold answer texts
            document_lengths,  # Character length of input document
            node_id_to_node_mappings  # Maps node id strings to nodes
        ) = self._prepare_input(
            instances,
            is_training=False
        )

        return {
            "input_texts": input_texts,
            "instances": instances,
            "node_spans": node_spans,
            "node_id_to_node_mappings": node_id_to_node_mappings
        }

    def training_collate_fn(self, instances: List[BaseInstance]) -> Any:
        raise NotImplementedError
