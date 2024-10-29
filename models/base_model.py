from __future__ import annotations

import abc
import json
import os
from typing import ClassVar, Optional, List, Any
import logging

import omegaconf.errors
import torch
import transformers
import yaml
from langchain.prompts import PromptTemplate
from peft import TaskType, LoraConfig, get_peft_model, PeftModel
from torch.nn import Module
try:
    from vllm import LLM, SamplingParams
except ImportError:
    pass

from config_lib.base_config import BaseConfig
from evaluation.common import Statistics, CustomDataset, BaseInstance, BasePrediction
from evaluation.util import parse_answer, make_format_explanation
from models.retrieve import BaseRetriever
from structformer.example_formatting import get_examples_and_prompt_template_and_format
from structformer.format_enforcement import get_prefix_allowed_tokens_fn
from structformer.input_preparation import prepare_input

logger = logging.getLogger()

SPECIAL_HF_CLASSES = {
    'longt5': transformers.LongT5ForConditionalGeneration,
    'flan-t5-xxl': transformers.T5ForConditionalGeneration,
    'flan-ul2': transformers.T5ForConditionalGeneration,
    'flan-t5-xl': transformers.T5ForConditionalGeneration
}


class BaseModel(Module, abc.ABC):
    """
    Base class for models.

    :param model_class_name: Standardized name of the model class, see models/
        for available classes and their names.
    :param config: Configuration object
    :param stats: Statistics object
    :param train_dataset: Dataset object to get examples for in context learning.
    """

    model_class_name: ClassVar[str] = "BaseModel"

    hydra_config: BaseConfig
    stats: "Statistics"
    is_lora_model: bool
    train_dataset: CustomDataset

    def __init__(
            self,
            config: Optional[BaseConfig],
            stats: Optional["Statistics"],
            train_dataset: CustomDataset
    ) -> None:
        super(BaseModel, self).__init__()
        assert config.model.model_class == self.model_class_name
        self.hydra_config = config
        self.stats = stats
        self.is_lora_model = False # set to True in instance when needed
        self.train_dataset = train_dataset
        self.model_name = config.model.model_name
        self.task_has_statement = config.task.has_statement
        self.extraction_mode = config.extraction_mode

        # Set torch dtype
        if self.hydra_config.precision == 32:
            dtype = torch.float32
        elif self.hydra_config.precision in ['bf16', 'bf16-true', 'bf16-mixed']:
            dtype = torch.bfloat16
        elif self.hydra_config.precision == 16:
            dtype = torch.float16
        else:
            raise NotImplementedError
        self.torch_dtype = dtype

        if not self.model_name == 'oracle':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.hydra_config.model.hf_model_id,
                trust_remote_code=True
            )

        if self.model_class_name in ['seq2seq_lm', 'causal_lm']:
            self.model = self.load_model()

        if not self.model_name == 'oracle':
            (
                self.prompt_template,
                self.dummy_prompt_length,
                self.task_input_template,
                self.task_input_variables,
                self.example,
                self.system_prompt,
                self.task_explanation,
                self.format_explanation
            ) = self.set_prompt_elements()

        if self.hydra_config.do_retrieve_then_read:
            self.retrieve_then_read_model = BaseRetriever.load_model(
                'retrieve_and_reduce',
                self.hydra_config.retrieve_then_read_model,
                self.hydra_config.task.answer_has_multiple_statements,
                self.hydra_config.retrieve_then_read_k,
                self.hydra_config.retrieve_then_read_threshold,
                classes=self.hydra_config.task.classes,
                instances=self.train_dataset,
                sbert_model_name=self.hydra_config.retrieve_then_read_sbert_model_name
            )

    @abc.abstractmethod
    def training_collate_fn(self, instances: List[BaseInstance]) -> Any:
        """
        Collate function that creates the training/evaluation batches.
        Implemented by each individual model wrapper.
        """
        raise NotImplementedError

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
            document_lengths,  # The number of characters in the input documents
            node_id_to_node_mappings  # Maps node id strings to nodes
        ) = self._prepare_input(
            instances,
            is_training=False
        )

        tokenized_input = None
        if not self.hydra_config.use_vllm:
            # Tokenize prompt
            tokenized_input = self.tokenizer(
                input_texts,
                return_tensors="pt",
                truncation="longest_first",
                max_length=self.hydra_config.model.max_input_length,
            )
            print('Input length:', tokenized_input['input_ids'].shape[1])

        return {
            "input_texts": input_texts,
            "instances": instances,
            "node_spans": node_spans,
            "tokenized_input": tokenized_input,
            "node_id_to_node_mappings": node_id_to_node_mappings
        }

    def forward(self, input_ids, labels, attention_mask):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return output

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def generate_hf(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> List[str]:
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        input_length = input_ids.shape[1]
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=self.hydra_config.task.max_new_tokens,
            return_dict_in_generate=True,
            prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
            use_cache=True
        )

        generated_texts = []
        for sequence in output.sequences:
            # Decode
            if self.model_class_name == 'causal_lm':
                sequence = sequence[input_length:]
            generated_text = self.tokenizer.decode(
                sequence,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            generated_texts.append(generated_text)

        return generated_texts

    def generate_vllm(
            self,
            input_texts
    ):
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self.hydra_config.task.max_new_tokens,
            seed=self.hydra_config.random_seed
        )
        outputs = self.model.generate(input_texts, sampling_params)
        generated_texts = [
            output.outputs[0].text
            for output in outputs
        ]
        return generated_texts

    def validation_step(self, batch, batch_idx) -> List[BasePrediction]:

        if self.hydra_config.use_vllm:
             generated_texts = self.generate_vllm(batch['input_texts'])
        else:
            generated_texts = self.generate_hf(
                batch['tokenized_input']['input_ids'],
                batch['tokenized_input']['attention_mask']
            )

        free_text_answers = []
        extraction_node_lists = []

        for (
                generated_text,
                instance,
                node_id_to_node_mapping
        ) in zip(
            generated_texts,
            batch['instances'],
            batch['node_id_to_node_mappings']
        ):
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
                node_id_to_node_mapping=node_id_to_node_mapping,
                classes=self.hydra_config.task.classes,
                return_all_extraction_candidates=self.hydra_config.return_all_extraction_candidates
            )
            free_text_answers.append(free_text_answer)
            extraction_node_lists.append(extraction_nodes)

        predictions = []
        for (
                generated_text,
                free_text_answer,
                extraction_nodes,
                instance
        ) in zip(
            generated_texts,
            free_text_answers,
            extraction_node_lists,
            batch['instances']
        ):
            prediction = BasePrediction(
                task_name=instance.task_name,
                example_id=instance.example_id,
                free_text_answer=free_text_answer,
                extraction_nodes=extraction_nodes,
                raw_generation=generated_text
            )
            predictions.append(prediction)

        return predictions

    def load_model(self) -> transformers.PreTrainedModel:
        if self.model_name in SPECIAL_HF_CLASSES:
            if self.hydra_config.use_vllm:
                raise NotImplementedError
            hf_model_class = SPECIAL_HF_CLASSES[self.model_name]
        else:
            hf_model_class = transformers.AutoModelForCausalLM

        if self.hydra_config.use_vllm and any([
            self.hydra_config.load_model,
            self.hydra_config.do_train
        ]):
            raise NotImplementedError

        checkpoint_is_lora_model = False
        if self.hydra_config.load_model:
            # Load model from checkpoint if checkpoint is not just lora adapters
            # Placement changed, so we implement two ways to find the model
            model_path = self.hydra_config.location.results / self.hydra_config.hash_to_load / 'model'
            if not os.path.exists(model_path):
                # Legacy
                model_path = self.hydra_config.location.models / self.hydra_config.hash_to_load

            if self.hydra_config.checkpoint_step_idx is not None:
                checkpoint_path = model_path / f'checkpoint-{self.hydra_config.checkpoint_step_idx}'
            else:
                checkpoint_path = model_path
            with open(
                    model_path / 'custom_config.json'
            ) as f:
                checkpoint_config = json.load(f)
            checkpoint_is_lora_model = checkpoint_config['model']['use_lora']
            if not checkpoint_is_lora_model:
                logger.info('Loading model from checkpoint')
                model = hf_model_class.from_pretrained(
                    checkpoint_path
                )
                return model

        if self.hydra_config.model.load_from_shared:
            model_path = self.hydra_config.model.shared_checkpoint_path
        else:
            model_path = self.hydra_config.model.hf_model_id

        if self.hydra_config.use_vllm:
            model = LLM(model_path)
        else:
            try:
                # Try to use flash attention
                model = hf_model_class.from_pretrained(
                    model_path,
                    use_cache=self.hydra_config.use_cache,
                    attn_implementation='flash_attention_2',
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True
                )
            except (TypeError, ValueError, ImportError):
                logger.warning(
                    f'Loading model with "use_flash_attention" set to True failed,'
                    f' loading again with default value.'
                )
                model = hf_model_class.from_pretrained(
                    model_path,
                    use_cache=self.hydra_config.use_cache,
                    trust_remote_code=True,
                    torch_dtype=self.torch_dtype
                )

            if self.hydra_config.load_model and checkpoint_is_lora_model:
                model = self.load_lora_model(
                    model,
                    checkpoint_step_idx=self.hydra_config.checkpoint_step_idx
                )

            # Convert model to LoRA model if needed
            # Only if training and not continuing from an existing model
            if (
                    self.hydra_config.model.use_lora
                    and not self.hydra_config.load_model
                    and self.hydra_config.do_train
            ):
                model = self._convert_to_lora_model(model)
            else:
                self.hydra_config.model.use_lora = False

        return model

    def _convert_to_lora_model(self, model):
        """Add LoRA adapters to the model."""
        logger.info('Converting to LoRA model')
        config = self.hydra_config

        if config.model.model_class == 'causal_lm':
            task_type = TaskType.CAUSAL_LM
        else:
            task_type = TaskType.SEQ_2_SEQ_LM

        self.is_lora_model = True
        lora_target_modules = config.model.lora_target_modules
        if lora_target_modules is not None:
            lora_target_modules = list(lora_target_modules)
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=lora_target_modules
        )
        # This is needed for gradient checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model = get_peft_model(
            model,
            lora_config
        )
        return model

    def load_lora_model(
            self,
            model,
            checkpoint_step_idx: int = None
    ):
        logger.info('Loading LoRA adapters from checkpoint')
        if checkpoint_step_idx is not None:
            checkpoint_path = self.hydra_config.location.results / self.hydra_config.hash_to_load / 'model' / f'checkpoint-{checkpoint_step_idx}'
        else:
            checkpoint_path = self.hydra_config.location.results / self.hydra_config.hash_to_load / 'model'

        # This is needed for gradient checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
            is_trainable=True
        )
        self.hydra_config.model.use_lora = True
        self.is_lora_model = True
        return model

    def set_prompt_elements(
            self
    ):
        """
        Load the prompts file and set the prompt elements according to the
        indices specified in the config. Store the strings of the prompt
        elements in the config for later reference.
        :return:
        """
        config = self.hydra_config
        with open(config.location.prompts) as f:
            prompts = yaml.safe_load(f)

        task_name = config.task.task_name
        extraction_mode = config.extraction_mode

        # Task explanation
        if config.task_explanation_idx == 'auto':
            config.task_explanation = \
                prompts['task_explanation'][task_name][config.required_aspects]
        elif config.task_explanation_idx == 'empty':
            config.task_explanation = ''
        else:
            config.task_explanation = \
                prompts['task_explanation'][task_name][config.task_explanation_idx]
        task_explanation = config.task_explanation

        # Format explanation
        if config.task.answer_has_multiple_statements:
            n_statements = 'multi'
        else:
            n_statements = 'single'
        if config.format_explanation_idx == 'empty':
            complete_format_explanation = ''
        else:
            if config.format_explanation_idx == 'auto':
                if config.answer_format == 'text' and config.required_aspects == 'answer_only':
                    # No format explanation needed when only requiring
                    # answer in text format
                    format_explanation_str = None
                else:
                    format_explanation_str = \
                        prompts['format_explanation'][extraction_mode][n_statements][config.answer_format]
            else:
                format_explanation_str = \
                    prompts['format_explanation'][extraction_mode][n_statements][config.format_explanation_idx]
            if format_explanation_str is None:
                complete_format_explanation = ''
            else:
                complete_format_explanation = make_format_explanation(
                    config.extraction_mode,
                    config.answer_format,
                    config.required_aspects,
                    format_explanation_str,
                    config.task.answer_has_multiple_statements,
                    config.node_id_template,
                    config.format_explanation_n_dummy_segments
                )
        # HACK: Replace curly braces in format expolanation as they will lead to
        # errors when using as prompt template
        # This is reset in input_preparation.prepare_input()
        complete_format_explanation = complete_format_explanation.replace('{', '{{')
        complete_format_explanation = complete_format_explanation.replace('}', '}}')
        config.format_explanation = complete_format_explanation
        format_explanation = complete_format_explanation

        # System prompt
        system_prompt = prompts['system_prompt'][config.system_prompt_idx]
        if config.model.system_message_template is not None:
            # If there is a special template for the system message, use it
            candidate_input_vars = {
                'system_prompt': system_prompt,
                'task_explanation': task_explanation,
                'format_explanation': format_explanation
            }
            input_vars = {
                k: v
                for k, v in candidate_input_vars.items()
                if k in config.model.system_message_template
            }
            system_message_template = PromptTemplate(
                template=config.model.system_message_template,
                input_variables=list(input_vars.keys())
            )
            system_message = system_message_template.format(**input_vars)
            config.system_prompt = system_message
        else:
            config.system_prompt = system_prompt
        system_prompt = config.system_prompt

        # examples
        example_str = ''
        if config.n_examples > 0:
            # Get the ids of the instances to use as examples
            example_ids = config.task.example_ids[:config.n_examples]
            # Format examples
            example = get_examples_and_prompt_template_and_format(
                example_ids,
                self.train_dataset,
                config
            )
            # Example is list of messages for ChatGPT etc
            if isinstance(example, list):
                # Make string out of example to compute example length later
                # and to document in config
                example_str = json.dumps(example, indent=2)
            else:
                # Replace single curly braces as they will lead to errors
                example = example.replace('{', '{{')
                example = example.replace('}', '}}')
                example_str = example
            try:
                config.example = example_str
            except omegaconf.errors.GrammarParseError:
                config.example = 'GrammarParseError'
        else:
            example = ''
            config.example = ''
        example = example

        # Initialize output format enforcer
        if config.format_explanation_n_dummy_segments > 1:
            single_segment_only = False
        else:
            single_segment_only = True
        if self.model_class_name in ['seq2seq_lm', 'causal_lm']:
            self.prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
                self.tokenizer,
                config.required_aspects,
                config.answer_format,
                single_segment_only
            )

        # Get length of prompt without question, statement and document in tokens
        prompt_variables = {
            'system_prompt': system_prompt,
            'task_explanation': task_explanation,
            'format_explanation': format_explanation,
            'example': example,
            'task_input': '{task_input}'
        }

        if self.hydra_config.do_train or self.hydra_config.load_model:
            prompt_template_str = self.hydra_config.model.prompt_template_training
        elif self.hydra_config.n_examples > 0:
            prompt_template_str = self.hydra_config.model.prompt_template_with_example
        else:
            prompt_template_str = self.hydra_config.model.prompt_template_without_example

        # Remove unneeded prompt variables
        prompt_variables = {
            k: v for k, v in prompt_variables.items()
            if k in prompt_template_str
        }

        tmp_prompt_template = PromptTemplate(
            template=prompt_template_str,
            input_variables=list(prompt_variables.keys())
        )

        # Put prompt elements into single string ("{task_input}" remains as variable)
        prompt_with_task_input_missing = tmp_prompt_template.format(**prompt_variables)

        # Get length of prompt without task input
        dummy_prompt_tokenized = self.tokenizer([prompt_with_task_input_missing], return_tensors='pt')
        dummy_prompt_length = dummy_prompt_tokenized['input_ids'].shape[1]
        if self.model_class_name == 'api_lm':
            # Add example length and system prompt length
            additional_text = system_prompt + example_str
            additional_text_tokenized = self.tokenizer([additional_text], return_tensors='pt')
            dummy_prompt_length += additional_text_tokenized['input_ids'].shape[1]

        # Get prompt template
        prompt_template = PromptTemplate(
            template=prompt_with_task_input_missing,
            input_variables=['task_input']
        )

        # Task specific template
        task_input_variables = list(self.hydra_config.task_input_variables)
        task_input_template_str = self.hydra_config.task.task_input_template
        task_input_variables = [
            var for var in task_input_variables
            if var in task_input_template_str
        ]
        task_input_template = PromptTemplate(
            template=task_input_template_str,
            input_variables=task_input_variables
        )

        return (
            prompt_template,
            dummy_prompt_length,
            task_input_template,
            task_input_variables,
            example,
            system_prompt,
            task_explanation,
            format_explanation
        )

    def _prepare_input(
            self,
            instances: List[BaseInstance],
            is_training: bool
    ):
        """Prepare input instances for data collation"""
        return prepare_input(
            instances=instances,
            tokenizer=self.tokenizer,
            config=self.hydra_config,
            is_training=is_training,
            model_class_name=self.model_class_name,
            prompt_template=self.prompt_template,
            task_input_template=self.task_input_template,
            dummy_prompt_length=self.dummy_prompt_length
        )
