import re
from typing import List

import torch
from torch.nn import Module
from transformers import AutoTokenizer, BatchEncoding, T5ForConditionalGeneration, PreTrainedTokenizer, PreTrainedModel, \
    AutoModelForSeq2SeqLM

from attribution_eval.util import AttributionInstance

class AttributionBaseModel(Module):
    model_name: str = 'base'
    def __init__(
            self,
            predict_max_in_batch: bool = False
    ):
        super(AttributionBaseModel, self).__init__()
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        self.prompt_template = None
        self.predict_max_in_batch = predict_max_in_batch

    @staticmethod
    def load_model(
            model_name: str,
            predict_max_in_batch: bool = False,
            predict_binary: bool = True
    ):
        if model_name == 'true_nli':
            return TRUEModel(
                predict_max_in_batch,
                predict_binary=predict_binary
            )
        elif model_name == 'attrscore':
            return AttrScoreModel(
                predict_max_in_batch,
                predict_binary=predict_binary
            )
        elif model_name == 'minicheck':
            return MiniCheckModel(
                predict_max_in_batch,
                predict_binary=predict_binary
            )
        elif model_name == 'oracle':
            return OracleAttributionModel(predict_max_in_batch)
        else:
            raise NotImplementedError

    def _load_model(self) -> PreTrainedModel:
        raise NotImplementedError

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        raise NotImplementedError

    def collate_fn(
            self,
            batch: List[AttributionInstance]
    ) -> BatchEncoding:
        texts = []
        for instance in batch:
            formatted_text = self.prompt_template.format(
                claim=instance.claim,
                evidence=instance.evidence
            )
            texts.append(formatted_text)

        tokenized_input = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation='longest_first',
            max_length=1000
        )

        return tokenized_input

    def predict(
            self,
            inputs: BatchEncoding
    ) -> List[int]:
        raise NotImplementedError


class OracleAttributionModel(AttributionBaseModel):
    model_name: str = 'oracle'

    def _load_model(
            self
    ):
        return None

    def _load_tokenizer(self):
        return None

    def collate_fn(
            self,
            batch: List[AttributionInstance]
    ) -> List[AttributionInstance]:
        return batch

    def predict(
            self,
            inputs: List[AttributionInstance]
    ) -> List[int]:
        labels = [instance.label for instance in inputs]

        if self.predict_max_in_batch:
            labels = [max(labels)]

        return labels


class AttrScoreModel(AttributionBaseModel):
    model_name: str = 'attrscore'
    def __init__(
            self,
            predict_max_in_batch: bool = False,
            predict_binary: bool = True
    ):
        super().__init__(
            predict_max_in_batch=predict_max_in_batch
        )

        self.predict_binary = predict_binary

        # "attribution-no-definition"
        # from https://github.com/OSU-NLP-Group/AttrScore/blob/main/prompt_and_demo/task_prompts.json
        label_map = {
            "Attributable": "Attributable",
            "Extrapolatory": "Extrapolatory",
            "Contradictory": "Contradictory"
        }
        self.label_regex = r"|".join(list(label_map.keys()))
        self.attributable_label = label_map["Attributable"]

        # Pre-compute the token ids of the labels
        self.label_ids = {
            label_name: self.tokenizer(
                [label_map[label_name]],
                return_tensors='pt',
                add_special_tokens=False
            )['input_ids']
            for label_name in label_map
        }

        self.prompt_template = (
            "Below is an instruction that describes a task, paired with an input"
            " that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            "### Instruction:\n"
            "Verify whether a given reference can support the claim. Options: "
            "Attributable, Extrapolatory or Contradictory.\n\n"
            "### Input:\n"
            "Claim: {claim}\n\n"
            "Reference: {evidence}\n\n\n"
            "### Response:"
        )

    def _load_model(self) -> PreTrainedModel:
        hf_model_id = 'osunlp/attrscore-flan-t5-xxl'
        # load model
        model = T5ForConditionalGeneration.from_pretrained(
            hf_model_id,
            torch_dtype=torch.bfloat16
        )
        return model

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        hf_model_id = 'osunlp/attrscore-flan-t5-xxl'
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_id
        )
        return tokenizer

    def predict(
            self,
            inputs: BatchEncoding
    ) -> List[int]:
        """
                Code partly taken from https://github.com/OSU-NLP-Group/AttrScore/blob/main/inference_alpaca.py
                Predict attribution for each input sequence. Returns 1 or 0 for each
                input_sequence if self.predict_max is False. If self.predict_max is True,
                Returns a single score as the maximum of all predicted scores.
                :param batch: tokenized input
                :return: A list of inferences 1 (entailed) or 0 (not entailed)
                """
        for label_name in self.label_ids:
            self.label_ids[label_name] = self.label_ids[label_name].to(self.model.device)
        result = []
        input_ids = inputs['input_ids']
        input_ids = input_ids.to(self.model.device)
        attention_mask = inputs['attention_mask']
        attention_mask = attention_mask.to(self.model.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # generation_config=generation_config,
            max_new_tokens=3,
            return_dict_in_generate=True,
            output_scores=True
        )
        input_length = 1 if self.model.config.is_encoder_decoder else inputs.input_ids.shape[1]
        relevant_probs = outputs.scores[0].softmax(-1)
        for i, (sequence, probs) in enumerate(
                zip(outputs.sequences, relevant_probs)
        ):
            if self.predict_binary:
                generated_tokens = sequence[input_length:]
                prediction = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                pred_label = re.search(self.label_regex, prediction, re.IGNORECASE).group() if re.search(
                    self.label_regex,
                    prediction, re.IGNORECASE
                ) is not None else 'None'
                if pred_label == self.attributable_label:
                    result.append(1)
                else:
                    result.append(0)
            else:
                # For more complex cases see https://discuss.huggingface.co/t/generation-probabilities-how-to-compute-probabilities-of-output-scores-for-gpt2/3175
                label_probs = {}
                for label_name, label_ids in self.label_ids.items():
                    # Use the probability of the first token of the label as the
                    # label probability
                    label_prob = probs[label_ids[0][0]].item()
                    label_probs[label_name] = label_prob

                score = (
                        label_probs['Attributable']
                        / sum(label_probs.values())
                )

                result.append(score)

        if self.predict_max_in_batch:
            result = [max(result)]

        return result


class TRUEModel(AttributionBaseModel):
    model_name: str = 'true_nli'
    def __init__(
            self,
            predict_max_in_batch: bool = False,
            predict_binary: bool = True
    ):
        super().__init__(
            predict_max_in_batch=predict_max_in_batch
        )

        self.predict_binary = predict_binary
        self.prompt_template = 'premise: {evidence} hypothesis: {claim}'

    def _load_model(self) -> PreTrainedModel:
        hf_model_id = 'google/t5_xxl_true_nli_mixture'
        # load model
        model = T5ForConditionalGeneration.from_pretrained(
            hf_model_id,
            torch_dtype=torch.bfloat16
        )
        return model

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        hf_model_id = 'google/t5_xxl_true_nli_mixture'
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_id,
            torch_dtype=torch.bfloat16
        )
        return tokenizer

    def predict(
            self,
            inputs: BatchEncoding
    ) -> List[int]:
        """
        Predict attribution for each input sequence. Returns 1 or 0 for each
        input_sequence if self.predict_max is False. If self.predict_max is True,
        Returns a single score as the maximum of all predicted scores.
        :param batch: tokenized input
        :return: A list of inferences 1 (entailed) or 0 (not entailed)
        """
        result = []
        input_ids = inputs['input_ids']
        input_ids = input_ids.to(self.model.device)
        attention_mask = inputs['attention_mask']
        attention_mask = attention_mask.to(self.model.device)
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            output_scores=True
        )
        relevant_probs = outputs.scores[0]

        for probs in relevant_probs:
            # For more complex cases see https://discuss.huggingface.co/t/generation-probabilities-how-to-compute-probabilities-of-output-scores-for-gpt2/3175
            # 3: not attributable, 209: attributable
            label_logits = probs[[3, 209]]
            label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
            score = label_probs[1].item()
            if self.predict_binary:
                if score > 0.5:
                    score = 1
                else:
                    score = 0
            result.append(score)

        if self.predict_max_in_batch:
            result = [max(result)]

        return result


class MiniCheckModel(AttributionBaseModel):
    model_name: str = 'minicheck'
    def __init__(
            self,
            predict_max_in_batch: bool = False,
            predict_binary: bool = True
    ):
        super().__init__(
            predict_max_in_batch=predict_max_in_batch
        )
        self.predict_binary = predict_binary
        self.prompt_template = "predict: {evidence}</s>{claim}"

    def _load_model(self) -> PreTrainedModel:
        hf_model_id = 'lytang/MiniCheck-Flan-T5-Large'
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_id)
        return model

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        hf_model_id = 'lytang/MiniCheck-Flan-T5-Large'
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        return tokenizer

    def predict(
            self,
            inputs: BatchEncoding
    ) -> List[int]:
        input_ids = inputs['input_ids']
        input_ids = input_ids.to(self.model.device)
        attention_mask = inputs['attention_mask']
        attention_mask = attention_mask.to(self.model.device)
        decoder_input_ids = torch.zeros((inputs['input_ids'].size(0), 1), dtype=torch.long).to(self.model.device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        logits = outputs.logits.squeeze(1)

        # 3 for no support and 209 for support
        label_logits = logits[:, torch.tensor([3, 209])].cpu()
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        if self.predict_binary:
            labels = torch.argmax(label_probs, dim=1).tolist()
        else:
            labels = label_probs[:, 1].tolist()

        if self.predict_max_in_batch:
            labels = [max(labels)]

        return labels
