from typing import List, Callable, Optional
import logging

from pydantic import BaseModel

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class AnswerAndMultipleSegmentsAnswerFormat(BaseModel):
    answer: str
    segments: List[str]


class AnswerOnlyAnswerFormat(BaseModel):
    answer: str


class MultipleSegmentsOnlyAnswerFormat(BaseModel):
    segments: List[str]


def get_prefix_allowed_tokens_fn(
        tokenizer: PreTrainedTokenizer,
        required_aspects: str,
        answer_format: str,
        single_segment_only: bool = False
) -> Optional[Callable]:
    if required_aspects == 'answer_and_segments':
        answer_format_spec = AnswerAndMultipleSegmentsAnswerFormat
    elif required_aspects == 'answer_only':
        answer_format_spec = AnswerOnlyAnswerFormat
    elif required_aspects == 'segments_only':
        answer_format_spec = MultipleSegmentsOnlyAnswerFormat
    else:
        logger.warning(
            f'Did not find suitable answer format for required aspects: {required_aspects}. '
            'Continuing without prefix_allowed_tokens_fn'
        )
        return None

    if answer_format == 'json':
        parser = JsonSchemaParser(answer_format_spec.model_json_schema())
    else:
        logger.warning(
            f'Did not find suitable parser answer format: {answer_format}. '
            f'Continuing without prefix_allowed_tokens_fn.'
        )
        return None

    prefix_function = build_transformers_prefix_allowed_tokens_fn(
        tokenizer,
        parser
    )

    return prefix_function
