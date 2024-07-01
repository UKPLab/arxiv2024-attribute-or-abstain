import math
from typing import List

import torch
import transformers
from transformers import BatchEncoding


def get_offset(
        full_text: str,
        span_to_find: str,
):
    """

    :param full_text:
    :param span_to_find:
    :return:
    """
    if full_text == span_to_find:
        return 0
    else:
        return full_text.find(span_to_find)


def shorten_text(
        text: str,
        batch_encoding: BatchEncoding,
        max_doc_length: int,
        tokenizer: transformers.PreTrainedTokenizer,

) -> str:
    """
    Shorten a text such that it results in max_doc_length tokens
    :return: The shortened text
    """
    # Define max deviation from target doc length in tokens
    max_deviation = 50
    if max_doc_length < 0:
        raise ValueError(f"max_doc_length is < 0: {max_doc_length}")
    # Shorten text to result in maximum allowed tokens for document
    try:
        final_char_idx = batch_encoding.token_to_chars(0, max_doc_length)[0]
        text = text[:final_char_idx]
    except Exception:
        # Some tokenizers (e.g. the ChatGLM3 tokenizer) do not have the
        # token_to_chars() function
        tokens_after_max_doc_length = batch_encoding['input_ids'][0][max_doc_length:]
        text_after_max_doc_length = tokenizer.decode(tokens_after_max_doc_length)
        final_char = get_offset(text.lower(), text_after_max_doc_length.lower())
        if final_char == -1:
            # If we can't find the offset, shorten coarsely
            full_doc_length = batch_encoding['input_ids'].shape[1]
            final_char = math.floor(len(text) * (max_doc_length / full_doc_length))
            # Because of more coarse method, redefine max deviation
            max_deviation = 100
        text = text[:final_char]
        tokenized_shortened_text = tokenizer(
            [text],
            return_tensors='pt'
        )['input_ids']
        assert (
                (tokenized_shortened_text.shape[1] > max_doc_length - max_deviation)
                and (tokenized_shortened_text.shape[1] < max_doc_length + max_deviation)
        ), 'Text shortening gave an unexpected result'

    return text


def truncate_batch_encoding(
        old_batch_encoding: BatchEncoding,
        new_length: int
) -> BatchEncoding:
    new_data = {}
    for key in old_batch_encoding.data:
        old_tensor = old_batch_encoding.data[key]
        if len(old_tensor.shape) == 1:
            new_tensor = old_tensor[:new_length]
        else:
            new_tensor = old_tensor[:, :new_length]
        new_data[key] = new_tensor

    new_encodings = []
    for old_encoding in old_batch_encoding.encodings:
        old_encoding.truncate(new_length)
        new_encodings.append(old_encoding)

    new_batch_encoding = BatchEncoding(
        data=new_data,
        encoding=new_encodings
    )

    return new_batch_encoding


def merge_batch_encoding_data(
        batch_encodings: List[BatchEncoding]
) -> BatchEncoding:
    # This assumes that all batch encodings have the same batch size
    if len(batch_encodings[0].data['input_ids'].shape) == 1:
        new_data = {}
        new_seq_len = sum(
            [
                batch_encoding.data['input_ids'].shape[0]
                for batch_encoding in batch_encodings
            ]
        )
        for key in batch_encodings[0].data:
            dtype = batch_encodings[0].data[key].dtype
            new_tensor = torch.zeros((new_seq_len), dtype=dtype)
            offset = 0
            for i in range(len(batch_encodings)):
                tensor = batch_encodings[i].data[key]
                tensor_len = tensor.shape[0]
                new_tensor[offset: offset + tensor_len] = tensor
                offset += tensor_len
            new_data[key] = new_tensor
    else:
        new_data = {}
        batch_size = batch_encodings[0].data['input_ids'].shape[0]
        new_seq_len = sum(
            [
                batch_encoding.data['input_ids'].shape[1]
                for batch_encoding in batch_encodings
            ]
        )
        for key in batch_encodings[0].data:
            dtype = batch_encodings[0].data[key].dtype
            new_tensor = torch.zeros((batch_size, new_seq_len), dtype=dtype)
            offset = 0
            for i in range(len(batch_encodings)):
                tensor = batch_encodings[i].data[key]
                tensor_len = tensor.shape[1]
                new_tensor[:, offset: offset + tensor_len] = tensor
                offset += tensor_len
            new_data[key] = new_tensor

    return BatchEncoding(data=new_data)
