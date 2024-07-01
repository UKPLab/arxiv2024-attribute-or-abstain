import copy
import json
import random
import re
from typing import Dict, List

import transformers
from intertext_graph import Node, IntertextDocument
from langchain.prompts import PromptTemplate

from config_lib.base_config import BaseConfig
from evaluation.common import BaseInstance
from evaluation.util import format_gold_answer
from structformer.collation_utils import get_offset, shorten_text
from structformer.input_sequence import to_input_sequence
from structformer.sequence_alignment import SpanMapping


def prepare_input(
        instances: List[BaseInstance],
        tokenizer: transformers.PreTrainedTokenizer,
        config: BaseConfig,
        is_training: bool,
        model_class_name: str,
        prompt_template: PromptTemplate,
        task_input_template: PromptTemplate,
        dummy_prompt_length: int
):
    """

    :param instances:
    :param tokenizer:
    :param config:
    :param is_training:
    :param model_class_name:
    :param prompt_template:
    :param task_input_template
    :param dummy_prompt_length: Prompt length without instance-specific inputs

    :return:
    """
    node_spans = []
    offsets = []
    input_texts = []
    answer_texts = []
    document_lengths = []
    node_id_to_node_mappings = []
    for instance in instances:
        if config.do_prune_to_extraction_nodes:
            instance.document = prune_to_extraction_nodes(
                instance.document,
                instance.extraction_nodes,
                config.task.keep_k_random_nodes_when_pruning,
                config.keep_structure_when_pruning
            )

        # Concatenate document to a single string
        text, spans = to_input_sequence(instance.document, config)
        node_spans.append(spans)

        if (
            (
                config.required_aspects
                in ['answer_and_segments', 'segments_only']
            )
            and (config.extraction_mode == 'node_id')
        ):
            node_id_to_node_mapping = map_node_ids_to_nodes(
                instance.document.nodes,
                config.node_id_template
            )
        else:
            node_id_to_node_mapping = None
        node_id_to_node_mappings.append(node_id_to_node_mapping)

        # Shorten document so that the complete input does not exceed the
        # maximum input length
        question_length = tokenizer(
            [instance.question],
            return_tensors='pt',
            add_special_tokens=False
        )['input_ids'].shape[1]
        statement_length = tokenizer(
            [instance.statement],
            return_tensors='pt',
            add_special_tokens=False
        )['input_ids'].shape[1]
        additional_info_length = tokenizer(
            [instance.additional_info],
            return_tensors='pt',
            add_special_tokens=False
        )['input_ids'].shape[0]

        if is_training:
            # Format gold answer according to extraction mode
            answer_text = format_gold_answer(
                instance=instance,
                extraction_mode=config.extraction_mode,
                answer_format=config.answer_format,
                required_aspects=config.required_aspects,
                answer_has_multiple_statements=config.task.answer_has_multiple_statements,
                shorten_text_to_first_n_chars=100,
                include_node_types=config.is_include_node_types,
                node_id_to_node_mapping=node_id_to_node_mapping
            )
            answer_texts.append(answer_text)

        if model_class_name == 'causal_lm' and is_training:
            # Need to incorporate answer length into the complete prompt length
            # when training a causal lm
            answer_length = tokenizer(
                [answer_text],
                return_tensors='pt',
                add_special_tokens=False
            )['input_ids'].shape[1]
            max_doc_length = (
                    config.model.max_input_length
                    - (
                        dummy_prompt_length
                        + question_length
                        + statement_length
                        + additional_info_length
                        + answer_length
                    )
            )
        else:
            if model_class_name in ['causal_lm', 'api_lm']:
                max_doc_length = (
                        config.model.max_input_length
                        - (
                            dummy_prompt_length
                            + question_length
                            + statement_length
                            + additional_info_length
                            + config.task.max_new_tokens
                        )
                )
            else:
                max_doc_length = (
                        config.model.max_input_length
                        - (
                                dummy_prompt_length
                                + question_length
                                + statement_length
                                + additional_info_length
                        )
                )

        document_tokens = tokenizer([text], return_tensors='pt')
        if max_doc_length < document_tokens['input_ids'].shape[1]:
            text = shorten_text(
                text,
                document_tokens,
                max_doc_length,
                tokenizer
            )

        task_input_variables = {
            'question': instance.question,
            'statement': instance.statement,
            'context': text,
            'additional_info': instance.additional_info
        }
        task_input_variables = {
            k: v for k, v in task_input_variables.items()
            if k in task_input_template.input_variables
        }
        task_input = task_input_template.format(**task_input_variables)

        # Get prompt elements
        prompt_variables = {
            'task_input': task_input
        }

        # Put prompt elements into single string
        input_text = prompt_template.format(**prompt_variables)
        # HACK: Re-replace double curly braces
        # See base_model.set_prompt_elements, the part on making format
        # explanation
        input_text = input_text.replace('}}', '}')
        input_text = input_text.replace('{{', '{')
        input_texts.append(input_text)
        # Get character start offset of input document in prompt
        offset = get_offset(input_text, text[:100])
        offsets.append(offset)
        # Get character length of document in prompt
        document_length = len(text)
        document_lengths.append(document_length)

    return (
        node_spans,
        offsets,
        input_texts,
        answer_texts,
        document_lengths,
        node_id_to_node_mappings
    )


def map_node_ids_to_nodes(
        nodes: List[Node],
        node_id_template: str,
) -> Dict[str, Node]:
    """Map id strings (e.g. '<<Segment_X>>' to nodes)"""
    node_id_to_node_mapping = {}
    for node in nodes:
        node_id = node.ix.split('_')[-1]
        node_id_to_node_mapping[eval(f"""f'{node_id_template}'""")] = node

    return node_id_to_node_mapping


def prune_to_extraction_nodes(
        itg: IntertextDocument,
        extraction_nodes: List[Node] | List[List[Node]],
        k_random_nodes: int,
        keep_structure: bool
):
    """
    Remove all nodes that are not annotated as evidence. If there are no
    annotated evidence nodes, keep n_replacement_nodes random nodes.
    :param itg:
    :param extraction_nodes: The annotated extraction nodes
    :param k_random_nodes: If there are no annotated extraction nodes, this
        determines the number of random nodes to keep.
    :param keep_structure: If True, the title and section titles are kept.
    :return:
    """
    def sample_nodes(
            itg_: IntertextDocument,
            k: int
    ) -> List[Node]:
        """Randomly sample k paragraph nodes (ntype == 'p') from itg_."""
        paragraph_nodes = [n for n in itg_.nodes if n.ntype == 'p']
        selected_nodes = random.sample(paragraph_nodes, k)
        return selected_nodes

    # Make hard copy of itg (otherwise will remove nodes from other instances
    # that share the same itg)
    itg = copy.deepcopy(itg)

    # If extraction_nodes is a single list, convert into list of lists
    if len(extraction_nodes) > 0:
        if isinstance(extraction_nodes[0], Node):
            extraction_nodes = [extraction_nodes]

    # Put extraction nodes into single list
    extraction_nodes_flat = [
        n for extraction_node_list in extraction_nodes
        for n in extraction_node_list
    ]

    if len(extraction_nodes_flat) > 0:
        # Because we made a deepcopy of the itg object, we need to find the
        # extraction nodes in the new copy
        extraction_node_ixs = [n.ix for n in extraction_nodes_flat]
        nodes_to_keep = [
            n for n in itg.nodes
            if n.ix in extraction_node_ixs
        ]
    else:
        nodes_to_keep = sample_nodes(itg, k_random_nodes)

    if keep_structure:
        for n in itg.nodes:
            if (
                (n.ntype not in ['article-title', 'title'])
                and (n not in nodes_to_keep)
            ):
                # If node is not a title or section title, remove it
                itg.remove_node(n, preserve_next_edge=True)
    else:
        for n in itg.nodes:
            if (
                    (n.ntype != 'article-title')
                    and (n not in nodes_to_keep)
            ):
                itg.remove_node(n, preserve_next_edge=True)

    return itg