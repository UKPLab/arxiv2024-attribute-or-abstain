import random
from typing import List, Dict

from intertext_graph import Node
from langchain.prompts import PromptTemplate

from config_lib.base_config import BaseConfig
from evaluation.util import format_gold_answer
from structformer import input_preparation
from structformer.input_sequence import to_input_sequence
from structformer.sequence_alignment import SpanMapping, Span

MODEL_TYPE_MAPPING = {
    'gpt-35-turbo-0301': 'openai_chat',
    'gpt-35-turbo-0613': 'openai_chat',
    'gpt-4-turbo-128k': 'openai_chat'
}


def get_examples_and_prompt_template_and_format(
        example_ids: List[str] | List[int],
        train_dataset,
        config: BaseConfig,
) -> str | List[Dict[str, str]]:
    """
    Get example instances from the train dataset and format them according to
    the example template (config.model.example_template_without_statement) and
    model specifics (e.g. for ChatGPT, each example instance results in a user
    and an assistant message).
    :param example_ids: IDs of train instances to be used as examples. Can be
        integer indices or strings.
    :param train_dataset: The train dataset to get the examples from.
    :param config: The config object
    :return:
    """
    # Get prompt template for example
    example_template_str = config.model.example_template
    prompt_template = get_example_prompt_template(
        example_template_str
    )

    # Get example instances
    example_instances = []
    for example_id in example_ids:
        if isinstance(example_id, int):
            example_instances.append(train_dataset[example_id])
        else:
            example_instances.append(train_dataset.get_examples_from_ids([example_id]))

    # Extract relevant info from examples (task input and answer)
    example_dicts = [
        convert_instance_to_example(instance, config)
        for instance in example_instances
    ]

    # Check if example needs model-specific formatting
    if config.model.model_name in MODEL_TYPE_MAPPING:
        model_type = MODEL_TYPE_MAPPING[config.model.model_name]
    else:
        model_type = 'standard'

    # Convert string template to LangChain prompt template
    if model_type == 'openai_chat':
        return put_examples_into_messages_for_openai_chat_lm(
            example_dicts,
            prompt_template
        )
    elif model_type == 'standard':
        return put_examples_into_template_and_join(
            example_dicts,
            prompt_template
        )
    else:
        raise NotImplementedError(f'Example formatting not implemented for model type {model_type}')


def get_example_prompt_template(
        example_template_str: str,
) -> PromptTemplate:
    """
    Convert template string into LangChain prompt template
    :param example_template_str: String template for example
    :param task_has_statement: Whether the task has a statement
    :param include_answer_in_template: Whether to include the answer in the
        resulting LangChain prompt template
    :return:
    """
    # All possible input variables for example
    example_input_variables = [
        'task_input',
        'answer'
    ]
    # Remove unneeded input variables
    example_input_variables = [
        var for var in example_input_variables
        if var in example_template_str
    ]
    # Get LangChain prompt template
    prompt_template = PromptTemplate(
        template=example_template_str,
        input_variables=example_input_variables
    )
    return prompt_template


def put_examples_into_template_and_join(
        examples: List[Dict[str, str]],
        template: PromptTemplate
) -> str:
    """
    # Evaluate (fill) prompt template with input variables
    :param examples: Dictionaries with input variables
    :param template: LangChain prompt template to be filled
    :return:
    """
    filled_templates = []
    for example in examples:
        filled_template = template.format(**example)
        filled_templates.append(filled_template)
    return '\n'.join(filled_templates)


def put_examples_into_messages_for_openai_chat_lm(
        examples: List[Dict[str, str]],
        template: PromptTemplate
) -> List[Dict[str, str]]:
    """
    Creates two dict objects for each supplied example. First dict is user
    message with input information (context, question etc.), second is the
    assistant message with the answer.
    :param examples: List of examples in dict form. Each dict contains relevant
        variables such as context, question and answer.
    :param template: Langchain prompt template for user message.
    :return:
    """
    messages = []
    for example in examples:
        # Get answer from example dict
        answer = example.pop('answer')
        user_message_str = template.format(**example)
        messages.append({
            'role': 'user', 'content': user_message_str
        })
        messages.append({
            'role': 'assistant', 'content': answer
        })

    return messages


def convert_instance_to_example(
        instance,
        config: BaseConfig
) -> Dict[str, str]:
    """
    Converts an instance to an example in dictionary format. Possible keys are
    'task_input' and 'answer'. These keys correspond to
    the input variables of the example template.
    :param instance:
    :param config:
    :return:
    """
    # Convert ITG to string and span mappings
    doc_text, node_spans = to_input_sequence(instance.document, config)

    # If doing extraction from node ids, get mapping from node ids to nodes
    if (
        (config.required_aspects in ['answer_and_segments', 'segments_only'])
        and (config.extraction_mode == 'node_id')
    ):
        node_id_to_node_mapping = input_preparation.map_node_ids_to_nodes(
            instance.document.nodes,
            config.node_id_template
        )
    else:
        node_id_to_node_mapping = None

    # Shorten example documents
    if config.shorten_example_docs == 'to_extraction_nodes':
        doc_text, node_spans = shorten_doc_to_surrounding_nodes(
            node_spans,
            doc_text,
            config.example_use_n_surrounding_nodes,
            instance.extraction_nodes[0],
            config.is_node_separator,
            config.task.keep_k_random_nodes_when_pruning,
            instance.extraction_candidates,
            config.random_seed
        )

    if config.example_doc_max_n_chars_per_node != -1:
        doc_text = shorten_doc_nodes_to_max_n_chars(
            node_spans,
            doc_text,
            config.example_doc_max_n_chars_per_node,
            config.is_node_separator
        )

    answer = format_gold_answer(
        instance,
        config.extraction_mode,
        config.answer_format,
        config.required_aspects,
        config.task.answer_has_multiple_statements,
        100,
        config.is_include_node_types,
        node_id_to_node_mapping=node_id_to_node_mapping
    )

    # Define all possible task input variables
    task_input_variables = {
        'question': instance.question,
        'statement': instance.statement,
        'context': doc_text,
        'additional_info': instance.additional_info
    }
    # Get template string
    task_input_template_str = config.task.task_input_template
    # Remove unneeded input variables
    task_input_variables = {
        k: v for k, v in task_input_variables.items()
        if k in task_input_template_str
    }
    # Make template for task input
    task_input_template = PromptTemplate(
        template=task_input_template_str,
        input_variables=list(task_input_variables.keys())
    )
    # Format task input
    task_input = task_input_template.format(**task_input_variables)

    return_dict = {
        'task_input': task_input,
        'answer': answer
    }

    return return_dict


def shorten_doc_nodes_to_max_n_chars(
        node_spans: SpanMapping,
        doc_text: str,
        max_n_chars: int,
        node_separator: str
) -> str:
    """To save space in examples, shorten each node to max_n_chars
    characters."""
    shortened_nodes = []
    len_node_separator = len(node_separator)
    max_n_chars = max_n_chars - len_node_separator

    for span in node_spans.spans:
        if span.right - span.left < max_n_chars:
            shortened_nodes.append(doc_text[span.left:span.right - len_node_separator])
        else:
            shortened_nodes.append(doc_text[span.left: span.left + max_n_chars])

    return node_separator.join(shortened_nodes)


def shorten_doc_to_surrounding_nodes(
        node_spans: SpanMapping,
        doc_text: str,
        n_surrounding_nodes: int,
        extraction_nodes: List[Node] | List[List[Node]],
        node_separator: str,
        k_random_nodes: int,
        extraction_candidates: List[Node],
        random_seed: int
):
    """Keep only those nodes in span mapping that are either
    the article title, the section title, an extraction node
    or in the n_surrounding_nodes vicinity of an extraction node."""

    if len(extraction_nodes) > 0:
        if isinstance(extraction_nodes[0], list):
            extraction_nodes = [
                n for nodes_list in extraction_nodes for n in nodes_list
            ]
    if len(extraction_nodes) == 0 and k_random_nodes > 0:
        # If there are no extraction nodes, sample randomly
        random.seed(random_seed)
        extraction_nodes = random.sample(
            extraction_candidates,
            k_random_nodes
        )

    # Get positions of extraction nodes
    extraction_node_positions = []
    for i, span in enumerate(node_spans.spans):
        if span.content in extraction_nodes:
            extraction_node_positions.append(i)

    surrounding_positions = []
    if n_surrounding_nodes > 0:
        for i in range(n_surrounding_nodes):
            dist = i + 1
            surrounding_positions.extend([
                pos + dist for pos in extraction_node_positions
            ])
            surrounding_positions.extend([
                pos - dist for pos in extraction_node_positions
            ])

    positions = extraction_node_positions + surrounding_positions

    filtered_spans = []
    for i, span in enumerate(node_spans.spans):
        if span.content.ntype in ['article-title', 'title']:
            filtered_spans.append(span)
        elif i in positions:
            filtered_spans.append(span)

    new_text = node_separator.join([
        doc_text[span.left: span.right - len(node_separator)]
        for span in filtered_spans
    ])
    new_span_mapping = recombine_spans_to_span_mapping(
        filtered_spans
    )

    return new_text, new_span_mapping


def recombine_spans_to_span_mapping(
        spans: List[Span],
) -> SpanMapping:
    """Take a list of span objects, update their lefts and
    rights."""
    current_pos = 0
    updated_spans = []
    for span in spans:
        left = current_pos
        right = current_pos + (span.right - span.left)
        updated_spans.append(Span(
            left,
            right,
            span.content
        ))
        current_pos = right

    span_mapping = SpanMapping(updated_spans)
    return span_mapping