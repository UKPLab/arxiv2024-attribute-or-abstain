import json
import re
from typing import Optional, List, Tuple, Dict
import logging
from decimal import Decimal

import nltk
from intertext_graph import Node

from evaluation.common import BasePrediction

logger = logging.getLogger(__name__)

"""
This is based on the code from the relpos_graph repository
"""

################################################################################
# Creation of format explanation
################################################################################


def make_format_explanation(
        extraction_mode: str,
        answer_format: str,
        required_aspects: str,
        explanation_str: str,
        answer_has_multiple_statements: bool,
        node_id_template: str = None,
        n_example_dummy_segments: int = 1
) -> str:
    """
    Automatically creates a format explanation.
    Example:
        extraction_mode = 'node_id'
        answer_format = 'json',
        required_aspects = 'answer_and_segments',
        explanation_str = 'Your answer must have the following format'.
        node_id_template = '[{node_id}]'
        n_example_dummy_segments = 2
        Return: "Your answer must have the following format
        {
            answer: '<answer>',
            segments: [
                '[X]',
                '[Y]'
            ]
        }
    """
    if answer_has_multiple_statements and answer_format != 'text':
        raise ValueError(
            'Multiple statement answers only possible with answer format "text".'
        )
    if (
        extraction_mode == 'text'
        and answer_format == 'text'
        and required_aspects != 'answer_only'
    ):
        raise ValueError(
            'Cannot combine extraction mode "text" with answer format "text".'
        )

    if answer_has_multiple_statements:
        dummy_free_text_answer = [
            '<answer_sentence_1>',
            '<answer_sentence_2>',
            '...'
        ]
    else:
        dummy_free_text_answer = '<answer>'

    if (
        (required_aspects in ['answer_and_segments', 'segments_only'])
        and (n_example_dummy_segments < 1)
    ):
        raise ValueError(
            f'Trying to produce format explanation with segments but '
            f'n_example_dummy_segments={n_example_dummy_segments}'
        )
    if extraction_mode == 'text':
        if n_example_dummy_segments == 1:
            dummy_segments = [
                '<Relevant sentence>'
            ]
        else:
            dummy_segments = [
                f'<Relevant sentence {i + 1}>' for i in range(n_example_dummy_segments)
            ]
    elif extraction_mode == 'node_id':
        node_ids = ['X', 'Y', 'Z']
        if answer_has_multiple_statements:
            # Hard code for now -> [['[X]'], ['[Y]', '[Z]']]
            node_id = node_ids[0]
            dummy_segments_1 = [eval(f'f"""{node_id_template}"""')]
            dummy_segments_2 = []
            for i in range(1, 3):
                node_id = node_ids[i]
                dummy_segments_2.append(eval(f'f"""{node_id_template}"""'))
            dummy_segments = [dummy_segments_1, dummy_segments_2, []]

        else:
            dummy_segments = []
            for i in range(n_example_dummy_segments):
                node_id = node_ids[i]
                dummy_segments.append(eval(f'f"""{node_id_template}"""'))
    else:
        raise ValueError(f'Unknown extraction mode: {extraction_mode}')

    if answer_format == 'text':
        formatted_dummy_answer = format_gold_answer_text(
            dummy_free_text_answer,
            dummy_segments,
            required_aspects
        )
    elif answer_format == 'structured_text':
        formatted_dummy_answer = format_gold_answer_structured_text(
            dummy_free_text_answer,
            dummy_segments,
            required_aspects
        )
    elif answer_format == 'json':
        formatted_dummy_answer = format_gold_answer_json(
            dummy_free_text_answer,
            dummy_segments,
            required_aspects
        )
    else:
        raise ValueError(f'Unknown answer format: {answer_format}')

    return explanation_str.format(dummy_answer=formatted_dummy_answer)


################################################################################
# Formatting of gold answer
################################################################################

# This is used for training and in the creation of the format explanation

def format_gold_answer(
        instance,
        extraction_mode: str,
        answer_format: str,
        required_aspects: str,
        answer_has_multiple_statements: bool,
        shorten_text_to_first_n_chars: Optional[int] = None,
        include_node_types: List[str] = None,
        node_id_to_node_mapping: Optional[Dict[str, Node]] = None
) -> str:
    """
    Creates a gold answer in the desired format.
    Example:
        instance.free_text_answer = 'The model is better than the baselines.'
        instance.extraction_nodes = [[node_a, node_b]]
        extraction_mode = 'node_id'
        answer_format = 'json',
        required_aspects = 'answer_and_segments',
        node_id_to_node_mapping = {
            '[0]': node_a,
            '[1]': node_b
        }
        Return: "
        {
            answer: 'The model is better than the baselines',
            segments: [
                '[0]',
                '[1]'
            ]
        }
    :param instance:
    :param extraction_mode:
    :param answer_format:
    :param required_aspects:
    :param shorten_text_to_first_n_chars: Only relevant when extraction_mode is
        'text'. Shorten the text of extraction nodes to the first n chats.
    :param include_node_types: If given, only extraction nodes of the given
        types are included in the formatted answer.
    :param node_id_to_node_mapping:
    :return:
    """
    if answer_has_multiple_statements and answer_format != 'text':
        raise ValueError(
            'Multiple statement answers only possible with answer format "text".'
        )
    if (
        extraction_mode == 'text'
        and answer_format == 'text'
        and required_aspects != 'answer_only'
    ):
        raise ValueError(
            'Cannot combine extraction mode "text" with answer format "text".'
        )
    newline = '\n'

    def get_gold_segments(
            extraction_nodes_: List[Node],
            extraction_mode_: str,
            shorten_text_to_first_n_chars_: int = None,
            node_id_to_node_mapping_: Dict[str, Node] = None
    ) -> List[str]:
        """Get textual representation of extraction nodes either as ids or
        verbatim quotes from the node text"""
        if extraction_mode_ == 'text':
            if shorten_text_to_first_n_chars_ is None:
                gold_segments = [
                    n.content for n in extraction_nodes_
                ]
            else:
                gold_segments = [
                    n.content[:shorten_text_to_first_n_chars]
                    for n in extraction_nodes_
                ]

        elif extraction_mode_ == 'node_id':
            if node_id_to_node_mapping_ is not None:
                node_to_node_id_mapping = {
                    node: node_id for node_id, node in node_id_to_node_mapping.items()
                }
                gold_segments = [node_to_node_id_mapping[n] for n in extraction_nodes_]
            else:
                gold_segments = []
        else:
            raise ValueError(f'Unknown extraction mode {extraction_mode}')

        return gold_segments

    if answer_has_multiple_statements:
        gold_segments = []
        for extraction_nodes_list in instance.extraction_nodes[0]:
            # Remove extraction nodes that are not included in the text
            extraction_nodes_list = [
                n for n in extraction_nodes_list
                if n.ntype in include_node_types
            ]
            gold_segments.append(get_gold_segments(
                extraction_nodes_list,
                extraction_mode,
                shorten_text_to_first_n_chars,
                node_id_to_node_mapping
            ))
    else:
        # Remove extraction nodes that are not included in the text
        extraction_nodes = [
            n for n in instance.extraction_nodes[0]
            if n.ntype in include_node_types
        ]
        gold_segments = get_gold_segments(
            extraction_nodes,
            extraction_mode,
            shorten_text_to_first_n_chars,
            node_id_to_node_mapping
        )

    if answer_format == 'text':
        gold_answer = format_gold_answer_text(
            instance.free_text_answer[0],
            gold_segments,
            required_aspects
        )
    elif answer_format == 'structured_text':
        gold_answer = format_gold_answer_structured_text(
            instance.free_text_answer[0],
            gold_segments,
            required_aspects
        )
    elif answer_format == 'json':
        gold_answer = format_gold_answer_json(
            instance.free_text_answer[0],
            gold_segments,
            required_aspects
        )
    else:
        raise ValueError(f'Unknown answer format {answer_format}')

    return gold_answer


def format_gold_answer_json(
        free_text_answer: str,
        gold_segments: List[str],
        required_aspects: str
) -> str:
    return_dict = {
        'answer': free_text_answer,
        'segments': gold_segments
    }
    if required_aspects == 'answer_only':
        return_dict.pop('segments')
    elif required_aspects == 'segments_only':
        return_dict.pop('answer')

    return f'\n{json.dumps(return_dict, indent=None)}'


def format_gold_answer_structured_text(
        free_text_answer: str,  # Might seem unused, but is used in format string
        gold_segments: List[str],
        required_aspects: str
) -> str:
    def escape_curly_braces(s: str):
        return s.replace('{', '{{').replace('}', '}}')

    newline = '\n'

    title = 'Segments'

    gold_segments = [
        escape_curly_braces(item) for item in gold_segments
    ]
    # Find faulty strings
    filtered_gold_segments = []
    for item in gold_segments:
        try:
            _ = f'{item}'
            filtered_gold_segments.append(item)
        except SyntaxError:
            pass

    free_text_answer_template = """{free_text_answer}"""
    segments_template = f"""{title}:\n{newline.join('- ' + item for item in filtered_gold_segments)}"""

    formatted_free_text_answer = eval(f'f"""{free_text_answer_template}"""')

    try:
        formatted_segments = eval(f'f"""{segments_template}"""')
    except SyntaxError:
        # TODO: check if this fix works
        segments_template = f"""{title}:"""
        formatted_segments = eval(f'f"""{segments_template}"""')

    if required_aspects == 'answer_only':
        return formatted_free_text_answer
    elif required_aspects == 'segments_only':
        return formatted_segments
    else:
        return formatted_free_text_answer + '\n' + formatted_segments


def format_gold_answer_text(
        free_text_answer: str | List[str],
        gold_segments: List[str] | List[List[str]],
        required_aspects: str
):
    if type(free_text_answer) == str:
        free_text_answer = [free_text_answer]
        gold_segments = [gold_segments]

    if required_aspects == 'answer_only':
        formatted_answer = ' '.join([sent for sent in free_text_answer])
    elif required_aspects == 'segments_only':
        formatted_answer = ', '.join([
            segment for segment_list in gold_segments for segment in segment_list
        ])
    else:
        formatted_sents = []
        for sent, citations in zip(free_text_answer, gold_segments):
            formatted_citations = ' '.join(citations)
            if len(formatted_citations) == 0:
                formatted_sent = sent
            else:
                sent = sent.strip()
                if sent.endswith('.'):
                    formatted_sent = f'{sent[:-1]} {formatted_citations}.'
                else:
                    formatted_sent = f'{sent} {formatted_citations}'
            formatted_sents.append(formatted_sent)
        formatted_answer = ' '.join(formatted_sents)

    return formatted_answer


################################################################################
# Answer parsing
################################################################################


def parse_answer(
        answer_text: str,
        extraction_candidates: List[Node],
        extraction_mode: str,
        answer_format: str,
        required_aspects: str,
        task_name: str,
        unanswerable_keywords: List[str],
        answer_has_multiple_statements: bool = None,
        node_id_to_node_mapping: Dict[str, Node] = None,
        classes: List[str] = None,
        return_all_extraction_candidates: bool = False
) -> Tuple[str, List[Node]]:
    """
    Parses a model answer according to the given configuration.
    :param answer_text:
    :param extraction_candidates:
    :param extraction_mode:
    :param answer_format:
    :param required_aspects:
    :param unanswerable_keywords: A list of strings that signify "unanswerable"
    :param answer_has_multiple_statements: Whether the answer consists of multiple
        statements (sentences).
    :param node_id_to_node_mapping:
    :return:
    """
    def get_extraction_nodes(
            extraction_node_strs_: List[str],
            extraction_candidates_: List[Node],
            extraction_mode_: str,
            node_id_to_node_mapping_: Dict[str, Node]
    ) -> List[Node]:
        extraction_nodes_ = []
        if extraction_mode_ == 'text':
            # Find extraction nodes by matching to node texts
            for s in extraction_node_strs_:
                if len(s) == 0:
                    continue
                for node in extraction_candidates_:
                    if s in node.content:
                        extraction_nodes_.append(node)
                        break

        elif extraction_mode_ == 'node_id':
            # Find extraction nodes by matching to node id
            for s in extraction_node_strs_:
                if len(s) == 0:
                    continue
                try:
                    s = s.strip()
                except AttributeError:
                    # Parsing resulted in non-str-object, continue
                    continue
                for k in node_id_to_node_mapping_:
                    if s in k:
                        # Sometimes the models generate only part of the original
                        # ID, e.g. "Segment-1" instead of "<<Segment-1>>
                        extraction_nodes_.append(node_id_to_node_mapping_[k])

        # de-duplicate extraction nodes
        extraction_nodes_ = list(set(extraction_nodes_))
        return extraction_nodes_

    if answer_has_multiple_statements and answer_format != 'text':
        raise ValueError(
            'Multiple statement answers only possible with answer format "text".'
        )
    if (
        extraction_mode == 'text'
        and answer_format == 'text'
        and required_aspects != 'answer_only'
    ):
        raise ValueError(
            'Cannot combine extraction mode "text" with answer format "text".'
        )

    if answer_format == 'text':
        free_text_answer, extraction_node_strs = parse_text_answer(
            answer_text,
            answer_has_multiple_statements
        )
    elif answer_format == 'structured_text':
        free_text_answer, extraction_node_strs = parse_structured_text_answer(
            answer_text,
            required_aspects
        )

    elif answer_format == 'json':
        free_text_answer, extraction_node_strs = parse_json_answer(
            answer_text,
            required_aspects
        )

    else:
        raise ValueError(f'Unknown answer format {answer_format}.')

    if required_aspects in ['answer_and_segments', 'segments_only']:
        # Get extraction nodes from string ids
        if answer_has_multiple_statements:
            extraction_nodes = []
            for l in extraction_node_strs:
                extraction_nodes.append(
                    get_extraction_nodes(
                        l,
                        extraction_candidates,
                        extraction_mode,
                        node_id_to_node_mapping
                    )
                )
        else:
            extraction_nodes = get_extraction_nodes(
                extraction_node_strs,
                extraction_candidates,
                extraction_mode,
                node_id_to_node_mapping
            )
    else:
        # Make empty list of extraction nodes
        if answer_has_multiple_statements:
            extraction_nodes = [[] for _ in free_text_answer]
        else:
            extraction_nodes = []

    # Process free text answer
    if classes is not None:
        # If we are doing a classification task, normalize the answer to canonical
        # label
        free_text_answer = find_label(
            classes, free_text_answer
        )
        if free_text_answer is None:
            free_text_answer = 'none'
    else:
        # If not classification task, check if answer is 'unanswerable'
        free_text_answer = map_to_unanswerable(
            free_text_answer,
            unanswerable_keywords,
            answer_has_multiple_statements
        )

    if return_all_extraction_candidates:
        # Replace extraction nodes with all extraction candidates
        if answer_has_multiple_statements:
            extraction_nodes = [
                extraction_candidates for _ in free_text_answer
            ]
        else:
            extraction_nodes = extraction_candidates

    # Remove extraction nodes if answer is unanswerable or in "not mentioned"
    # class
    if not should_prediction_be_considered_for_attribution(
        free_text_answer,
        classes,
        task_name
    ):
        if answer_has_multiple_statements:
            extraction_nodes = [[] for _ in free_text_answer]
        else:
            extraction_nodes = []

    return free_text_answer, extraction_nodes


def parse_json_answer(
        answer_text: str,
        required_aspects: str
) -> Tuple[str, List[str]]:
    """
    Expected format:
    {
        "answer": "<answer>",
        "segments": ["<segment_1>", "<segment_2>", ...]
    }
    :param answer_text: Raw generation from the model
    :param required_aspects: "answer_only", "segments_only", or "answer_and_segments"
    :return:
    """
    answer_text = answer_text.strip()
    try:
        answer_json = json.loads(answer_text)
    except json.JSONDecodeError:
        return 'parsing error', []

    if required_aspects == 'answer_only' or required_aspects == 'answer_and_segments':
        try:
            free_text_answer = answer_json['answer']
        except KeyError:
            free_text_answer = 'parsing error'
        except TypeError:
            free_text_answer = 'parsing error'
    else:
        free_text_answer = ''

    if required_aspects == 'segments_only' or required_aspects == 'answer_and_segments':
        if 'segment' in answer_json:
            key = 'segment'
        else:
            key = 'segments'
        try:
            extraction_nodes = answer_json[key]
            if isinstance(extraction_nodes, str):
                extraction_nodes = [extraction_nodes]
        except KeyError:
            extraction_nodes = []
        except TypeError:
            extraction_nodes = []
    else:
        extraction_nodes = []

    return free_text_answer, extraction_nodes


def parse_structured_text_answer(
        answer_text: str,
        required_aspects: str
) -> Tuple[str, List[str]]:
    """
    Expected format:
    <answer>
    Segments:
    - <segment_1>
    - <segment_2>
    - ...
    :param answer_text: Raw generation from the model
    :param required_aspects: "answer_only", "segments_only", or "answer_and_segments"
    :return:
    """
    if required_aspects == 'answer_only':
        # return the complete text
        return answer_text, []

    # Find the bullet points
    pattern = '\n\\s*[-\\*] (.+)(?=$|\n)'
    matches = re.findall(pattern, answer_text)
    # Remove whitespace and dots ('...')
    for i, match in enumerate(matches):
        matches[i] = match.strip().strip('.')

    if 'Segments:' in answer_text:
        # free text answer stops here
        pos = answer_text.find('Segments:')
        answer_text = answer_text[:pos]
    elif len(matches) > 0:
        # free text answer stops at first bullet
        pos = re.search(pattern, answer_text).start()
        answer_text = answer_text[:pos]

    if required_aspects == 'segments_only':
        return '', matches
    else:
        return answer_text.strip(), matches


def parse_text_answer(
        answer_text: str,
        answer_has_multiple_statements: bool
) -> Tuple[str, List[str]] | Tuple[List[str], List[List[str]]]:
    """
    Split answer text if it consists of multiple sentences
    Example text: '[0] I like pizza [1]. This is the next sentence [2], and a subclause[4]. Now this is yet another sentence. And here comes the final sentence [6]'
    if answer_has_multiple_statements is True, returns:
    (
        [
            'I like pizza.',
            'This is the next sentence, and a subclause.',
            'Now this is yet another sentence.'
            'And here comes the final sentence'
        ],
        [
            ['[0]', '[1]'],
            ['[2]', '[4]'],
            [],
            ['[6]']
        ]
    )
    else returns:
    (
        'I like pizza. This is the next sentence, and a subclause. Now this is yet another sentence. And here comes the final sentence',
        ['[0]', '[1]', '[2]', '[4]', '[6]']
    )
    :param answer_text:
    :param answer_has_multiple_statements:
    :return:
    """
    pattern = r'\[[0-9]+\]'

    statements = nltk.sent_tokenize(answer_text)

    # Find citations
    matches = [
        re.findall(pattern, statement)
        for statement in statements
    ]
    # Replace citations
    pattern = r' ?\[[0-9]+\] ?'
    statements = [
        re.sub(pattern, '', statement) for statement in statements
    ]

    if not answer_has_multiple_statements:
        # Re-join if answer does not have multiple statements
        return ' '.join(statements), [m for match_list in matches for m in match_list]

    if answer_has_multiple_statements and len(statements) == 0:
        # Handle cases with empty answer
        statements = ['']
        matches = [[]]

    return statements, matches


def make_unanswerable_phrases() -> List[str]:
    verbs = [
        'provide',
        'mention',
        'state',
        'specify',
        'define',
        'report',
        'name',
        'offer'
    ]
    participles = [
        'provided',
        'mentioned',
        'stated',
        'specified',
        'defined',
        'reported',
        'named',
        'offered'
    ]
    adverbs = [
        'explicitly',
        'specifically',
        'directly',
        'clearly'
    ]
    phrases = []
    phrases.extend([
        f'not {verb}' for verb in verbs
    ])
    phrases.extend([
        f'not {adverb} {verb}' for adverb in adverbs for verb in verbs
    ])
    phrases.extend([
        f'not {participle}' for participle in participles
    ])
    phrases.extend([
        f'not {adverb} {participle}' for adverb in adverbs for participle in participles
    ])
    return phrases


UNANSWERABLE_PHRASES = make_unanswerable_phrases()


def map_to_unanswerable(
        free_text_answer: str | List[str],
        unanswerable_keywords: List[str],
        answer_has_multiple_statements: bool
) -> str:
    """Replace free text answer with "unanswerable" if it is the same as any of
    the given keywords"""

    original_answer = free_text_answer
    if answer_has_multiple_statements:
        free_text_answer = ' '.join(free_text_answer)

    normalized_answer = free_text_answer.lower().strip()

    is_unanswerable = False
    if 'unanswerable' in normalized_answer:
        is_unanswerable = True
    elif any([
        keyword in normalized_answer
        for keyword in unanswerable_keywords
    ]):
        is_unanswerable = True
    elif " not " in normalized_answer:
        if any([
            phrase in normalized_answer
            for phrase in UNANSWERABLE_PHRASES
        ]):
            is_unanswerable = True

    if is_unanswerable:
        new_answer = 'unanswerable'
        if answer_has_multiple_statements:
            new_answer = [new_answer]
        return new_answer
    else:
        return original_answer


def get_unparsable_proportion(
        predictions
) -> float:
    parsing_error_count = 0
    for prediction in predictions:
        if prediction.free_text_answer == 'parsing error':
            parsing_error_count += 1
    return parsing_error_count / len(predictions)




################################################################################
# OpenAI Cost Estimation
################################################################################


def estimate_cost_for_single_document(
        prompt_tokens: int,
        document_tokens: int,
        answer_tokens: int,
        model: str
) -> Decimal:
    if 'gpt-35-turbo' in model:
        prompt_token_cost = Decimal(0.003) / 1000
        completion_token_cost = Decimal(0.004) / 1000

    elif 'gpt-4' in model:
        prompt_token_cost = Decimal(0.06) / 1000
        completion_token_cost = Decimal(0.12) / 1000

    else:
        raise NotImplementedError

    cost_per_document = (
        ((prompt_tokens + document_tokens) * prompt_token_cost)
        + (answer_tokens * completion_token_cost)
    )

    return cost_per_document


def estimate_cost_for_n_documents(
        n_documents: int,
        prompt_tokens: int,
        document_tokens: int,
        answer_tokens: int,
        model: str
) -> Decimal:
    cost = n_documents * estimate_cost_for_single_document(
        prompt_tokens,
        document_tokens,
        answer_tokens,
        model
    )

    return cost


def find_label(
        classes: List[str],
        free_text_answer: str
):
    """
    It sometimes happens that models do not just output the classification label,
    but add more text. In these cases, we still want to find the classification
    label in the produced text. We need to be careful with string matching, because
    some labels might be substrings of other labels (e.g. "supported" and "not
    supported"). Therefore, the labels need to be in the right order.
    :param classes: List of lowercase string class names. List should be ordered
        such that earlier items are not substrings of later items.
        Incorrect: ['supported', 'not supported']
        Correct: ['not supported', 'supported']
    :param free_text_answer: The predicted answer from the model
    :return:
    """
    if len(classes) == 0:
        raise ValueError('No classes given!')
    free_text_answer = free_text_answer.lower().strip()
    predicted_label = None
    for class_name in classes:
        if class_name in free_text_answer:
            predicted_label = class_name
            break
    return predicted_label


def is_instance_answerable(instance):
    """Determine if instance is unanswerable depending on number of annotations"""
    # Check if free text answer consists of multiple statements
    if isinstance(instance.free_text_answer[0], list):
        free_text_answers = [
            ' '.join(statement_list)
            for statement_list in instance.free_text_answer
        ]
    else:
        free_text_answers = instance.free_text_answer
    if len(free_text_answers) <= 3:
        # Instance is unanswerable if all annotators annotated unanswerable
        if all([answer.strip().lower() == 'unanswerable' for answer in free_text_answers]):
            is_answerable = False
        else:
            is_answerable = True
    else:
        # Instance is unanswerable if at least all but one annotator annotated
        # unanswerable
        if sum([
            1 if answer.strip().lower() == 'unanswerable' else 0
            for answer in free_text_answers
        ]) >= len(instance.free_text_answer) - 1:
            is_answerable = False
        else:
            is_answerable = True
    return is_answerable


def is_prediction_answerable(
        prediction: BasePrediction | str
):
    """Determine if prediction is unanswerable"""
    # CHeck if prediction consists of multiple statements
    if isinstance(prediction, BasePrediction):
        free_text_answer = prediction.free_text_answer
    else:
        free_text_answer = prediction
    if isinstance(free_text_answer, list):
        free_text_answer = ' '.join(free_text_answer)
    else:
        free_text_answer = free_text_answer

    if free_text_answer.lower().strip() == 'unanswerable':
        is_answerable = False
    else:
        is_answerable = True

    return is_answerable


def should_prediction_be_considered_for_attribution(
        prediction: BasePrediction | str,
        classes: List[str] = None,
        task_name: str = None
):
    if isinstance(prediction, BasePrediction):
        free_text_answer = prediction.free_text_answer
    else:
        free_text_answer = prediction
    if task_name is None:
        task_name = prediction.task_name
    if task_name in ['qasper', 'natural_questions']:
        should_be_considered = is_prediction_answerable(free_text_answer)
    elif task_name in ['contract_nli', 'wice', 'evidence_inference']:
        label = find_label(classes, free_text_answer)
        if label is None:
            should_be_considered = False
        elif task_name == 'contract_nli' and label == 'not mentioned':
            should_be_considered = False
        elif task_name == 'wice' and label == 'not supported':
            should_be_considered = False
        else:
            should_be_considered = True
    else:
        should_be_considered = True

    return should_be_considered
