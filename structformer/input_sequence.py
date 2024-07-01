"""Code for transforming ITG documents into transformer input strings.

The functionalities in this module should be accessed via the two public methods:

- to_input_sequence
- get_structural_tokens
"""
import logging
from typing import Tuple, List

from intertext_graph.itgraph import IntertextDocument, Etype, Node, Edge

from structformer.sequence_alignment import SpanMapping, Span
from config_lib.base_config import BaseConfig

logger = logging.getLogger(__name__)


def _join_parts_and_create_span_mapping(
        parts: List[str],
        nodes: List[Node],
        separator: str
) -> Tuple[str, SpanMapping]:
    """Join the given list of string parts and create a mapping to the nodes.

    The separators will always be counted to the parts that precede them.

    Args:
        parts: Given list of string parts.
        nodes: Given list of corresponding nodes.
        separator: Separator string.

    Returns:
        The joined string parts and a mapping to the nodes.
    """
    spans = []
    current_left = 0
    node_sep_len = len(separator)
    separators = []
    final_idx = len(parts) - 1
    for ix, (part, node) in enumerate(zip(parts, nodes)):
        right = current_left + len(part)
        if ix < len(parts) - 1:
            right += node_sep_len
        spans.append(Span(current_left, right, node))
        if ix < final_idx:
            if node.ntype in ['table-item'] and nodes[ix + 1].ntype in ['table-item']:
                separators.append(' ')
            else:
                separators.append(separator)
        else:
            separators.append('')

        current_left = right

    joined_text = ''.join([text + sep for text, sep in zip(parts, separators)])

    return joined_text, SpanMapping(spans)


################################################################################
# to text
################################################################################

_plain_text_structural_tokens = []


def _to_plain_text(
        document: IntertextDocument,
        config: BaseConfig
) -> Tuple[str, SpanMapping]:
    """Transform the given ITG document into a plain text linear string.

    This method requires the following configuration keys:

    - is_include_node_types
    - is_replace_newlines
    - is_node_separator

    Args:
        document: ITG document.
        config: Complete Hydra configuration.

    Returns:
        Transformer input string, Node mapping.
    """
    include_node_types = config.is_include_node_types
    replace_newlines = config.is_replace_newlines
    node_separator = config.is_node_separator

    parts = []
    nodes = []
    for node in document.unroll_graph():
        if include_node_types is None or node.ntype in include_node_types:
            if replace_newlines:
                text = node.content.replace("\n", " ")
            else:
                text = node.content

            parts.append(text)
            nodes.append(node)

    return _join_parts_and_create_span_mapping(parts, nodes, node_separator)


################################################################################
# to text with node ids
################################################################################

def _to_text_with_node_ids(
        document: IntertextDocument,
        config: BaseConfig
) -> Tuple[str, SpanMapping]:
    """Place the id of a node in front of it.

    Transform the given ITG document into a linear string and insert the id of
    a node in front of it.

    This method requires the following configuration keys:

    - is_include_node_types
    - is_replace_newlines
    - is_node_separator

    Args:
        document: ITG document.
        config: Complete Hydra configuration.

    Returns:
        Transformer input string, Node mapping.
    """
    include_node_types = config.is_include_node_types
    replace_newlines = config.is_replace_newlines
    node_separator = config.is_node_separator

    parts, nodes = [], []
    for node in document.unroll_graph():
        if include_node_types is None or node.ntype in include_node_types:
            node_id = node.ix.split('_')[-1]
            structural_token = eval(f'f"""{config.node_id_template}"""')

            if replace_newlines:
                text = node.content.replace("\n", " ")
            else:
                text = node.content

            node_input_seq = f"{structural_token} {text}"
            parts.append(node_input_seq)
            nodes.append(node)

    return _join_parts_and_create_span_mapping(parts, nodes, node_separator)


################################################################################
# access
################################################################################


def to_input_sequence(
        document: IntertextDocument,
        config: BaseConfig
) -> Tuple[str, SpanMapping]:
    """Transform the given ITG document into a linear string.

    This method requires the following configuration keys:

    - input_sequence/mode

    Args:
        document: ITG document.
        config: Complete Hydra configuration.

    Returns:
        Transformer input string, Node mapping.
    """
    mode = config.is_mode

    if mode == "vanilla":
        return _to_plain_text(document, config)
    elif mode == "text_with_node_ids":
        return _to_text_with_node_ids(document, config)
    else:
        logger.error(f"Unknown input sequence mode '{mode}'!")
        assert False, f"Unknown input sequence mode '{mode}'!"


if __name__ == "__main__":
    from omegaconf import OmegaConf

    doc = IntertextDocument([], [], "doc")

    title_node = Node("Some Great Title", ntype="article-title")
    doc.add_node(title_node)

    abstract_node = Node("Abstract", ntype="abstract")
    doc.add_node(abstract_node)
    doc.add_edge(Edge(title_node, abstract_node, Etype.PARENT))
    doc.add_edge(Edge(title_node, abstract_node, Etype.NEXT))

    abstract_paragraph_node = Node("This is a concise abstract.", ntype="p")
    doc.add_node(abstract_paragraph_node)
    doc.add_edge(Edge(abstract_node, abstract_paragraph_node, Etype.PARENT))
    doc.add_edge(Edge(abstract_node, abstract_paragraph_node, Etype.NEXT))

    section_node = Node("A Descriptive Section Title", ntype="title")
    doc.add_node(section_node)
    doc.add_edge(Edge(title_node, section_node, Etype.PARENT))
    doc.add_edge(Edge(abstract_paragraph_node, section_node, Etype.NEXT))

    paragraph_node_1 = Node("An interesting first paragraph.", ntype="p")
    doc.add_node(paragraph_node_1)
    doc.add_edge(Edge(section_node, paragraph_node_1, Etype.PARENT))
    doc.add_edge(Edge(section_node, paragraph_node_1, Etype.NEXT))

    paragraph_node_2 = Node("An interesting second paragraph.", ntype="p")
    doc.add_node(paragraph_node_2)
    doc.add_edge(Edge(section_node, paragraph_node_2, Etype.PARENT))
    doc.add_edge(Edge(paragraph_node_1, paragraph_node_2, Etype.NEXT))

    print("Document Plaintext:")
    print(doc.to_plaintext())

    print("\n" * 4)
    print("Input Sequence")
    config = OmegaConf.create({
        "input_sequence": {
            "mode": "text_with_node_depths",
            "node_separator": " ",
            "do_close": False,
            "replace_newlines": False,
            "include_node_types": ["article-title", "abstract", "title", "p"]
        }
    })

    text, spans = to_input_sequence(doc, config)
    print(text)

    print("\n" * 4)

    print(spans)
    for span in spans.spans:
        print(f"'{text[span.left:span.right]}'", span.content.ntype)
