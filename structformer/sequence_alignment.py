import bisect
import dataclasses
import logging
from typing import List, Any

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Span:
    """A span in a string."""
    left: int  # inclusive
    right: int  # exclusive
    content: Any


@dataclasses.dataclass
class SpanMapping:
    """A mapping that can quickly find the span at a given string index."""
    spans: List[Span]
    lefts: List[int] = dataclasses.field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.spans.sort(key=lambda x: x.left)
        self.lefts = [x.left for x in self.spans]

    def __getitem__(self, item: int) -> Any:
        """Get the content of the span at the given string index.

        Args:
            item: String index.

        Returns:
            Content of the span at the given string index.
        """
        idx = bisect.bisect(self.lefts, item)
        if idx > 0:
            idx -= 1
        return self.spans[idx].content

    def get_content_list(self, indices: List[int]) -> List[Any]:
        """Get the contents of the spans for the sorted list of string indices.

        The string indices must be sorted in ascending order!

        Args:
            indices: List of string indices sorted in ascending order.

        Returns:
            List of contents of the spans at the given string indices.
        """
        contents = []
        span_ix = 0
        current_span = self.spans[0]
        for left_ix in indices:
            while left_ix >= current_span.right:
                span_ix += 1
                current_span = self.spans[span_ix]
            contents.append(current_span.content)
        return contents


if __name__ == "__main__":
    spans = [
        Span(0, 12, "A"),
        Span(12, 13, "B"),
        Span(13, 25, "C"),
        Span(25, 30, "D")
    ]

    span_mapping = SpanMapping(spans)

    assert span_mapping[0] == "A", span_mapping[0]
    assert span_mapping[1] == "A", span_mapping[1]
    assert span_mapping[11] == "A", span_mapping[11]
    assert span_mapping[12] == "B", span_mapping[12]
    assert span_mapping[13] == "C", span_mapping[13]
    assert span_mapping[24] == "C", span_mapping[24]
    assert span_mapping[25] == "D", span_mapping[25]
    assert span_mapping[29] == "D", span_mapping[29]

    lefts = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29]
    contents = span_mapping.get_content_list(lefts)

    for left, content in zip(lefts, contents):
        print(left, content)
