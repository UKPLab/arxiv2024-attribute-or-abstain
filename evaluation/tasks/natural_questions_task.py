import json
import re
import time
from typing import List, ClassVar
import logging
from pathlib import Path

from intertext_graph.itgraph import IntertextDocument

from config_lib.base_config import BaseConfig
from evaluation.common import (
    BaseInstance, BaseTask, BaseResult, BasePrediction, Statistics,
    Partition, SingleFileDataset
)

logger = logging.getLogger(__name__)


class NQOracleWrapperBatch:
    def __init__(
            self,
            instances: List[BaseInstance]
    ):
        self.documents: List[IntertextDocument] = [
            instance.document
            for instance in instances
        ]


class NaturalQuestionsDataset(SingleFileDataset):
    """Dataset that gets NQ instances from NQ itgs stored in folder"""
    def __init__(
            self,
            path: Path,
            valid_filenames: List[str],
            null_answer_string: str,
            use_first_non_null_answer: bool
    ) -> None:
        super(NaturalQuestionsDataset, self).__init__(path, valid_filenames)
        self._null_answer_string = null_answer_string
        self._use_first_non_null_answer = use_first_non_null_answer

    def _create_instance_from_document(
            self,
            document: IntertextDocument
    ) -> BaseInstance:
        """
        Create NQInstance from ITG.
        If self._use_first_non_null_answer is True, the first non-null answer
        for cases with multiple annotations is used as the ground truth.
        Mod for extraction benchmark: Long answer only annotations are not
        considered! Instances which only have annotations with answer type
        long or unanswerable are not considered if they have more than one
        annotation with answer type "long". (See scripts/nq_filter.py).
        """
        question = document.meta['question_text']
        example_id = document.meta['example_id']
        long_answer_candidates = []

        # Get the long answer candidates from the document
        for entry in document.meta['long_answer_candidates']:
            if 'node_ix' not in entry:
                # For some long answer candidates the node was not found in the
                # document during preprocessing. We skip these candidates.
                continue
            long_answer_candidates.append(document.get_node_by_ix(entry['node_ix']))

        ########################################################################
        # Taken from the LongT5 code
        # https://github.com/google-research/longt5/blob/master/longt5/data/nq_preprocess.py
        def _pad_punctuation(text):
            """Adds spaces around punctuation."""
            # Add space around punctuation.
            text = re.sub(r"(\W)", r" \1 ", text)
            # Collapse consecutive whitespace into one space.
            text = re.sub(r"\s+", " ", text)
            return text

        def _has_short_answer(a_):
            return bool(a_["short_answers"])

        def _is_yes_no_answer(a_):
            return a_["yes_no_answer"] in ("YES", "NO")

        def _has_long_answer(a_):
            return a_['long_answer']['candidate_index'] != -1

        def _is_unanswerable(a_):
            has_short_answer = _has_short_answer(a_)
            is_yes_no_answer = _is_yes_no_answer(a_)
            has_long_answer = _has_long_answer(a_)
            return not any([has_short_answer, is_yes_no_answer, has_long_answer])

        # Answer -- train example has one annotation and dev example has up to five.
        free_text_answers = []
        answer_types = []
        extraction_nodes = []
        document_tokens = document.meta['document_text'].split(' ')
        # Go over the annotations, get short answers, answer types and long answers
        for i, a in enumerate(document.meta["annotations"]):
            answer_tokens = []
            short_answer = ''
            answer_type = "unanswerable"
            # Check if answer is short, yes/no or does not exist
            if _has_short_answer(a):
                # Answer is short, get short answer tokens
                answer_type = "short"
                for sa in a["short_answers"]:
                    for pos in range(sa["start_token"], sa["end_token"]):
                        answer_tokens.append(document_tokens[pos])
                answer = " ".join(answer_tokens)
                short_answer = answer
            elif _is_yes_no_answer(a):
                # Get yes/no answer
                short_answer = a["yes_no_answer"].lower()
                answer_type = "boolean"
            elif _has_long_answer(a):
                short_answer = 'null'
                answer_type = 'long'
            else:
                answer_type = 'unanswerable'
                short_answer = 'unanswerable'

            if not _has_long_answer(a):
                # We ignore answerable annotations without evidence
                extraction_nodes_for_annotation = []
            else:
                extraction_nodes_for_annotation = [document.get_node_by_ix(
                    document.meta['long_answer_candidates'][a['long_answer']['candidate_index']]['node_ix']
                )]

            # Filter
            if answer_type == 'long':
                # We ignore long answer only annotations
                continue
            if answer_type != 'unanswerable' and not extraction_nodes_for_annotation:
                # We ignore answerable annotations without evidence
                continue
            if (
                    (answer_type != 'unanswerable')
                    and extraction_nodes_for_annotation[0].ntype != 'p'
            ):
                # We ignore annotations with evidence not in paragraphs
                continue

            answer_types.append(answer_type)
            free_text_answers.append(short_answer)
            extraction_nodes.append(extraction_nodes_for_annotation)

        free_text_answers = [_pad_punctuation(a) for a in free_text_answers]

        ########################################################################

        instance = BaseInstance(
            task_name='natural_questions',
            example_id=str(example_id),
            document=document,
            prompt=question,
            extraction_level='p',
            extraction_candidates=long_answer_candidates,
            free_text_answer=free_text_answers,
            answer_type=answer_types,
            extraction_nodes=extraction_nodes,
            question=question,
            statement=''
        )

        self._remove_trash_at_end(instance)

        self._remove_empty_nodes(instance)

        return instance

    @staticmethod
    def _remove_empty_nodes(
        instance: BaseInstance
    ):
        nodes = instance.document.nodes
        for node in nodes:
            if len(node.content) == 0 and node.ntype in ['list-item', 'table-item']:
                instance.document.remove_node(node, preserve_next_edge=True)
                if node in instance.extraction_candidates:
                    instance.extraction_candidates.remove(node)


    @staticmethod
    def _remove_trash_at_end(
            instance: BaseInstance
    ):
        TRASH_STRS = [
            'See also ( edit )',
            'References ( edit )'
        ]

        first_trash_node_idx = len(instance.document.nodes)
        # Find first trash node
        for i, n in enumerate(instance.document.nodes):
            if any(trash_str in n.content for trash_str in TRASH_STRS):
                first_trash_node_idx = i
                break

        trash_nodes = instance.document.nodes[first_trash_node_idx:]

        assert all(en not in trash_nodes for en in instance.extraction_nodes)

        for n in trash_nodes:
            instance.document.remove_node(n, preserve_next_edge=False)
            if n in instance.extraction_candidates:
                instance.extraction_candidates.remove(n)


    def get_subset(self, items: slice):
        """Get the subset of the dataset at the specified indices."""
        filenames = self.filenames[items]
        return self.__class__(
            self.dir_path,
            filenames,
            self._null_answer_string,
            self._use_first_non_null_answer
        )


class NaturalQuestionsTask(BaseTask):
    """Implements data loading and evaluation for Natural Questions."""
    task_name: ClassVar[str] = "natural_questions"

    def __init__(self, config: BaseConfig, stats: Statistics) -> None:
        super(NaturalQuestionsTask, self).__init__(config, stats)
    
    def load_instances_from_itg(self) -> List[List[BaseInstance | SingleFileDataset]]:
        logger.info("Load the NQ-ITG dataset.")
        tick = time.time()

        data_path = self.config.location.datasets / 'natural_questions' / 'natural_questions_itg'

        # Get lists of filenames for train, dev and test datasets
        # First check if path to files with valid filenames are given. If not,
        # get the full list of filenames from the respective directories.
        train_filenames = []
        dev_filenames = []
        test_filenames = []
        if self.config.task.train_filenames is not None:
            with open(data_path / self.config.task.train_filenames) as file:
                train_filenames = json.load(file)
        if self.config.task.dev_filenames is not None:
            with open(data_path / self.config.task.dev_filenames) as file:
                dev_filenames = json.load(file)
        if self.config.task.test_filenames is not None:
            with open(data_path / self.config.task.test_filenames) as file:
                test_filenames = json.load(file)

        # Initialize dataset objects
        # Instances are loaded from disk when requested from the dataloader
        train_path = data_path / 'train'
        train_instances = NaturalQuestionsDataset(
            train_path,
            valid_filenames=train_filenames,
            null_answer_string=self.config.task.null_answer_string,
            use_first_non_null_answer=self.config.task.natural_questions_use_first_non_null_answer
        )

        dev_path = data_path / 'val'
        dev_instances = NaturalQuestionsDataset(
            dev_path,
            valid_filenames=dev_filenames,
            null_answer_string=self.config.task.null_answer_string,
            use_first_non_null_answer=self.config.task.natural_questions_use_first_non_null_answer
        )

        test_path = data_path / 'dev'
        test_instances = NaturalQuestionsDataset(
            test_path,
            valid_filenames=test_filenames,
            null_answer_string=self.config.task.null_answer_string,
            use_first_non_null_answer=self.config.task.natural_questions_use_first_non_null_answer
        )

        tack = time.time()
        logger.info(f"Created {len(train_instances)} train instances, "
                    f"{len(dev_instances)} dev instances, and "
                    f"{len(test_instances)} test instances in {tack - tick:0.4f}s.")

        self.stats.stats["task-initialization.num-train-instances"] = len(train_instances)

        return [train_instances, dev_instances, test_instances]