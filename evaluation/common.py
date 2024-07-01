from __future__ import annotations

import abc
import dataclasses
import enum
import hashlib
import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, List, ClassVar, Tuple, Optional, Union

from intertext_graph.itgraph import IntertextDocument, Node
from torch.utils.data import Dataset

from config_lib.base_config import BaseConfig
from config_lib.config_container import ConfigContainer

logger = logging.getLogger(__name__)


################################################################################
# Base classes for dataset instances, predictions, results, and tasks
################################################################################

@dataclasses.dataclass
class BaseInstance(abc.ABC):
    """
    Base class for all dataset instances.

    :param task_name: Name of the task.
    :param example_id: ID of the example.

    :param document: IntertextDocument object that contains evidence.
    :param prompt: Prompt string that contains whatever is needed besides the document.
        This is mostly used for fine tuned models, which do not need specific prompt tuning.
    :param question: For datasets that have a question, this is the question string. In
        other cases, this might be generic, e.g. "Is the claim true?".
    :param statement: For datasets that have a statement, this is the statement string.
        E.g. in a fact verification task, this is the claim.
    :param extraction_level: Whether the evidence is on the paragraph or sentence level.
    :param extraction_candidates: List of candidate evidence nodes.

    :param free_text_answer: List of gold free text answers. Depending on the task, this can
    be an actual answer, a veracity label, an entailment label, or even None.
    This is a list because some tasks have multiple correct answers / annotations.
    :param extraction_nodes: List of lists of gold evidence nodes. Can be None or empty.
    This is a list of lists because some tasks have multiple answers / annotations.
    """
    task_name: str
    example_id: str

    # input:
    document: IntertextDocument
    prompt: Optional[str]
    question: Optional[str]
    statement: Optional[str]
    extraction_level: str
    extraction_candidates: List[Node]

    # output:
    free_text_answer: Optional[List[str]] | Optional[List[List[str]]]  # There can be multiple correct answers
    answer_type: Optional[List[str]]
    extraction_nodes: Optional[List[List[Node]]] | Optional[List[List[List[Node]]]]  # There can be multiple correct sets of evidence nodes

    # Optional
    additional_info: str = ''
    answer_has_multiple_statements: bool = False

    def to_json_dict(self):
        if self.answer_has_multiple_statements:
            extraction_nodes = [
                [
                    [
                        n.ix for n in extraction_nodes_for_statement
                    ] for extraction_nodes_for_statement in extraction_nodes_for_annotation
                ] for extraction_nodes_for_annotation in self.extraction_nodes
            ]
        else:
            extraction_nodes = [
                [
                    n.ix for n in extraction_nodes_for_annotation
                ] for extraction_nodes_for_annotation in self.extraction_nodes
            ]
        json_dict = {
            'task_name': self.task_name,
            'example_id': self.example_id,
            'document': self.document.to_json(indent=None),
            'prompt': self.prompt,
            'question': self.question,
            'statement': self.statement,
            'extraction_level': self.extraction_level,
            'extraction_candidates': [node.ix for node in self.extraction_candidates],
            'free_text_answer': self.free_text_answer,
            'answer_type': self.answer_type,
            'extraction_nodes': extraction_nodes,
            'answer_has_multiple_statements': self.answer_has_multiple_statements
        }
        return json_dict

    @classmethod
    def from_json_dict(cls, json_dict: Dict):
        doc_dict = json.loads(json_dict['document'])
        document = IntertextDocument._from_json(doc_dict)
        extraction_candidates = [
            document.get_node_by_ix(node_ix)
            for node_ix in json_dict['extraction_candidates']
        ]
        extraction_nodes = json_dict['extraction_nodes']
        if json_dict['answer_has_multiple_statements']:
            extraction_nodes = [
                [
                    [
                        document.get_node_by_ix(node_ix)
                        for node_ix in extraction_nodes_for_statement
                    ] for extraction_nodes_for_statement in extraction_nodes_for_annotation
                ] for extraction_nodes_for_annotation in extraction_nodes
            ]
        else:
            extraction_nodes = [
                [
                    document.get_node_by_ix(node_ix)
                    for node_ix in extraction_nodes_for_annotation
                ] for extraction_nodes_for_annotation in extraction_nodes
            ]
        return cls(
            task_name=json_dict['task_name'],
            example_id=json_dict['example_id'],
            document=document,
            prompt=json_dict['prompt'],
            question=json_dict['question'],
            statement=json_dict['statement'],
            extraction_level=json_dict['extraction_level'],
            extraction_candidates=extraction_candidates,
            free_text_answer=json_dict['free_text_answer'],
            answer_type=json_dict['answer_type'],
            extraction_nodes=extraction_nodes,
            answer_has_multiple_statements=json_dict['answer_has_multiple_statements']
        )


@dataclasses.dataclass
class BasePrediction(abc.ABC):
    """
    Base class for all dataset predictions.

    :param task_name: Name of the task.
    :param example_id: ID of the example.
    :param free_text_answer: Free text answer. Depending on the task, this can
    be an actual answer, a veracity label, an entailment label, or even None.
    :param extraction_nodes: List of evidence nodes. Can be None or empty.
    :param raw_generation: The raw generation output. This is mostly for
    debugging purposes
    """
    task_name: str
    example_id: str

    free_text_answer: str | List[str]
    extraction_nodes: List[Node] | List[List[Node]]
    raw_generation: str

    def to_json_dict(self) -> Dict[str, Any]:
        """Return the prediction object as a JSON-serializable dictionary."""
        if len(self.extraction_nodes) > 0:
            if isinstance(self.extraction_nodes[0], list):
                serialized_extraction_nodes = [
                    [node.ix for node in node_list]
                    for node_list in self.extraction_nodes
                ]
            else:
                serialized_extraction_nodes = [
                    node.ix for node in self.extraction_nodes
                ]
        else:
            serialized_extraction_nodes = []
        return {
            "task_name": self.task_name,
            "example_id": self.example_id,
            "free_text_answer": self.free_text_answer,
            "extraction_nodes": serialized_extraction_nodes,
            'raw_generation': self.raw_generation
        }

    @classmethod
    def from_json_dict(
            cls,
            json_dict: Dict,
            instance: BaseInstance
    ):
        extraction_node_ids = json_dict['extraction_nodes']
        if len(extraction_node_ids) > 0:
            if isinstance(extraction_node_ids[0], list):
                extraction_nodes = [
                    [
                        instance.document.get_node_by_ix(node_ix)
                        for node_ix in node_id_list
                    ]
                    for node_id_list in extraction_node_ids
                ]
            else:
                extraction_nodes = [
                    instance.document.get_node_by_ix(node_ix)
                    for node_ix in extraction_node_ids
                ]
        else:
            extraction_nodes = []
        json_dict['extraction_nodes'] = extraction_nodes
        return cls(**json_dict)


@dataclasses.dataclass
class BaseResult(abc.ABC):
    """
    Base class for all dataset evaluation results.

    :param task_name: Name of the task.
    :param extraction_f1: F1 score for the evidence nodes.

    :param unparsable_proportion: The proportion of replies from the model that
        could not be parsed according to the specified answer format.

    :param gold_descriptive_stats: Descriptive statistics about the gold
        (ground-truth) data
    :param prediction_descriptive_stats: Descriptive statistics about the predictions
    """
    task_name: str
    extraction_f1: float

    unparsable_proportion: float

    gold_descriptive_stats: Optional[BaseDescriptiveStats]
    prediction_descriptive_stats: Optional[BaseDescriptiveStats]

    @abc.abstractmethod
    def to_json_dict(self) -> Dict[str, Any]:
        """Return the result object as a JSON-serializable dictionary."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def score(self) -> float:
        """Score that determines what is considered as 'better' during fine-tuning."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def table_entries(self) -> Dict[str, float]:
        """Result values that should appear in the overall results table."""
        raise NotImplementedError


@dataclasses.dataclass
class BaseDescriptiveStats:
    """Descriptive statistics for gold data and predictions"""
    mean_free_text_answer_length: float = None
    mean_n_extraction_nodes: float = None

    def to_json_dict(self) -> Dict[str, Any]:
        return_dict = {}
        for k, v in self.__dict__.items():
            if v is not None:
                return_dict[k] = v
        return return_dict


class BaseTask(abc.ABC):
    """
    Base class for all downstream evaluation tasks.

    :param task_name: Name of the task.
    :param single_evidence_only: Indicates whether there is a single true
    evidence node per instance.
    :param config: Configuration object.
    :param stats: Statistics object.
    """
    task_name: ClassVar[str] = "BaseTask"

    # Task properties
    single_evidence_only: bool = False  # Indicates whether there is a single true evidence node per instance

    config: BaseConfig
    stats: "Statistics"

    def __init__(self, config: BaseConfig, stats: "Statistics") -> None:
        super(BaseTask, self).__init__()
        assert config.task.task_name == self.task_name
        self.config = config
        self.stats = stats

        if config.instances_path is not None:
            self._train_instances, self._dev_instances, self._test_instances = (
                self.load_preprocessed_instances(config.instances_path)
            )
            assert self._train_instances[0].task_name == self.task_name
        else:
            self._train_instances, self._dev_instances, self._test_instances = (
                self.load_instances_from_itg()
            )

    @property
    def train_instances(self) -> List[BaseInstance] | SingleFileDataset:
        """Train instances of the dataset. Implemented by each individual task."""
        return self._train_instances

    @property
    def dev_instances(self) -> List[BaseInstance] | SingleFileDataset:
        """Dev instances of the dataset. Implemented by each individual task."""
        return self._dev_instances

    @property
    def test_instances(self) -> List[BaseInstance] | SingleFileDataset:
        """Test instances of the dataset. Implemented by each individual task."""
        return self._test_instances

    @staticmethod
    def save_predictions(
            predictions: List[BasePrediction],
            output_path: Path
    ) -> None:
        """Save the predictions to disk."""

        with open(output_path, "w") as f:
            for prediction in predictions:
                f.write(json.dumps(prediction.to_json_dict(), indent=None))
                f.write('\n')

    @staticmethod
    def load_predictions(
            path: Path,
            instances: List[BaseInstance]
    ) -> List[BasePrediction]:
        prediction_dicts = []
        if os.path.isdir(path):
            path = path / (str(path.stem) + '.jsonl')
        with open(path) as f:
            for line in f:
                prediction_dicts.append(json.loads(line))

        if not isinstance(instances, list):
            instances = [
                instance for instance in instances
            ]
        predictions = []
        instance_map = {
            instance.example_id: instance for instance in instances
        }
        for prediction_dict in prediction_dicts:
            predictions.append(BasePrediction.from_json_dict(
                prediction_dict,
                instance_map[prediction_dict['example_id']]
            ))
        # sort predictions according to instance order
        predictions.sort(key=lambda p: instances.index(instance_map[p.example_id]))
        return predictions

    def load_preprocessed_instances(
            self,
            path: Path
    ) -> List[List[BaseInstance]]:
        instances = []
        for split_name in ['train', 'dev', 'test']:
            instances_for_split = []
            with open(path / f'{split_name}.jsonl') as f:
                for line in f.readlines():
                    data = json.loads(line)
                    instance = BaseInstance.from_json_dict(data)
                    instances_for_split.append(instance)
            instances.append(instances_for_split)
        return instances

    def load_instances_from_itg(self) -> List[List[BaseInstance | SingleFileDataset]]:
        raise NotImplementedError


################################################################################
# Model-specific abstract classes for models, datasets, and collate functions
################################################################################



################################################################################
# evaluation
################################################################################


class Partition(enum.Enum):
    """Partition of the dataset."""
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


@dataclasses.dataclass
class Statistics:
    """Statistics object for statistics gathered throughout the evaluation process."""
    model_name: str
    task_name: str
    description: str

    config: BaseConfig

    stats: Dict[str, Any] = dataclasses.field(init=False, default_factory=dict)

    results_by_step: List[Tuple[int, BaseResult]] = dataclasses.field(init=False, default_factory=list)
    best_num_steps: int = dataclasses.field(init=False, default=0)
    test_result: BaseResult = dataclasses.field(init=False)

    total_time: float = dataclasses.field(init=False, default=0.0)

    def to_json_dict(self) -> Dict[str, Any]:
        """Return the statistics object as a JSON-serializable dictionary."""

        # serialize the config

        return {
            "model_name": self.model_name,
            "task_name": self.task_name,
            "description": self.description,
            "config": ConfigContainer.dict_config_to_json_dict(self.config),
            "hash": self.to_hash(),
            "stats": self.stats,
            "results_by_step": [(n, r.to_json_dict()) for n, r in self.results_by_step],
            "best_num_steps": self.best_num_steps,
            "test_result": self.test_result,
            "total_time": self.total_time
        }

    def to_hash(self) -> str:
        h = hashlib.sha256(bytes(f"{self.model_name}{self.task_name}"
                                 f"{self.description}{dict(self.config)}", "utf-8")).hexdigest()
        return f"{h[:4]}-{h[4:8]}"


class CustomDataset(Dataset):
    """Simple generic PyTorch dataset implementation."""
    instances: Union[List[BaseInstance], SingleFileDataset]

    def __init__(self, instances: List[BaseInstance] | SingleFileDataset) -> None:
        super(CustomDataset, self).__init__()
        self.instances = instances

    def __getitem__(self, item: int | slice) -> BaseInstance:
        """Return the instances at the given indices."""
        return self.instances[item]

    def __len__(self) -> int:
        return len(self.instances)

    def get_subset(self, indices: slice) -> CustomDataset:
        """
        Load the files at the specified index and return the corresponding
        instances.
        """
        if isinstance(self.instances, list):
            return CustomDataset(self.instances[indices])
        elif isinstance(self.instances, SingleFileDataset):
            instances = self.instances.get_subset(indices)
            return CustomDataset(instances)
        else:
            raise NotImplementedError(
                f'item should be int or slice but got {type(indices)}'
            )

    def get_examples_from_ids(self, ids: List[str]) -> List[BaseInstance]:
        """
        Given a list of string ids, return the instances whose filenames contain
        the specified ids.
        """
        if isinstance(self.instances, list):
            found_instances = []
            for instance in self.instances:
                if instance.example_id in ids:
                    found_instances.append(instance)
        elif isinstance(self.instances, SingleFileDataset):
            found_instances = self.instances.get_examples_from_ids(ids)
        else:
            raise ValueError(
                f'self.instances should be of type List[BaseInstance] or '
                f'SingleFileDataset, but is {type(self.instances)}'
            )
        return found_instances


class SingleFileDataset(Dataset):
    """
    Dataset that loads instances from single ITG files. This can be more
    convenient for large datasets, where loading everything into memory takes time

    :param dir_path: Directory with files to load
    :param valid_filenames: Optional list of valid filenames, if given,
        only these files are loaded
    """
    dir_path: Path
    valid_filenames: List = None

    def __init__(
            self,
            dir_path: Path,
            valid_filenames: List = None,
    ) -> None:
        super(SingleFileDataset, self).__init__()
        self.dir_path = dir_path

        if valid_filenames is None or len(valid_filenames) == 0:
            # If valid_filenames is not given, get the complete list of filenames
            # from the directory
            self.filenames = os.listdir(self.dir_path)
        else:
            self.filenames = valid_filenames

    def __getitem__(self, item: int | slice) -> Union[BaseInstance, List[BaseInstance]]:
        """
        Load the files at the specified index and return the corresponding
        instances.
        """
        if isinstance(item, int):
            with open(self.dir_path / self.filenames[item], 'r') as file:
                document = IntertextDocument.load_json(file)
                instance = self._create_instance_from_document(document)
                return instance

        elif isinstance(item, slice):
            instances = []
            for filename in self.filenames[item]:
                with open(self.dir_path / filename, 'r') as file:
                    document = IntertextDocument.load_json(file)
                    instance = self._create_instance_from_document(document)
                    instances.append(instance)
            return instances

        else:
            raise NotImplementedError(
                f'item should be int or slice but got {type(item)}'
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def get_subset(self, items: slice) -> SingleFileDataset:
        """
        Returns a subset of the current dataset with the specified indices
        :param items: indices of items in dataset that should form the new dataset
        :return:
        """
        if isinstance(self, SingleFileDataset):
            filenames = self.filenames[items]
            return self.__class__(
                self.dir_path,
                filenames
            )
        else:
            raise NotImplementedError

    def get_examples_from_ids(self, ids: List[str]):
        """
        Given a list of string ids, return the instances whose filenames contain
        the specified ids.
        """
        found_instances = []
        for filename in self.filenames:
            for id_ in ids:
                if id_ in filename:
                    with open(self.dir_path / filename, 'r') as file:
                        document = IntertextDocument.load_json(file)
                    instance = self._create_instance_from_document(document)
                    found_instances.append(instance)
        return found_instances

    @abc.abstractmethod
    def _create_instance_from_document(
            self,
            document: IntertextDocument
    ) -> BaseInstance:
        """Given a document, return an instance of the respective task."""
        raise NotImplementedError
