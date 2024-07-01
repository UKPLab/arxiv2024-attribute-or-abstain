import logging
import time
from io import StringIO
from pathlib import Path
from typing import List, Dict, Any, ClassVar, Tuple

from intertext_graph.itgraph import IntertextDocument, Node

from evaluation.common import BaseInstance, BaseTask, Statistics, SingleFileDataset
from config_lib.base_config import BaseConfig

"""
This is based on the code from the relpos_graph repository
"""

logger = logging.getLogger(__name__)


class QASPERTask(BaseTask):
    """Implements instance loading and evaluation for QASPER"""
    task_name: ClassVar[str] = "qasper"

    train_documents: List[IntertextDocument]
    dev_documents: List[IntertextDocument]
    test_documents: List[IntertextDocument]

    _train_instances: List[BaseInstance]
    _dev_instances: List[BaseInstance]
    _test_instances: List[BaseInstance]

    def __init__(
            self,
            config: BaseConfig,
            stats: Statistics
    ) -> None:
        super(QASPERTask, self).__init__(config, stats)

    def load_instances_from_itg(self) -> List[List[BaseInstance | SingleFileDataset]]:
        logger.info("Load the QASPER-ITG dataset.")
        tick = time.time()

        # There is both a shallow and a deep version of the dataset in ITG format
        deep_or_shallow = self.config.task.deep_or_shallow

        data_path = Path(self.config.location.datasets) / "qasper" / "QASPER-ITG"

        # Load the train itg files
        train_path = data_path / f"{deep_or_shallow}-train.jsonl"
        train_documents = []
        with open(train_path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    train_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-train-documents"] = len(train_documents)

        # Load the dev itg files
        dev_path = data_path / f"{deep_or_shallow}-dev.jsonl"
        dev_documents = []
        with open(dev_path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    dev_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-dev-documents"] = len(dev_documents)

        # Load the test itg files
        test_path = data_path / f"{deep_or_shallow}-test.jsonl"
        test_documents = []
        with open(test_path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    test_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-test-documents"] = len(test_documents)

        tack = time.time()
        logger.info(f"Loaded {len(train_documents)} train documents, "
                    f"{len(dev_documents)} dev documents, and "
                    f"{len(test_documents)} test documents in {tack - tick:0.4f}s.")

        logger.info("Create the QASPER instances.")
        tick = time.time()

        # Create BaseInstance objects from the loaded documents
        # Create train instances
        train_instances = []
        for document in train_documents:
            train_instances += self._create_train_instances_from_document(document)
        self.stats.stats["task-initialization.num-train-instances"] = len(train_instances)

        # Create dev instances
        dev_instances = []
        for document in dev_documents:
            dev_instances += self._create_eval_instances_from_document(document)
        self.stats.stats["task-initialization.num-dev-instances"] = len(dev_instances)

        # Create test instances
        test_instances = []
        for document in test_documents:
            test_instances += self._create_eval_instances_from_document(document)
        self.stats.stats["task-initialization.num-test-instances"] = len(test_instances)

        tack = time.time()
        logger.info(f"Created {len(train_instances)} train instances, "
                    f"{len(dev_instances)} dev instances, and "
                    f"{len(test_instances)} test instances in {tack - tick:0.4f}s.")

        logger.info("Gather evidence statistics.")
        tick = time.time()
        self.stats.stats["task-initialization.evidence-statistics"] = {0: 0, 1: 0}
        # Get the number of evidence and non-evidence nodes in the dataset
        for instance in train_instances:
            num_nodes = len(instance.document.nodes)
            num_evidence_nodes = len(instance.extraction_nodes[0])
            self.stats.stats["task-initialization.evidence-statistics"][0] += num_nodes - num_evidence_nodes
            self.stats.stats["task-initialization.evidence-statistics"][1] += num_evidence_nodes

        logger.info(f"Evidence statistics: {self.stats.stats['task-initialization.evidence-statistics']}")

        tack = time.time()
        logger.info(f"Gathered evidence statistics in {tack - tick:0.4f}s.")

        return [train_instances, dev_instances, test_instances]

    def _find_evidence_nodes(
            self,
            document: IntertextDocument,
            answer: Dict[str, Any]
    ) -> List[Node]:
        """Get all true evidence nodes for a given document and annotation.
        We only consider nodes of type 'p'."""
        evidence_nodes = []
        for node in document.nodes:
            if node.ntype == 'p':
                for evidence_for in node.meta["is_evidence_for"]:
                    if evidence_for["annotation_id"] == answer["annotation_id"]:
                        evidence_nodes.append(node)
                        break
        return evidence_nodes

    @staticmethod
    def _find_evidence_candidates(
            document: IntertextDocument
    ) -> List[Node]:
        """Get the evidence candidates as all nodes of node type 'p'."""
        evidence_candidates = []
        for node in document.nodes:
            if node.ntype == "p":
                evidence_candidates.append(node)
        return evidence_candidates

    def _get_answer_text_and_type(self, answer: Dict[str, Any]) -> Tuple[str, str]:
        # this code is adapted from QASPER's get_answers_and_evidence method
        if answer["answer"]["unanswerable"]:
            return "Unanswerable", "unanswerable"
        elif answer["answer"]["extractive_spans"]:
            return ", ".join(answer["answer"]["extractive_spans"]), 'extractive'
        elif answer["answer"]["free_form_answer"]:
            return answer["answer"]["free_form_answer"], 'abstractive'
        elif answer["answer"]["yes_no"]:
            return "Yes", "boolean"
        elif answer["answer"]["yes_no"] is not None:
            return "No", "boolean"
        else:
            raise RuntimeError(f"Annotation {answer['answer']['annotation_id']} does not contain an answer")

    @staticmethod
    def _make_prompt(
            question: str
    ) -> str:
        """Return question as the basic prompt."""
        return f"{question}"

    def _create_train_instances_from_document(
            self,
            doc: IntertextDocument
    ) -> List[BaseInstance]:
        instances = []

        # create multiple instances per question (one for each answer)
        for qas in doc.meta["qas"]:
            for answer in qas["answers"]:
                answer_text, answer_type = self._get_answer_text_and_type(answer)
                evidence_nodes = self._find_evidence_nodes(doc, answer)
                evidence_candidates = self._find_evidence_candidates(doc)
                prompt = self._make_prompt(qas["question"])

                instance = BaseInstance(
                    task_name=self.task_name,
                    example_id=qas["question_id"],
                    document=doc,
                    prompt=prompt,
                    question=qas['question'],
                    statement='',
                    free_text_answer=[answer_text],
                    answer_type=[answer_type],
                    extraction_nodes=[evidence_nodes],
                    extraction_candidates=evidence_candidates,
                    extraction_level='paragraph',
                )
                instances.append(instance)

        return instances

    def _create_eval_instances_from_document(
            self,
            doc: IntertextDocument
    ) -> List[BaseInstance]:
        instances = []

        # create one instance per question (with all answers)
        for qas in doc.meta["qas"]:
            answer_texts = []
            answer_types = []
            for answer in qas["answers"]:
                answer_text, answer_type = self._get_answer_text_and_type(answer)
                answer_texts.append(answer_text)
                answer_types.append(answer_type)

            prompt = self._make_prompt(qas["question"])
            evidence_nodes = [self._find_evidence_nodes(doc, a) for a in qas["answers"]]
            evidence_candidates = self._find_evidence_candidates(doc)

            instance = BaseInstance(
                task_name=self.task_name,
                example_id=qas["question_id"],
                document=doc,
                prompt=prompt,
                question=qas['question'],
                statement='',
                free_text_answer=answer_texts,
                answer_type=answer_types,
                extraction_nodes=evidence_nodes,
                extraction_candidates=evidence_candidates,
                extraction_level='paragraph',
            )
            instances.append(instance)

        return instances
