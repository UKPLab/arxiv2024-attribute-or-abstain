import logging
import random
import time
from io import StringIO
from pathlib import Path
from typing import List, Dict, Any, ClassVar, Tuple

import nltk
from intertext_graph.itgraph import IntertextDocument, Node

from evaluation.common import BaseInstance, BaseTask, Statistics, SingleFileDataset
from config_lib.base_config import BaseConfig

"""
This is based on the code from the relpos_graph repository
"""

logger = logging.getLogger(__name__)


class QASATask(BaseTask):
    """Implements instance loading and evaluation for QASA"""
    task_name: ClassVar[str] = "qasa"

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
        super(QASATask, self).__init__(config, stats)

    def load_instances_from_itg(self) -> List[List[BaseInstance | SingleFileDataset]]:
        logger.info("Load the QASA-ITG dataset.")
        tick = time.time()

        data_path = Path(self.config.location.datasets) / "qasa" / "QASA-ITG" 

        # Load the dev itg files
        dev_path = data_path / "qasa-dev-split.jsonl"
        dev_documents = []
        with open(dev_path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    dev_documents.append(IntertextDocument.load_json(f))
        # Convert citation identifiers in document texts as these will lead to problems
        # When doing attribution
        for doc in dev_documents:
            self.replace_square_brackets_in_doc(doc)

        # Load the test itg files
        test_path = data_path / "qasa-test-split.jsonl"
        test_documents = []
        with open(test_path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    test_documents.append(IntertextDocument.load_json(f))
        # Convert citation identifiers in document texts as these will lead to problems
        # When doing attribution
        for doc in test_documents:
            self.replace_square_brackets_in_doc(doc)

        self.stats.stats["task-initialization.num-test-documents"] = len(test_documents)
        train_documents = []
        tack = time.time()
        logger.info(f"Loaded {len(train_documents)} train documents, "
                    f"{len(dev_documents)} dev documents, and "
                    f"{len(test_documents)} test documents in {tack - tick:0.4f}s.")

        logger.info("Create the QASA instances.")
        tick = time.time()

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

        # Get a few test instances and put them into the train set to be used as
        # examples
        train_instances = [
            instance for instance in test_instances
            if instance.example_id in self.config.task.qasa_example_ids_in_test_set
        ]
        test_instances = [
            instance for instance in test_instances
            if not instance in train_instances
        ]
        assert len(train_instances) == len(self.config.task.qasa_example_ids_in_test_set)

        # shuffle
        random.seed(self.config.random_seed)
        test_instances = random.sample(test_instances, len(test_instances))

        tack = time.time()
        logger.info(f"Created {len(train_instances)} train instances, "
                    f"{len(dev_instances)} dev instances, and "
                    f"{len(test_instances)} test instances in {tack - tick:0.4f}s.")

        logger.info("Gather evidence statistics.")
        tick = time.time()
        self.stats.stats["task-initialization.evidence-statistics"] = {0: 0, 1: 0}
        # Get the number of evidence and non-evidence nodes in the dataset
        for instance in test_instances:
            num_nodes = len(instance.document.nodes)
            num_evidence_nodes = len(instance.extraction_nodes[0])
            self.stats.stats["task-initialization.evidence-statistics"][0] += num_nodes - num_evidence_nodes
            self.stats.stats["task-initialization.evidence-statistics"][1] += num_evidence_nodes

        logger.info(f"Evidence statistics: {self.stats.stats['task-initialization.evidence-statistics']}")

        tack = time.time()
        logger.info(f"Gathered evidence statistics in {tack - tick:0.4f}s.")

        return [train_instances, dev_instances, test_instances]

    @staticmethod
    def _find_evidence_nodes(
            document: IntertextDocument,
            que_id: int
    ) -> List[Node]:
        """Get all true evidence nodes for a given document and annotation.
        We only consider nodes of type 'p'."""
        evidence_nodes = []
        for node in document.nodes:
            if node.ntype == 'p':
                for evidence_for in node.meta["is_evidence_for"]:
                    if evidence_for['question_id'] == que_id:
                        evidence_nodes.append(node)
                        break
        return evidence_nodes

    def replace_square_brackets_in_doc(self, doc: IntertextDocument):
        for n in doc.nodes:
            n.content = self.replace_square_brackets(n.content)

    @staticmethod
    def replace_square_brackets(text):
        text = text.replace('[', '(')
        text = text.replace(']', ')')
        return text

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

    @staticmethod
    def _make_prompt(
            question: str
    ) -> str:
        """Return question as the basic prompt."""
        return f"{question}"

    def _create_eval_instances_from_document(
            self,
            doc: IntertextDocument
    ) -> List[BaseInstance]:
        instances = []
        # create one instance per question
        paper_id = doc.meta["paper_id"]
        annotations = doc.meta["annotations"]
        for annotation_dict in annotations:
            question_id = annotation_dict['question_id']
            evi_info = annotation_dict['evidential_info']
            if annotation_dict['composition'] == 'unanswerable':
                composition = ['unanswerable']
                question_type = 'unanswerable'
                evidence_nodes = [[]]
            else:
                composition = annotation_dict['composition']
                if composition.startswith('['):
                    composition = composition[1:-1]
                # Replace square brackets in gold answer
                composition = self.replace_square_brackets(composition)
                # Split into sentences
                composition = nltk.sent_tokenize(composition)
                question_type = annotation_dict['question_type'] 
                evidence_nodes = self._find_evidence_nodes(doc, question_id)
                if len(evidence_nodes) != len(evi_info):
                    # Skip instances where we were not able to match evidence
                    # to the paper text
                    continue
                # The dataset was only annotated with one set of evidence nodes
                # for the whole question, so we copy the evidence nodes for each
                # sentence
                evidence_nodes = [evidence_nodes for _ in composition]
            prompt = self._make_prompt(annotation_dict["question"])
            evidence_candidates = self._find_evidence_candidates(doc)
            example_id = paper_id + '_' + str(question_id)
            instance = BaseInstance(
                task_name=self.task_name,
                example_id=example_id,
                document=doc,
                prompt=prompt,
                question=annotation_dict['question'],
                statement='',
                free_text_answer=[composition],
                answer_type=[question_type],
                extraction_nodes=[evidence_nodes],
                extraction_candidates=evidence_candidates,
                extraction_level='paragraph',
                answer_has_multiple_statements=True
            )
            instances.append(instance)

        return instances
