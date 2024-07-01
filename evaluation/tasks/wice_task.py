import time
from io import StringIO
from pathlib import Path
from typing import ClassVar, List, Dict, Any
import logging

from intertext_graph import IntertextDocument, Node

from config_lib.base_config import BaseConfig
from evaluation.common import BaseTask, BaseInstance, Statistics, SingleFileDataset

logger = logging.getLogger(__name__)

LABEL_MAP = {
    'supported': 'supported',
    'partially_supported': 'partially supported',
    'not_supported': 'not supported'
}

class WiceTask(BaseTask):
    """Implements instance loading and evaluation for QASPER"""
    task_name: ClassVar[str] = "wice"

    _train_instances: List[BaseInstance]
    _dev_instances: List[BaseInstance]
    _test_instances: List[BaseInstance]

    def __init__(
            self,
            config: BaseConfig,
            stats: Statistics
    ) -> None:
        super().__init__(config, stats)

    def load_instances_from_itg(self) -> List[List[BaseInstance | SingleFileDataset]]:
        logger.info("Load the WiCE-ITG dataset.")
        tick = time.time()

        # There is both a shallow and a deep version of the dataset in ITG format
        claim_or_subclaim = self.config.task.wice_claim_or_subclaim

        data_path = Path(self.config.location.datasets) / "WiCE" / "wice_itg"

        # Load the train itg files
        train_path = data_path / f"{claim_or_subclaim}-train.jsonl"
        train_documents = []
        with open(train_path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    train_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-train-documents"] = len(train_documents)

        # Load the dev itg files
        dev_path = data_path / f"{claim_or_subclaim}-dev.jsonl"
        dev_documents = []
        with open(dev_path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    dev_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-dev-documents"] = len(dev_documents)

        # Load the test itg files
        test_path = data_path / f"{claim_or_subclaim}-test.jsonl"
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

        logger.info("Create the WiCE instances.")
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

    @staticmethod
    def _find_extraction_nodes(
            document: IntertextDocument,
            annotation_idx: int
    ) -> List[Node]:
        """Get all true evidence nodes for a given document and annotation.
        We only consider nodes of type 'p'."""
        evidence_nodes = []
        for node in document.nodes:
            if node.ntype == 'p':
                for evidence_for in node.meta["is_evidence_for"]:
                    if evidence_for["annotation_idx"] == annotation_idx:
                        evidence_nodes.append(node)
                        break
        return evidence_nodes

    @staticmethod
    def _find_extraction_candidates(
            document: IntertextDocument
    ) -> List[Node]:
        """Get the evidence candidates as all nodes of node type 'p'."""
        evidence_candidates = []
        for node in document.nodes:
            if node.ntype == "p":
                evidence_candidates.append(node)
        return evidence_candidates

    def _make_additional_info(
            self,
            claim_context: str
    ) -> str:
        additional_info = f'The claim was made in the following context: "{claim_context}"'
        return additional_info

    def _create_train_instances_from_document(
            self,
            doc: IntertextDocument
    ) -> List[BaseInstance]:
        instances = []

        example_id = doc.meta['id']
        prompt = ''
        question = ''
        statement = doc.meta['claim']
        extraction_candidates = self._find_extraction_candidates(doc)
        free_text_answer = LABEL_MAP[doc.meta['label']]
        answer_type = free_text_answer
        additional_info = self._make_additional_info(doc.meta['claim_context'])

        # Create multiple instances per claim (one per evidence set)
        for i, idx_list in enumerate(doc.meta['supporting_sentences']):
            extraction_nodes = self._find_extraction_nodes(
                doc,
                i
            )
            instance = BaseInstance(
                task_name=self.task_name,
                example_id=example_id,
                document=doc,
                prompt=prompt,
                question=question,
                statement=statement,
                free_text_answer=[free_text_answer],
                answer_type=[answer_type],
                extraction_nodes=[extraction_nodes],
                extraction_candidates=extraction_candidates,
                extraction_level='paragraph',
                additional_info=additional_info
            )
            instances.append(instance)

        return instances

    def _create_eval_instances_from_document(
            self,
            doc: IntertextDocument
    ) -> List[BaseInstance]:
        instances = []

        example_id = doc.meta['id']
        prompt = ''
        question = ''
        statement = doc.meta['claim']
        extraction_candidates = self._find_extraction_candidates(doc)
        free_text_answer = []
        answer_type = []
        additional_info = self._make_additional_info(doc.meta['claim_context'])
        extraction_nodes = []

        # Create multiple instances per claim (one per evidence set)
        for i, idx_list in enumerate(doc.meta['supporting_sentences']):
            free_text_answer_for_annotation = LABEL_MAP[doc.meta['label']]
            answer_type_for_annotation = free_text_answer_for_annotation
            extraction_nodes_for_annotation = self._find_extraction_nodes(
                doc,
                i
            )
            free_text_answer.append(free_text_answer_for_annotation)
            answer_type.append(answer_type_for_annotation)
            extraction_nodes.append(extraction_nodes_for_annotation)

        instance = BaseInstance(
            task_name=self.task_name,
            example_id=example_id,
            document=doc,
            prompt=prompt,
            question=question,
            statement=statement,
            free_text_answer=free_text_answer,
            answer_type=answer_type,
            extraction_nodes=extraction_nodes,
            extraction_candidates=extraction_candidates,
            extraction_level='paragraph',
            additional_info=additional_info
        )
        instances.append(instance)

        return instances