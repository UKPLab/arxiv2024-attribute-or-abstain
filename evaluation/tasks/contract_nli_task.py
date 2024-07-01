import time
from io import StringIO
from pathlib import Path
from typing import ClassVar, List, Dict, Any
import logging
import json
import os
from intertext_graph import IntertextDocument, Node, SpanNode

from config_lib.base_config import BaseConfig
from evaluation.common import BaseTask, BaseInstance, Statistics, SingleFileDataset

logger = logging.getLogger(__name__)

LABEL_MAP = {
    "Entailment": "entailment",
    "Contradiction": "contradiction",
    "NotMentioned": "not mentioned"
}

class ContractNLITask(BaseTask):
    """Implements instance loading and evaluation for CONTRACT"""
    task_name: ClassVar[str] = "contract_nli"

    train_labels: Dict
    dev_labels: Dict
    test_labels: Dict

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

        logger.info("Load the Contract-ITG dataset.")
        tick = time.time()
        train_file = self.config.location.datasets / 'contract_nli' / 'train.json'
        dev_file = self.config.location.datasets / 'contract_nli' / 'dev.json'
        test_file = self.config.location.datasets / 'contract_nli' / 'test.json'

        with open(train_file, "r", encoding="utf-8") as file:
            train_data = json.load(file)
            self.train_labels = train_data['labels']

        with open(dev_file, "r", encoding="utf-8") as file:
            dev_data = json.load(file)
            self.dev_labels = dev_data['labels']

        with open(test_file, "r", encoding="utf-8") as file:
            test_data = json.load(file)
            self.test_labels = test_data['labels']

        data_path = Path(self.config.location.datasets) / "contract_nli" / "contract_nli_itg"

        # Load the train itg files
        train_path = data_path / "contract-train-nl-latest.jsonl"
        train_documents = []
        with open(train_path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    train_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-train-documents"] = len(train_documents)

        # Load the dev itg files
        dev_path = data_path / f"contract-dev-nl-latest.jsonl"
        dev_documents = []
        with open(dev_path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    dev_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-dev-documents"] = len(dev_documents)

        # Load the test itg files
        test_path = data_path / f"contract-test-nl-latest.jsonl"
        test_documents = []
        with open(test_path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    test_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-test-documents"] = len(test_documents)

        for document in train_documents:
            for tmp_node in document.nodes:
                if isinstance(tmp_node, SpanNode):
                    document.remove_node(tmp_node)

        for document in dev_documents:
            for tmp_node in document.nodes:
                if isinstance(tmp_node, SpanNode):
                    document.remove_node(tmp_node)

        for document in test_documents:
            for tmp_node in document.nodes:
                if isinstance(tmp_node, SpanNode):
                    document.remove_node(tmp_node)

        tack = time.time()
        logger.info(f"Loaded {len(train_documents)} train documents, "
                    f"{len(dev_documents)} dev documents, and "
                    f"{len(test_documents)} test documents in {tack - tick:0.4f}s.")

        logger.info("Create the Contract instances.")
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
            test_instances += self._create_test_instances_from_document(document)
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

    def _find_extraction_nodes(
            self,
            document: IntertextDocument,
            nda_key: str
    ) -> List[Node]:
        """Get all true evidence nodes for a given document and annotation.
        We only consider nodes of type 'p'."""
        evidence_nodes = []
        for node in document.nodes:
            if node.ntype == 'p' and node.meta is not None:
                for evidence_for in node.meta["is_evidence_for"]:
                    if evidence_for["nda"] == nda_key:
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
        additional_info = f'The claim was made in the following context: {claim_context}'
        return additional_info

    def _create_train_instances_from_document(
            self,
            doc: IntertextDocument
    ) -> List[BaseInstance]:
        instances = []
        example_id = doc.nodes[0].ix.split('_')[0]
        prompt = ''
        question = "What is the relation between the contract and the statement? Choose between 'entailment', 'contradiction' and 'not mentioned'"
        extraction_candidates = self._find_extraction_candidates(doc)
        answer_type = "None"

        nda_dict = doc.meta['annotation_sets'][0]['annotations']
        for nda_key, nda_value in nda_dict.items():
            statement = self.train_labels[nda_key]['hypothesis']
            free_text_answer = LABEL_MAP[nda_value['choice']]
            extraction_nodes = self._find_extraction_nodes(
                doc,
                nda_key
            )
            example_id_instance = example_id + '_' + nda_key
            instance = BaseInstance(
                task_name=self.task_name,
                example_id=example_id_instance,
                document=doc,
                prompt=prompt,
                question=question,
                statement=statement,
                free_text_answer=[free_text_answer],
                answer_type=[answer_type],
                extraction_nodes=[extraction_nodes],
                extraction_candidates=extraction_candidates,
                extraction_level='paragraph'
            )
            instances.append(instance)

        return instances

    def _create_eval_instances_from_document(
            self,
            doc: IntertextDocument
    ) -> List[BaseInstance]:
        instances = []
        example_id = doc.nodes[0].ix.split('_')[0]
        prompt = ''
        question = "What is the relation between the contract and the statement? Choose between 'entailment', 'contradiction' and 'not mentioned'"
        extraction_candidates = self._find_extraction_candidates(doc)
        nda_dict = doc.meta['annotation_sets'][0]['annotations']

        for nda_key, nda_value in nda_dict.items():
            statement = self.dev_labels[nda_key]['hypothesis']
            free_text_answer = LABEL_MAP[nda_value['choice']]
            answer_type = free_text_answer
            extraction_nodes = self._find_extraction_nodes(
                                    doc,
                                    nda_key
                                    )
            example_id_instance = example_id + '_' + nda_key
            instance = BaseInstance(
                                    task_name=self.task_name,
                                    example_id=example_id_instance,
                                    document=doc,
                                    prompt=prompt,
                                    question=question,
                                    statement=statement,
                                    free_text_answer=[free_text_answer],                                            
                                    answer_type=[answer_type],
                                    extraction_nodes=[extraction_nodes],
                                    extraction_candidates=extraction_candidates,
                                    extraction_level='paragraph',
                                    )
            instances.append(instance)

        return instances

    def _create_test_instances_from_document(
            self,
            doc: IntertextDocument
    ) -> List[BaseInstance]:
        instances = []
        example_id = doc.nodes[0].ix.split('_')[0]
        prompt = ''
        question = "What is the relation between the contract and the statement? Choose between 'entailment', 'contradiction' and 'not mentioned'"
        extraction_candidates = self._find_extraction_candidates(doc)
        answer_type = "None"
        nda_dict = doc.meta['annotation_sets'][0]['annotations']

        for nda_key, nda_value in nda_dict.items():
            statement = self.test_labels[nda_key]['hypothesis']
            free_text_answer = LABEL_MAP[nda_value['choice']]
            extraction_nodes = self._find_extraction_nodes(
                    doc,
                    nda_key
                    )
            example_id_instance = example_id + '_' + nda_key
            instance = BaseInstance(
                                    task_name=self.task_name,
                                    example_id=example_id_instance,
                                    document=doc,
                                    prompt=prompt,
                                    question=question,
                                    statement=statement,
                                    free_text_answer=[free_text_answer],
                                    answer_type=[answer_type],
                                    extraction_nodes=[extraction_nodes],
                                    extraction_candidates=extraction_candidates,
                                    extraction_level='paragraph',)
            instances.append(instance)

        return instances
