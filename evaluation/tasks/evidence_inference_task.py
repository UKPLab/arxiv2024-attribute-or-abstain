import dataclasses
import json
import logging
import time
from io import StringIO
from pathlib import Path
from typing import List, Dict, Any, ClassVar

from intertext_graph.itgraph import IntertextDocument, Node

from evaluation.common import BaseInstance, BasePrediction, BaseResult, BaseTask, Statistics, Partition, \
    SingleFileDataset
from config_lib.base_config import BaseConfig
from evaluation.util import get_unparsable_proportion

"""
This is based on the code from the relpos_graph repository
"""

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EvidenceInferenceInstance(BaseInstance):
    """
    An instance for the Evidence Inference task.
    This inherits from BaseInstance and adds task-specific fields
    for convenience. They are only used internally in the task and
    should not be used in the model.
    """
    pmc_id: str = None
    prompt_id: str = None

    # input:
    outcome: str = None
    intervention: str = None
    comparator: str = None

    # output:
    labels: List[str] = None
    label_codes: List[int] = None


class EvidenceInferenceTask(BaseTask):
    """Implements data loading and evaluation for Evidence Inference."""
    task_name: ClassVar[str] = "evidence_inference"

    _train_instances: List[EvidenceInferenceInstance]
    _dev_instances: List[EvidenceInferenceInstance]
    _test_instances: List[EvidenceInferenceInstance]

    def __init__(
            self,
            config: BaseConfig,
            stats: Statistics
    ) -> None:
        super(EvidenceInferenceTask, self).__init__(config, stats)

    def load_instances_from_itg(self) -> List[List[BaseInstance | SingleFileDataset]]:
        logger.info("Load the (ITG-formatted) Evidence Inference dataset.")
        tick = time.time()

        # There is both a shallow and a deep version of the dataset in ITG format
        deep_or_shallow = self.config.task.deep_or_shallow
        self.stats.stats["task-initialization.deep-or-shallow"] = deep_or_shallow

        # Load the train ITG files
        train_documents = []
        path = (
                Path(self.config.location.datasets)
                / 'evidence_inference'
                / 'evidence_inference_itg'
                / f"{deep_or_shallow}-train.jsonl"
        )
        with open(path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    train_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-train-documents"] = len(train_documents)

        # Load the dev ITG files
        dev_documents = []
        path = (
            Path(self.config.location.datasets)
            / 'evidence_inference'
            / 'evidence_inference_itg'
            / f"{deep_or_shallow}-dev.jsonl"
        )
        with open(path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    dev_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-dev-documents"] = len(dev_documents)

        # Load the test ITG files
        test_documents = []
        path = (
                Path(self.config.location.datasets)
                / 'evidence_inference'
                / 'evidence_inference_itg'
                / f"{deep_or_shallow}-test.jsonl"
        )
        with open(path, "r", encoding="utf-8") as file:
            for document_json_str in file:
                with StringIO(document_json_str) as f:
                    test_documents.append(IntertextDocument.load_json(f))
        self.stats.stats["task-initialization.num-test-documents"] = len(test_documents)

        tack = time.time()
        logger.info(f"Loaded {len(train_documents)} train documents, "
                    f"{len(dev_documents)} dev documents, and "
                    f"{len(test_documents)} test documents in {tack - tick:0.4f}s.")

        logger.info("Create the Evidence Inference instances.")
        tick = time.time()

        self.stats.stats["task-initialization.prompt-has-no-annotations"] = 0
        self.stats.stats["task-initialization.annotation-has-empty-evidence-text"] = 0
        self.stats.stats["task-initialization.annotation-has-no-evidence-node"] = 0
        self.stats.stats["task-initialization.annotation-has-more-than-one-evidence-node"] = 0

        # Convert ITGs to instances
        train_instances = []
        for document in train_documents:
            train_instances += self._create_train_instances_from_document(document)
        self.stats.stats["task-initialization.num-train-instances"] = len(train_instances)

        dev_instances = []
        for document in dev_documents:
            dev_instances += self._create_eval_instances_from_document(document)
        self.stats.stats["task-initialization.num-dev-instances"] = len(dev_instances)

        test_instances = []
        for document in test_documents:
            test_instances += self._create_eval_instances_from_document(document)
        self.stats.stats["task-initialization.num-test-instances"] = len(test_instances)

        tack = time.time()
        logger.info(f"Created {len(train_instances)} train instances, "
                    f"{len(dev_instances)} dev instances, and "
                    f"{len(test_instances)} test instances in {tack - tick:0.4f}s.")

        logger.info("Gather label and evidence statistics.")
        tick = time.time()
        self.stats.stats["task-initialization.label-statistics"] = {-1: 0, 0: 0, 1: 0}
        self.stats.stats["task-initialization.evidence-statistics"] = {0: 0, 1: 0}

        for instance in train_instances:
            self.stats.stats["task-initialization.label-statistics"][instance.label_codes[0]] += 1
            num_nodes = len(instance.document.nodes)
            num_evidence_nodes = len(instance.extraction_nodes[0])
            self.stats.stats["task-initialization.evidence-statistics"][0] += num_nodes - num_evidence_nodes
            self.stats.stats["task-initialization.evidence-statistics"][1] += num_evidence_nodes

        logger.info(f"Label statistics: {self.stats.stats['task-initialization.label-statistics']}")
        logger.info(f"Evidence statistics: {self.stats.stats['task-initialization.evidence-statistics']}")

        tack = time.time()
        logger.info(f"Gathered label and evidence statistics in {tack - tick:0.4f}s.")

        return [train_instances, dev_instances, test_instances]

    def _find_evidence_nodes(self, document: IntertextDocument, annotation: Dict[str, Any]) -> List[Node]:
        """
        Find the ground truth evidence nodes for an annotation.
        Only nodes of ntype 'p' can be evidence nodes.
        """
        evidence_nodes = []
        if annotation["evidence_text"] == "":
            self.stats.stats["task-initialization.annotation-has-empty-evidence-text"] += 1
            return evidence_nodes
        for node in document.nodes:
            if node.ntype == 'p':
                for evidence_for in node.meta["is_evidence_for"]:
                    if (
                            evidence_for["prompt_id"] == annotation["prompt_id"]
                            and evidence_for["user_id"] == annotation["user_id"]
                    ):
                        evidence_nodes.append(node)
                        break
        if len(evidence_nodes) == 0:
            self.stats.stats["task-initialization.annotation-has-no-evidence-node"] += 1
        elif len(evidence_nodes) > 1:
            self.stats.stats["task-initialization.annotation-has-more-than-one-evidence-node"] += 1
        return evidence_nodes

    @staticmethod
    def _find_evidence_candidates(
            document: IntertextDocument,
    ) -> List[Node]:
        """Get evidence candidates as all nodes of type 'p'."""
        evidence_candidates = []
        for node in document.nodes:
            if node.ntype == "p":
                evidence_candidates.append(node)
        return evidence_candidates

    def _create_train_instances_from_document(self, doc: IntertextDocument) -> List[EvidenceInferenceInstance]:
        instances = []
        extraction_candidates = self._find_evidence_candidates(doc)

        # create multiple instances per prompt (one for each annotation)
        for prompt_id, prompt in doc.meta["prompts"].items():
            if prompt_id not in doc.meta["annotations"].keys():
                self.stats.stats["task-initialization.prompt-has-no-annotations"] += 1
                continue

            for annotation in doc.meta["annotations"][prompt_id]:
                evidence_nodes = self._find_evidence_nodes(doc, annotation)
                verbalized_prompt = self._create_prompt(prompt["outcome"], prompt["intervention"], prompt["comparator"])
                instance = EvidenceInferenceInstance(
                    task_name=self.task_name,
                    example_id=f"prompt_{annotation['prompt_id']}_user_{annotation['user_id']}",
                    statement='',
                    question=verbalized_prompt,
                    prompt=verbalized_prompt,
                    extraction_level='p',
                    extraction_candidates=extraction_candidates,
                    free_text_answer=[annotation["label"]],
                    answer_type=['None'],
                    extraction_nodes=[evidence_nodes],
                    pmc_id=doc.meta["pmc_id"],
                    prompt_id=prompt_id,
                    document=doc,
                    outcome=prompt["outcome"],
                    intervention=prompt["intervention"],
                    comparator=prompt["comparator"],
                    labels=[annotation["label"]],
                    label_codes=[annotation["label_code"]],
                )
                instances.append(instance)

        return instances

    def _create_eval_instances_from_document(self, doc: IntertextDocument) -> List[EvidenceInferenceInstance]:
        instances = []

        extraction_candidates = self._find_evidence_candidates(doc)

        # create one instance per prompt (with all annotations)
        for prompt_id, prompt in doc.meta["prompts"].items():
            if prompt_id not in doc.meta["annotations"].keys():
                self.stats.stats["task-initialization.prompt-has-no-annotations"] += 1
                continue

            all_labels = []
            all_label_codes = []
            all_evidence_nodes = []
            verbalized_prompt = self._create_prompt(prompt["outcome"], prompt["intervention"], prompt["comparator"])

            for annotation in doc.meta["annotations"][prompt_id]:
                evidence_nodes = self._find_evidence_nodes(doc, annotation)

                all_label_codes.append(annotation["label_code"])
                all_labels.append(annotation["label"])
                all_evidence_nodes.append(evidence_nodes)

            instance = EvidenceInferenceInstance(
                task_name=self.task_name,
                example_id=f"prompt_{prompt_id}",
                document=doc,
                question=verbalized_prompt,
                statement='',
                prompt=verbalized_prompt,
                extraction_level='p',
                extraction_candidates=extraction_candidates,
                free_text_answer=all_labels,
                answer_type=['None' for _ in all_labels],
                extraction_nodes=all_evidence_nodes,
                pmc_id=doc.meta["pmc_id"],
                prompt_id=prompt_id,
                outcome=prompt["outcome"],
                intervention=prompt["intervention"],
                comparator=prompt["comparator"],
                labels=all_labels,
                label_codes=all_label_codes,
            )
            instances.append(instance)

        return instances

    @staticmethod
    def _create_prompt(outcome: str, intervention: str, comparator: str) -> str:
        return f"With respect to {outcome}, characterize the reported difference" \
               f" between patients receiving {intervention} and those receiving {comparator}." \
               f"Choose between 'significantly decreased', 'no significant difference', " \
               f"and 'significantly increased'."

