import logging
import time
from io import StringIO
from typing import List, ClassVar

import nltk
from intertext_graph.itgraph import IntertextDocument

from config_lib.base_config import BaseConfig
from evaluation.common import BaseInstance, BaseTask, Statistics, SingleFileDataset

logger = logging.getLogger(__name__)


class GovReportTask(BaseTask):
    task_name: ClassVar[str] = "govreport"

    _train_instances: List[BaseInstance]
    _dev_instances: List[BaseInstance]
    _test_instances: List[BaseInstance]

    def __init__(self, config: BaseConfig, stats: Statistics) -> None:
        super(GovReportTask, self).__init__(config, stats)

    def load_instances_from_itg(self) -> List[List[BaseInstance | SingleFileDataset]]:
        tick = time.time()

        logger.info("Load the GovReport-ITG dataset.")

        if self.config.task.govreport_part == "full":
            parts = ["gao", "crs"]
        elif self.config.task.govreport_part == "gao":
            parts = ["gao"]
        elif self.config.task.govreport_part == "crs":
            parts = ["crs"]
        else:
            assert False, f"Unknown dataset part '{self.config.task.govreport_part}'!"

        train_documents = []
        dev_documents = []
        test_documents = []
        govreport_or_fastfact = 'govreport'
        folder_path = self.config.location.datasets / 'govreport' / 'GovReport-ITG'
        for part in parts:
            train_path = folder_path / f"deep-{part}-train.jsonl"
            with open(train_path, "r", encoding="utf-8") as file:
                content = file.readlines()
            for document_json_str in content:
                with StringIO(document_json_str) as f:
                    intertext_document = IntertextDocument.load_json(f)
                    if govreport_or_fastfact != "fastfact" or intertext_document.meta["fastfact"] != []:
                        train_documents.append(intertext_document)

            dev_path = folder_path / f"deep-{part}-dev.jsonl"
            with open(dev_path, "r", encoding="utf-8") as file:
                content = file.readlines()
            for document_json_str in content:
                with StringIO(document_json_str) as f:
                    intertext_document = IntertextDocument.load_json(f)
                    if govreport_or_fastfact != "fastfact" or intertext_document.meta["fastfact"] != []:
                        dev_documents.append(intertext_document)

            test_path = folder_path / f"deep-{part}-test.jsonl"
            with open(test_path, "r", encoding="utf-8") as file:
                content = file.readlines()
            for document_json_str in content:
                with StringIO(document_json_str) as f:
                    intertext_document = IntertextDocument.load_json(f)
                    if govreport_or_fastfact != "fastfact" or intertext_document.meta["fastfact"] != []:
                        test_documents.append(intertext_document)

        self.stats.stats["task-initialization.num-train-documents"] = len(train_documents)
        self.stats.stats["task-initialization.num-dev-documents"] = len(dev_documents)
        self.stats.stats["task-initialization.num-test-documents"] = len(test_documents)

        tack = time.time()
        logger.info(f"Loaded {len(train_documents)} train documents, "
                    f"{len(dev_documents)} dev documents, and "
                    f"{len(test_documents)} test documents in {tack - tick:0.4f}s.")

        logger.info("Create the GovReport instances.")
        tick = time.time()

        train_instances = []
        for document in train_documents:
            train_instances += self._create_instances_from_document(document)
        self.stats.stats["task-initialization.num-train-instances"] = len(train_instances)

        dev_instances = []
        for document in dev_documents:
            dev_instances += self._create_instances_from_document(document)
        self.stats.stats["task-initialization.num-dev-instances"] = len(dev_instances)

        test_instances = []
        for document in test_documents:
            test_instances += self._create_instances_from_document(document)
        self.stats.stats["task-initialization.num-test-instances"] = len(test_instances)

        tack = time.time()
        logger.info(f"Created {len(train_instances)} train instances, "
                    f"{len(dev_instances)} dev instances, and "
                    f"{len(test_instances)} test instances in {tack - tick:0.4f}s.")

        return [train_instances, dev_instances, test_instances]

    def _create_instances_from_document(self, doc: IntertextDocument) -> List[BaseInstance]:
        if 'highlight' in doc.meta:
            highlight_questions = [
                f'{i + 1}. {section["section_title"]}'
                for i, section in enumerate(doc.meta['highlight'])
            ]
            question = (
                f'Structure your summary '
                f'according to the following questions: {" ".join(highlight_questions)} '
            )

        else:
            question = ''

        prompt = question
        summary = nltk.sent_tokenize(doc.meta['flat_summary'])

        if len(summary) < 5:
            return []

        return [BaseInstance(
            task_name='govreport',
            example_id=doc.meta['id'],
            document=doc,
            prompt=prompt,
            question=question,
            statement='',
            extraction_level='p',
            extraction_candidates=[n for n in doc.nodes if n.ntype == 'p'],
            free_text_answer=[summary],
            answer_type=['summary'],
            extraction_nodes=[[[] for _ in summary]],
            answer_has_multiple_statements=True
        )]


