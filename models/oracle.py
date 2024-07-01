import logging
import random
from typing import Any, List, ClassVar

import torch
import torch.nn.functional as F

from config_lib.base_config import BaseConfig
from evaluation.common import Statistics, BaseInstance, BasePrediction
from models.base_model import BaseModel
from evaluation.util import format_gold_answer, parse_answer

logger = logging.getLogger(__name__)


class Batch:
    instances: List[BaseInstance]

    def __init__(self, instances) -> None:
        self.instances = instances


class OracleForExtractionModel(BaseModel):
    model_class_name: ClassVar[str] = "oracle"

    dummy_layer: torch.nn.Linear

    def __init__(
            self,
            config: BaseConfig,
            stats: Statistics,
            train_dataset
    ) -> None:
        super(OracleForExtractionModel, self).__init__(config, stats, train_dataset)
        self.dummy_layer = torch.nn.Linear(
            in_features=3,
            out_features=2
        )

    def training_step(self, batch, batch_idx):
        tmp = torch.tensor([[0.1, 1.2, 2.3]], device=self.device)
        tmp = self.dummy_layer(tmp)
        loss = F.cross_entropy(tmp, torch.tensor([1], device=self.device))
        return loss

    def validation_step(self, batch, batch_idx):
        predictions = []
        for instance in batch['instances']:
            # make sure that for any of the annotations of the prompt, the oracle works perfectly
            idx = random.randint(0, len(instance.free_text_answer) - 1)
            if self.hydra_config.model.oracle_return_random_non_gold_evidence:
                all_gold_extraction_nodes = set([
                    n for l in instance.extraction_nodes
                    for n in l
                ])
                extraction_candidates_without_gold = [
                    n for n in instance.extraction_candidates
                    if n not in all_gold_extraction_nodes
                ]
                extraction_nodes = random.sample(
                    extraction_candidates_without_gold,
                    self.hydra_config.task.keep_k_random_nodes_when_pruning
                )
            else:
                extraction_nodes = instance.extraction_nodes[idx]
            # Create answer string to test answer parsing
            node_id_to_node_mapping = {}
            for node in instance.document.nodes:
                node_id = node.ix.split('_')[-1]
                structural_token = eval(f'f"""{self.hydra_config.node_id_template}"""')
                node_id_to_node_mapping[structural_token] = node
            oracle_generated_text = format_gold_answer(
                instance,
                self.extraction_mode,
                self.hydra_config.answer_format,
                self.hydra_config.required_aspects,
                self.hydra_config.task.answer_has_multiple_statements,
                shorten_text_to_first_n_chars=None,
                include_node_types=self.hydra_config.task.is_include_node_types,
                node_id_to_node_mapping=node_id_to_node_mapping
            )
            # Get extraction nodes from free text answer
            free_text_answer, extraction_nodes = parse_answer(
                oracle_generated_text,
                instance.extraction_candidates,
                self.extraction_mode,
                self.hydra_config.answer_format,
                self.hydra_config.required_aspects,
                instance.task_name,
                unanswerable_keywords=self.hydra_config.unanswerable_keywords,
                answer_has_multiple_statements=self.hydra_config.task.answer_has_multiple_statements,
                node_id_to_node_mapping=node_id_to_node_mapping,
                classes=self.hydra_config.task.classes,
                return_all_extraction_candidates=self.hydra_config.return_all_extraction_candidates
            )
            prediction = BasePrediction(
                task_name=instance.task_name,
                example_id=instance.example_id,
                free_text_answer=free_text_answer,
                extraction_nodes=extraction_nodes,
                raw_generation=oracle_generated_text
            )
            predictions.append(prediction)
        return predictions

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def training_collate_fn(self, instances: List[BaseInstance]) -> Any:
        return instances

    def validation_collate_fn(self, instances: List[BaseInstance]) -> Any:
        if self.hydra_config.do_retrieve_then_read:
            if self.hydra_config.do_retrieve_then_read:
                instances = [
                    self.retrieve_then_read_model.retrieve_and_reduce(instance)
                    for instance in instances
                ]
        return {
            'instances': instances,
            'input_texts': ['']
        }

    def adjust_batch_size_and_accumulation_steps(
            self
    ) -> None:
        pass