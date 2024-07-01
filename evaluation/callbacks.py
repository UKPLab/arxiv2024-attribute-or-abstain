import datetime
import json
import os
from pathlib import Path
from typing import List
import logging

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from config_lib.base_config import BaseConfig
from evaluation.common import BasePrediction, BaseTask, Statistics, BaseInstance, Partition
from evaluation.util import logger
from evaluate_predictions import analyze_predictions

logger = logging.getLogger(__name__)

class CustomCallback():
    predictions: List[BasePrediction]
    task: BaseTask
    stats: Statistics
    config: BaseConfig

    def __init__(
            self,
            task: BaseTask,
            stats: Statistics,
            config: BaseConfig,
            tb_writer: SummaryWriter,
            test_predictions: List[BasePrediction],
    ):
        super(CustomCallback, self).__init__()
        self.predictions = []
        self.task = task
        self.stats = stats
        self.config = config
        self.test_predictions: List[BasePrediction] = test_predictions
        self.tb_writer = tb_writer
        self.log_loss_every_n_batches = config.log_loss_every_n_batches
        # Log the prompt for the first test / dev instance
        # Store when the prompt has been logged
        self.prompt_logged = False

        # Keep running average of loss
        self.avg_loss = 0
        self.batch_counter = 0

    @staticmethod
    def _format_prompt_for_logging(
            prompt: str
    ):
        prompt = prompt.replace('\n', '  \n  \n')

        log_text = f'####Prompt  \n  \n```  \n  \n{prompt}  \n  \n```'

        return log_text

    @staticmethod
    def _format_prediction_for_logging(
            predictions: List[BasePrediction],
            instances: List[BaseInstance],
    ) -> str:
        def format_key_value(
                key: str,
                value: str | List[str] | List[List[str]]
        ) -> str:
            if isinstance(value, list):
                value = '- ' + '  \n- '.join(v for v in value)

            return f'####{key.title()}  \n{value}'

        prediction_log_keys = [
            'free_text_answer', 'extraction_nodes', 'raw_generation'
        ]

        log_texts = []
        for prediction, instance in zip(predictions, instances):
            log_texts.append(f'{instance.document.nodes[0].content}')
            log_texts.append(f'Example ID: {instance.example_id}')
            log_texts.append(format_key_value('question', instance.question))
            if instance.statement:
                log_texts.append(format_key_value('statement', instance.statement))

            log_texts.append('***')

            log_texts.append(format_key_value('gold answer', instance.free_text_answer[0]))
            if len(instance.extraction_nodes[0]) > 0:
                if isinstance(instance.extraction_nodes[0][0], list):
                    ids_to_log = [
                        ', '.join(n.ix for n in extraction_node_list)
                        for extraction_node_list in instance.extraction_nodes[0]
                    ]
                    node_texts_to_log = ''
                else:
                    ids_to_log = ', '.join(n.ix for n in instance.extraction_nodes[0])
                    node_texts_to_log = [
                        n.content[:100] + '...'
                        for n in instance.extraction_nodes[0]
                    ]
            else:
                ids_to_log = ''
                node_texts_to_log = []

            log_texts.append(format_key_value(
                'gold extraction node texts',
                node_texts_to_log
            ))
            log_texts.append(
                format_key_value(
                    'gold extraction node ids',
                    ids_to_log
                )
            )

            log_texts.append('***')

            # Log prediction
            log_texts.append(
                format_key_value(
                    'predicted free text answer',
                    prediction.free_text_answer
                )
            )
            if len(prediction.extraction_nodes) > 0:
                if isinstance(prediction.extraction_nodes[0], list):
                    ids_to_log = [
                        ', '.join(n.ix for n in extraction_node_list)
                        for extraction_node_list in prediction.extraction_nodes
                    ]
                    node_texts_to_log = ''
                else:
                    ids_to_log = ', '.join(n.ix for n in prediction.extraction_nodes)
                    node_texts_to_log = [
                        n.content[:100] + '...'
                        for n in prediction.extraction_nodes
                    ]
            else:
                ids_to_log = ''
                node_texts_to_log = []

            log_texts.append(format_key_value(
                'predicted extraction node texts',
                node_texts_to_log
            ))
            log_texts.append(
                format_key_value(
                    'predicted extraction node ids',
                    ids_to_log
                )
            )

            log_texts.append(
                format_key_value(
                    'raw generation',
                    f'```  \n  \n{prediction.raw_generation}  \n  \n```'
                )
            )

        log_text = '  \n  \n'.join(log_texts)
        return log_text

    def log_openai_cost(
            self,
            prompt_tokens: int,
            completion_tokens: int,
            model_name: str,
            path: Path,
            hash: str
    ):
        with open(path / 'cost.json') as f:
            cost_dict = json.load(f)

        prompt_cost_per_token = cost_dict[model_name]['prompt']
        completion_cost_per_token = cost_dict[model_name]['completion']

        prompt_cost = prompt_tokens * prompt_cost_per_token
        completion_cost = completion_tokens * completion_cost_per_token
        total_cost = prompt_cost + completion_cost

        self.tb_writer.add_scalar(
            f'test/openai_cost',
            total_cost,
            0
        )

        # Write cost per model and day
        date_str = str(datetime.date.today())
        csv_filename = f'{date_str}_{model_name}.csv'
        if os.path.exists(path / csv_filename):
            daily_model_cost_table = pd.read_csv(path / csv_filename)
            current_cumulative_model_cost = daily_model_cost_table['Cumulative Cost'].iloc[-1]
            current_cumulative_model_cost += total_cost
            with open(path / csv_filename, 'a') as f:
                f.write(f'{hash},{total_cost},{current_cumulative_model_cost}\n')
        else:
            daily_model_cost_table = pd.DataFrame({
                'Hash': [hash],
                'Experiment Cost': [total_cost],
                'Cumulative Cost': [total_cost]
            })
            daily_model_cost_table.to_csv(path / csv_filename, index=False)

        # Write cumulative cost combined for all models
        cumulative_cost_table = pd.read_csv(path / 'cost.csv')
        current_cumulative_cost = cumulative_cost_table['Cumulative Cost'].iloc[-1]
        current_cumulative_cost += total_cost

        with open(path / f'cost.csv', 'a') as f:
            f.write(
                f'{hash},{model_name},{total_cost},{current_cumulative_cost}\n'
            )

        return total_cost, current_cumulative_cost

    def on_test_batch_end(
            self,
            outputs: List[BasePrediction],
            batch,
            batch_idx
    ) -> None:
        self.predictions += outputs
        self.test_predictions += outputs

        if self.config.log_test_predictions:
            log_text = self._format_prediction_for_logging(
                outputs,
                batch['instances'],
            )

            if not self.prompt_logged:
                # Add prompt to first log
                prompt_text = self._format_prompt_for_logging(batch["input_texts"][0])
                self.prompt_logged = True
                log_text = log_text + '  \n  \n***  \n  \n' + prompt_text

            self.tb_writer.add_text(
                f'Test',
                log_text,
                global_step=batch_idx
            )

    def on_test_end(
            self,
            global_step = 1,
            openai_completion_tokens: int = None,
            openai_prompt_tokens: int = None
    ) -> None:
#        import pdb
#        pdb.set_trace()
        if self.config.use_dev_as_test_data:
            logger.info(f"Final evaluation on {Partition.DEV.value} data:")
            instances = self.task.dev_instances
            partition = Partition.DEV
        else:
            logger.info(f"Final evaluation on {Partition.TEST.value} data:")
            instances = self.task.test_instances
            partition = Partition.TEST

        if self.config.use_first_n_test_instances == -1:
            # Using all test instances
            assert len(self.predictions) == len(instances)

        # Create output directories
        out_dir = self.config.location.results / self.stats.to_hash()

        predictions_path = out_dir / f"predictions.jsonl"
        self.task.save_predictions(self.predictions, Path(predictions_path))

        os.mkdir(out_dir / 'evaluation')
        os.mkdir(out_dir / 'evaluation' / '1')
        results = analyze_predictions(
            self.predictions,
            instances,
            self.config.task.metrics,
            False,
            self.config.do_post_hoc_extract,
            False,
            True,
            self.config.task.answer_has_multiple_statements,
            out_dir=out_dir / 'evaluation' / '1',
            classes=self.config.task.classes,
            train_instances=self.task.train_instances,
            post_hoc_retrieval_model=self.config.post_hoc_retrieval_model,
            post_hoc_retrieval_k=self.config.post_hoc_retrieval_k,
            post_hoc_retrieval_threshold=self.config.post_hoc_retrieval_threshold,
            post_hoc_sbert_model_name=self.config.post_hoc_sbert_model_name,
            use_all_annotations_for_evidence_f1=self.config.task.use_all_annotations_for_evidence_f1
        )

        if openai_completion_tokens is not None:
            total_cost, cumulative_cost = self.log_openai_cost(
                openai_prompt_tokens,
                openai_completion_tokens,
                self.config.model.model_name,
                self.config.location.openai_cost_tracking,
                self.stats.to_hash()
            )
            results['OpenAI Cost'] = {
                'Experiment': total_cost,
                'Cumulative': cumulative_cost
            }

        self.stats.test_result = results

        for metric_name in self.config.task.metrics:
            score = results[metric_name]['score']
            self.tb_writer.add_scalar(
                f'test/{metric_name}',
                score,
                global_step,
            )
            evidence_f1 = results[metric_name]['evidence_f1']
            self.tb_writer.add_scalar(
                f'test/evidence_f1_{metric_name}',
                evidence_f1,
                global_step,
            )

        lines = json.dumps(results, indent=4).split("\n")
        for line in lines:
            logger.info(line)

        self.predictions = []
