"""
Run the evaluation environment that fine-tunes and evaluates a model on an end task.
"""
import logging

import hydra
import pytorch_lightning as pl
import nltk

from evaluation.tasks.govreport_task import GovReportTask
from models.oracle import OracleForExtractionModel
from evaluation.run import run
from evaluation.tasks.evidence_inference_task import EvidenceInferenceTask
from evaluation.tasks.qasper_task import QASPERTask
from evaluation.tasks.natural_questions_task import NaturalQuestionsTask
from evaluation.tasks.wice_task import WiceTask
from evaluation.tasks.contract_nli_task import ContractNLITask
from evaluation.tasks.qasa_task import QASATask
from models.seq2seq_lm import (
    Seq2SeqForExtractionModel
)
from models.causal_lm import CausalLMForExtractionModel
from models.api_lm import APILMForExtractionModel
from config_lib.base_config import BaseConfig, init_config

logger = logging.getLogger()

# Get all available task classes
TASK_CLASSES = [
    EvidenceInferenceTask,
    QASPERTask,
    NaturalQuestionsTask,
    GovReportTask,
    WiceTask,
    ContractNLITask,
    QASATask
]

# Get all available model classes
MODEL_CLASSES = [
    OracleForExtractionModel,
    Seq2SeqForExtractionModel,
    CausalLMForExtractionModel,
    APILMForExtractionModel
]


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: BaseConfig) -> None:
    # Run config automations
    init_config(config)

    logger.info(f"Commit hash: {config.commit_hash}")

    if config.remote_debug:
        # Set up remote debugging
        import pydevd_pycharm
        pydevd_pycharm.settrace('10.167.11.14', port=3851, stdoutToServer=True, stderrToServer=True)
        # Don't do multiprocessing when debugging
        config.dataloader_num_workers = 0

    logger.info("Run training.")

    logger.info("Fix random seeds.")
    pl.seed_everything(config.random_seed)

    # select task
    task_class = None
    for t in TASK_CLASSES:
        if t.task_name == config.task.task_name:
            task_class = t
            break
    if task_class is None:
        logger.error("Did not find a suitable task!")
        assert False, "Did not find a suitable task!"

    # select model
    model_class = None
    for mw in MODEL_CLASSES:
        if mw.model_class_name == config.model.model_class:
            model_class = mw
            break

    if model_class is None:
        logger.error("Did not find a suitable model!")
        assert False, "Did not find a suitable model!"

    run(
        model_class=model_class,
        task_class=task_class,
        config=config
    )

    logger.info(f"All done!")


if __name__ == "__main__":
    import sys

    import absl.flags as flags

    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    main()
