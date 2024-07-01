import json
import logging
import os
import shutil
import time
import traceback
from typing import Type

import torch.cuda
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from config_lib.config_container import ConfigContainer
from evaluation.common import BaseTask, Statistics, CustomDataset
from models.base_model import BaseModel
from config_lib.base_config import BaseConfig
from evaluation.callbacks import CustomCallback
from evaluation.validator import Validator

logger = logging.getLogger(__name__)


def run(
        model_class: Type[BaseModel],
        task_class: Type[BaseTask],
        config: BaseConfig | DictConfig
) -> None:
    """This function handles training and evaluation"""
    logger.info(f"Evaluation of '{config.model.model_name}' on '{task_class.task_name}':")
    logger.info(f"Description: '{config.description}'")

    model_name = config.model.model_name
    task_name = task_class.task_name

    # Statistics object stores basic statistics of training and evaluation
    stats = Statistics(
        model_name=model_name,
        task_name=task_name,
        description=config.description,
        config=config
    )

    start_time = time.time()

    ############################################################################
    # task initialization
    ############################################################################
    logger.info("Initialize the task.")
    tick = time.time()
    task = task_class(config, stats)
    tack = time.time()
    logger.info(f"Initialized the task in {tack - tick:0.4f}s.")

    ############################################################################
    # dataloader initialization
    ############################################################################
    logger.info("Initialize the dataloaders.")
    tick = time.time()

    # initialize the dataloaders
    train_dataset = CustomDataset(task.train_instances)

    dev_dataset = CustomDataset(task.dev_instances)

    if config.use_dev_as_test_data:
        logger.info("Using the dev instances as test instances.")
        test_dataset = CustomDataset(task.dev_instances)
    else:
        logger.info("USING THE ACTUAL TEST INSTANCES!!!")
        test_dataset = CustomDataset(task.test_instances)

    # When use_first_n_test_instance is not set to -1, shorten the test dataset
    # to the first n instances
    if config.use_first_n_test_instances != -1:
        test_dataset = test_dataset.get_subset(slice(0, config.use_first_n_test_instances))

    tack = time.time()
    logger.info(f"Initialized the dataloaders in {tack - tick:0.4f}s.")

    ############################################################################
    # model initialization
    ############################################################################
    logger.info("Initialize the model.")
    tick = time.time()
    logger.info("Create new model.")
    model = model_class(config, stats, train_dataset)

    tack = time.time()
    logger.info(f"Initialized the model in {tack - tick:0.4f}s.")

    ############################################################################
    # dataloader initialization
    ############################################################################
    # Train and dev dataloaders are initialized in Trainer

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=model.validation_collate_fn,
        batch_size=config.model.batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
    )

    ############################################################################
    # logger initialization
    ############################################################################
    logger.info("Initialize the logger.")
    tick = time.time()

    tensorboard_logger = SummaryWriter(
        log_dir=str(config.location.tensorboard / f"{stats.task_name}" / f'{stats.model_name} {stats.description} {stats.to_hash()} {config.slurm_job_id}'),
        filename_suffix=''
    )

    tack = time.time()
    logger.info(f"Initialized the logger in {tack - tick:0.4f}s.")

    ############################################################################
    # callback initialization
    ############################################################################
    logger.info("Initialize the callbacks.")
    tick = time.time()

    test_predictions = []

    custom_callback = CustomCallback(
        task,
        stats,
        config,
        tensorboard_logger,
        test_predictions
    )

    tack = time.time()
    logger.info(f"Initialized the callbacks in {tack - tick:0.4f}s.")

    ############################################################################
    # Store config
    ############################################################################
    results_dir_path = config.location.results / stats.to_hash()
    if os.path.exists(results_dir_path):
        critical_files_exist = False
        if os.path.exists(results_dir_path / 'model'):
            model_path = results_dir_path / 'model'
            if len(os.listdir(model_path)) > 0:
                critical_files_exist = True
        if os.path.exists(results_dir_path / 'predictions.jsonl'):
            critical_files_exist = True
        if critical_files_exist:
            raise FileExistsError(f'Output dir contains critical files!')
        else:
            shutil.rmtree(results_dir_path)
    os.mkdir(results_dir_path)
    # Store config
    logger.info(f'Hash: {stats.to_hash()}')
    with open(
        results_dir_path / 'config.json', 'w'
    ) as file:
        json.dump(
            ConfigContainer.dict_config_to_json_dict(config),
            file,
            indent=2
        )

    train_steps = 0
    if config.do_train:

        ############################################################################
        # trainer initialization
        ############################################################################
        logger.info("Initialize the trainer.")
        tick = time.time()

        training_args = TrainingArguments(
            output_dir=str(results_dir_path / 'model'),
            evaluation_strategy="steps",
            eval_steps=config.model.val_check_interval,
            per_device_train_batch_size=config.model.batch_size,
            per_device_eval_batch_size=config.model.batch_size,
            gradient_accumulation_steps=config.model.accumulate_grad_batches,
            learning_rate=config.model.learning_rate,
            max_steps=config.model.max_steps,
            lr_scheduler_type=config.lr_scheduler_type,
            warmup_ratio=0.03,
            logging_dir=str(config.location.tensorboard / f"{stats.task_name}" / f'{stats.model_name} {stats.description} {stats.to_hash()} {config.slurm_job_id}'),
            logging_strategy='steps',
            logging_steps=config.log_loss_every_n_batches,
            seed=config.random_seed,
            bf16=True,
            dataloader_num_workers=config.dataloader_num_workers,
            save_steps=500,
            save_strategy='steps',
            save_total_limit=2,
            remove_unused_columns=False,
            gradient_checkpointing=True
        )

        trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=model.training_collate_fn
        )

        tack = time.time()
        logger.info(f"Initialized the trainer in {tack - tick:0.4f}s.")

        ############################################################################
        # fine-tuning
        ############################################################################
        logger.info("Fine-tuning.")
        tick = time.time()

        with open(
            results_dir_path / 'model' / 'custom_config.json',
            'w'
        ) as out_file:
            json.dump(
                ConfigContainer.dict_config_to_json_dict(config),
                out_file,
                indent=2
            )

        trainer.train()
        train_steps = trainer.state.global_step

        trainer.save_model()

        tack = time.time()
        logger.info(f"Fine-tuning done in {tack - tick:0.4f}s.")

    ############################################################################
    # evaluation
    ############################################################################
    logger.info("Evaluation.")
    tick = time.time()

    if not config.do_train and torch.cuda.is_available():
        # Loaded new model, need to send to device
            device = torch.device('cuda:0')
            model = model.to(device)

    validator = Validator(callbacks=custom_callback)

    if config.remote_debug:
        validator.validate(
            model=model,
            val_dataloader=test_dataloader,
            is_test=True
        )
    else:
        # When not debugging, catch any testing error to save predictions later
        try:
            validator.validate(
                model=model,
                val_dataloader=test_dataloader,
                is_test=True
            )
        except Exception:
            logger.error(
                'Testing failed: '
                f'{traceback.format_exc()}'
            )

    tack = time.time()
    logger.info(f"Evaluation done in {tack - tick:0.4f}s.")

    ############################################################################
    # end
    ############################################################################

    end_time = time.time()
    logger.info(f"All done in {end_time - start_time:0.4f}s.")

    if model.model_class_name == 'api_lm':
        openai_prompt_tokens = model.prompt_tokens
        openai_completion_tokens = model.completion_tokens
    else:
        openai_prompt_tokens = None
        openai_completion_tokens = None

    custom_callback.on_test_end(
        train_steps,
        openai_completion_tokens=openai_completion_tokens,
        openai_prompt_tokens=openai_prompt_tokens
    )

    stats.total_time = end_time - start_time

    logger.info("Save the statistics.")
    path = results_dir_path / 'statistics.json'
    with open(path, "w", encoding="utf-8") as file:
        json.dump(stats.to_json_dict(), file, indent=4)
