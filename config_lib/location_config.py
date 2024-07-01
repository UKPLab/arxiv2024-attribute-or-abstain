from dataclasses import dataclass
from pathlib import Path

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from config_lib.config_container import ConfigContainer


@dataclass
class BaseLocationConfig(ConfigContainer):
    """
    This class defines the base config for the location of the data and the models.
    :param datasets: The path to the datasets (raw and ITG format)
    :param models: The path to the model stored during training.
    :param tensorboard: The path to the tensorboard logs.
    :param results: The path to the results (metrics and dump of the config).
    :param predictions: The path to the predictions.
    :param configs: The path to the config files stored for each compute job.
    :param openai_cost_tracking: Path to folder for openai cost tracking
    :param prompts: The path to the prompt elements.
    """

    datasets: Path = MISSING
    models: Path = MISSING
    tensorboard: Path = MISSING
    results: Path = MISSING
    predictions: Path = MISSING
    configs: Path = MISSING
    openai_cost_tracking: Path = MISSING
    prompts: Path = MISSING

cs = ConfigStore.instance()
cs.store(
    name="base_location_config",
    group='location',
    node=BaseLocationConfig
)