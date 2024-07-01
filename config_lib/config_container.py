from __future__ import annotations
from pathlib import PosixPath

from omegaconf import OmegaConf, DictConfig

class ConfigContainer:

    @classmethod
    def dict_config_to_json_dict(cls, dict_config: DictConfig):
        config_object = OmegaConf.to_object(dict_config)
        return config_object.to_json_dict()

    def to_json_dict(self):
        return_dict = {}
        for k,v in self.__dict__.items():
            if isinstance(v, ConfigContainer):
                return_dict[k] = v.to_json_dict()
            elif isinstance(v, PosixPath):
                return_dict[k] = str(v)
            else:
                return_dict[k] = v

        return return_dict