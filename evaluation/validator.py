from typing import Dict, Union, Any, Optional, List, Tuple

import torch.cuda
from torch import nn
from torch.utils.data import DataLoader
import tqdm
from transformers import Trainer as TFTRainer

from evaluation.callbacks import CustomCallback
from evaluation.common import BasePrediction
from models.base_model import BaseModel


class Validator:

    def __init__(
            self,
            precision: str | int = 32,
            callbacks: CustomCallback = None
    ):
        self.callbacks = callbacks
        if precision == 32:
            dtype = torch.float32
        elif precision in ['bf16', 'bf16-true', 'bf16-mixed']:
            dtype = torch.bfloat16
        elif precision == 16:
            dtype = torch.float16
        else:
            raise NotImplementedError
        self.torch_dtype = dtype

    def validate(
            self,
            model: BaseModel,
            val_dataloader: DataLoader,
            val_epoch: int = None,
            is_test: bool = False
    ):
        if torch.cuda.is_available():
            device_type = 'cuda'
        else:
            device_type = 'cpu'

        model.eval()

        with tqdm.tqdm(val_dataloader, unit='batch') as bar:
            if is_test:
                bar.set_description('Test')
            else:
                bar.set_description(f'Validation epoch {val_epoch}')

            for _batch_ix, batch in enumerate(bar):
                batch_ix = _batch_ix + 1
                if self.torch_dtype != torch.float32:
                    with torch.autocast(device_type, dtype=self.torch_dtype):
                        predictions = model.validation_step(
                            batch, batch_ix
                        )
                else:
                    predictions = model.validation_step(
                            batch, batch_ix
                        )

                self.callbacks.on_test_batch_end(
                    predictions,
                    batch,
                    batch_ix,
                )

class Trainer(TFTRainer):

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> List[BasePrediction]:
        pass
