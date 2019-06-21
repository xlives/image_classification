import logging
from typing import Dict, List, Union, Any
from modules.models.model import Model


class TrainerBase:
    def __init__(self, serialization_dir: str, cuda_device: int = -1) -> None:
        self._serialization_dir = serialization_dir

        # not support multiple gpu yet
        self._cuda_device = cuda_device

    def _move_to_gpu(self, model: Model) -> Model:
        if self._cuda_device != -1:
            return model.cuda(self._cuda_device)
        else:
            return model

    def train(self) -> Dict[str, Any]:
        raise NotImplementedError
