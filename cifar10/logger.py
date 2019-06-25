import torch
from core.training.custom_logger import CustomLogger
from cifar10.predictor import Predictor


class CifarLogger(CustomLogger):
    def __init__(self, predictor: Predictor, dataloader: torch.utils.data.DataLoader, log_epoch_interval: int):
        self.predictor = predictor
        self.dataloader = dataloader
        self.log_epoch_interval = log_epoch_interval

    def log_epoch(self, epoch: int, **kwargs):
        if epoch % self.log_epoch_interval != 0:
            return

        pred_metrics = self.predictor.predict(self.dataloader)
        name = "epoch={:d}, accuracy={:.4f}".format(epoch, pred_metrics["accuracy"])
        self.predictor.save_confusion_matrix(name)

        
