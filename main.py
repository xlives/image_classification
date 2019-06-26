import logging
import os
import sys
import torch
import datetime
import shutil
import torchvision
import torchvision.transforms as transforms

from core.training.trainer import Trainer
from core.models.model import Model
from core.training.temperature_schedulers import InverseTimestepDecay
from core.nn.regularizers import L2Regularizer, RegularizerApplicator
from cifar10.model import CifarModel
from cifar10.util import get_train_validation_loader, get_test_loader
from cifar10.predictor import Predictor
from cifar10.modules import SimpleConvNet, EfficientNet
from cifar10.modules.efficient_net import efficient_net_b0_cfg
from cifar10.logger import CifarLogger

import config

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

train_loader, val_loader = get_train_validation_loader(
    config.DATA_PATH, batch_size=config.BATCH_SIZE, validation_size=config.VALIDATION_SIZE, class_list=config.CLASS_LIST, augment=config.AUGMENT_DATA
)
test_loader = get_test_loader(config.DATA_PATH, batch_size=config.BATCH_SIZE, class_list=config.CLASS_LIST)

if config.USE_PROGRESSIVE_LEARNING:
    similarity_vectors_fn = os.path.join(config.SIMILARITY_VECTORS_PATH, "{}.th".format(config.SIMILARITY_VECTORS_FN))
    temperature_scheduler = InverseTimestepDecay(t_initial=config.T_INITIAL, decay_rate=config.DECAY_RATE)
    experiment_name = "T0={}, decay_rate={}".format(config.T_INITIAL, config.DECAY_RATE)
else:
    similarity_vectors_fn = None
    temperature_scheduler = None
    experiment_name = "no_progressive_learning"

time_now = datetime.datetime.now().strftime("%m-%d-%Y, %H-%M-%S")
experiment_name += " time={}".format(time_now)

core_module = SimpleConvNet(num_classes=len(config.CLASS_LIST))
# core_module = EfficientNet(cfg=efficient_net_b0_cfg, num_classes=len(config.CLASS_LIST))
model = CifarModel(core_module=core_module, similarity_vectors_fn=similarity_vectors_fn)

if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
    print("GPU available.")
else:
    cuda_device = -1

# optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)


figure_path = os.path.join(config.FIGURES_PATH, experiment_name)
serialization_dir = os.path.join(config.CHECKPOINTS_PATH, experiment_name)

predictor = Predictor(model, class_list=config.CLASS_LIST, figure_path=figure_path, cuda_device=cuda_device)
custom_logger = CifarLogger(predictor, test_loader, config.LOG_EPOCH_INTERVAL)

trainer = Trainer(
    model,
    optimizer,
    train_dataloader=train_loader,
    validation_dataloader=test_loader,
    cuda_device=cuda_device,
    num_epochs=config.NUM_EPOCHS,
    serialization_dir=serialization_dir,
    patience=config.PATIENCE,
    temperature_scheduler=temperature_scheduler,
    custom_logger=custom_logger,
)
trainer.train()

pred_metrics = predictor.predict(test_loader)
print(pred_metrics)
predictor.save_confusion_matrix("best, accuracy={:.4f}".format(pred_metrics["accuracy"]))

