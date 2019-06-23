import logging
import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms

from modules.training.trainer import Trainer
from modules.models.model import Model
from modules.training.temperature_schedulers import InverseTimestepDecay
from cifar10.simple_model import SimpleModel
from cifar10.util import get_train_validation_loader, get_test_loader
from cifar10.predictor import Predictor

import config

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

train_loader, val_loader = get_train_validation_loader(
    "./data", batch_size=config.BATCH_SIZE, validation_size=config.VALIDATION_SIZE, class_list=config.CLASS_LIST
)
test_loader = get_test_loader("./data", batch_size=config.BATCH_SIZE, class_list=config.CLASS_LIST)

if config.USING_PROGRESSIVE_LEARNING:
    similarity_vectors_fn = os.path.join(config.SIMILARITY_VECTORS_PATH, "{}.th".format(config.SIMILARITY_VECTORS_FN))
    temperature_scheduler = InverseTimestepDecay(t_initial=config.T_INITIAL, decay_rate=config.DECAY_RATE)
else:
    similarity_vectors_fn = None
    temperature_scheduler = None

model = SimpleModel(similarity_vectors_fn=similarity_vectors_fn)

if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
    print("GPU available.")
else:
    cuda_device = -1

optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

trainer = Trainer(
    model,
    optimizer,
    train_dataloader=train_loader,
    validation_dataloader=test_loader,
    cuda_device=cuda_device,
    num_epochs=config.NUM_EPOCHS,
    serialization_dir=config.CHECKPOINTS_PATH,
    patience=config.PATIENCE,
    temperature_scheduler=temperature_scheduler
)
trainer.train()

predictor = Predictor(model, cuda_device=cuda_device)
pred_metrics = predictor.predict(test_loader)
print(pred_metrics)
predictor.save_confusion_matrix()

