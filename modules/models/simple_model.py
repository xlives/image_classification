import torch
from modules.models.model import Model
import torch.nn.functional as F
from overrides import overrides


class SimpleModel(Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        self.loss = torch.nn.CrossEntropyLoss()

    @overrides
    def forward(self, batch):
        inputs, labels = batch 
        x = self.pool(F.relu(self.conv1(inputs)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        outputs = self.fc3(x)

        loss = self.loss(outputs, labels)

        output_dict = {
            "loss": loss
        }
        return output_dict