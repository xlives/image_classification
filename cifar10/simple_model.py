import torch
from modules.models.model import Model
import torch.nn.functional as F
from overrides import overrides
from modules.training.metrics import CategoricalAccuracy, Average
import config


class SimpleModel(Model):
    def __init__(self, similarity_vectors_fn: str = None):
        super(SimpleModel, self).__init__()

        num_classes = len(config.CLASS_LIST)

        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, num_classes)

        self.accuracy = CategoricalAccuracy()
        self.real_loss = Average()
        if similarity_vectors_fn is not None:
            self.similarity_vectors = torch.nn.Embedding.from_pretrained(
                torch.load(similarity_vectors_fn)
            )
        else:
            self.similarity_vectors = None

    def loss_helper(self, logits, labels, temperature=None):
        probs = F.log_softmax(logits, dim=-1)

        real_loss = F.nll_loss(probs, labels, reduction="mean")

        # transform targets into probability distributions using Embedding
        # then compute loss using torch.nn.functional.kl_div
        if (
            self.training
            and self.similarity_vectors is not None
            and temperature is not None
        ):
            similarity_vectors = self.similarity_vectors(labels)
            similarity_targets = F.softmax(similarity_vectors / temperature, dim=1)
            train_loss = F.kl_div(probs, similarity_targets, reduction="mean")
        else:
            train_loss = real_loss
        return train_loss, real_loss

    @overrides
    def forward(self, batch, temperature=None):
        inputs, labels = batch
        x = self.pool(F.relu(self.conv1(inputs)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        loss, real_loss = self.loss_helper(logits, labels, temperature)

        self.accuracy(logits, labels)
        self.real_loss(real_loss)

        output_dict = {"loss": loss, "logits": logits}
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False):
        return {
            "accuracy": self.accuracy.get_metric(reset=reset),
            "real_loss": self.real_loss.get_metric(reset=reset).item(),
        }
