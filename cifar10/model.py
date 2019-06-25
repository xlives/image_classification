import torch
from core.models.model import Model
import torch.nn.functional as F
from overrides import overrides
from core.training.metrics import CategoricalAccuracy, Average
from core.nn.regularizers import RegularizerApplicator
import config


class CifarModel(Model):
    def __init__(self, core_module, similarity_vectors_fn: str = None, regularizer: RegularizerApplicator = None):
        super().__init__(regularizer=regularizer)

        self.core_module = core_module

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
            similarity_targets = F.softmax(similarity_vectors / temperature, dim=-1)
            train_loss = F.kl_div(probs, similarity_targets, reduction="batchmean")
        else:
            train_loss = real_loss
        return train_loss, real_loss

    @overrides
    def forward(self, batch, temperature=None):
        inputs, labels = batch
        logits = self.core_module(inputs)

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
