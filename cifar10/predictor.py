import torch
from typing import List
import os
from core.models.model import Model
from core.nn import util as nn_util
from core.training import util as training_util
from core.common.tqdm import Tqdm
from core.common.checks import ConfigurationError
import numpy as np
import matplotlib.pyplot as plt


class Predictor:
    def __init__(
        self,
        model: Model,
        class_list: List[str],
        figure_path: str,
        cuda_device: int = -1,
    ):
        self.model = model
        self.cuda_device = cuda_device
        self.class_list = class_list
        self.figure_path = figure_path

        self.reset()

    def reset(self):
        num_classes = len(self.class_list)
        self.confusion_matrix = torch.zeros((num_classes, num_classes))

    def update_confusion_matrix(self, logits, labels):
        num_classes = len(self.class_list)
        assert logits.size(-1) == num_classes

        if labels.dim() != logits.dim() - 1:
            raise ConfigurationError(
                "labels must have dimension == logits.size() - 1 but "
                "found tensor of shape: {}".format(logits.size())
            )
        if (labels >= num_classes).any():
            raise ConfigurationError(
                "A label passed to Categorical Accuracy contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )
        logits = logits.view((-1, num_classes))
        labels = labels.view(-1).long()

        top_k = logits.max(-1)[1].unsqueeze(-1)
        labels = labels.unsqueeze(-1)

        for i, label in enumerate(labels):
            pred = top_k[i]
            label, pred = label.item(), pred.item()
            self.confusion_matrix[label][pred] += 1

    def save_similarity_matrix(self, name, temperature):
        if self.model.similarity_vectors is None:
            return
        similarity_matrix = self.model.similarity_vectors.weight.data
        similarity_matrix = torch.nn.functional.softmax(similarity_matrix / temperature, dim=-1)
        similarity_matrix = similarity_matrix.cpu().numpy()
        save_similarity_matrix_figure(
            self.figure_path, name, similarity_matrix, self.class_list, self.class_list
        )

    def save_confusion_matrix(self, name="cifar10_confusion_matrix"):
        save_confusion_matrix_figure(
            self.figure_path,
            name,
            self.confusion_matrix.numpy(),
            self.class_list,
            self.class_list,
        )

    def predict(self, dataloader: torch.utils.data.DataLoader):
        self.reset()

        dataloader_tqdm = Tqdm.tqdm(dataloader)
        self.model.eval()
        with torch.no_grad():
            batches_this_epoch = 0
            pred_loss = 0
            for batch in dataloader_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                output_dict = self.model(batch)

                logits = output_dict["logits"]
                labels = batch[1]
                self.update_confusion_matrix(logits, labels)

                loss = output_dict["loss"]
                if loss is not None:
                    # You shouldn't necessarily have to compute a loss for validation, so we allow for
                    # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                    # currently only used as the divisor for the loss function, so we can safely only
                    # count those batches for which we actually have a loss.  If this variable ever
                    # gets used for something else, we might need to change things around a bit.
                    batches_this_epoch += 1
                    pred_loss += loss.detach().cpu().numpy()

                # Update the description with the latest metrics
                pred_metrics = training_util.get_metrics(
                    self.model, pred_loss, batches_this_epoch
                )
                description = training_util.description_from_metrics(
                    pred_metrics)
                dataloader_tqdm.set_description(description, refresh=False)

        return pred_metrics


def save_similarity_matrix_figure(path, name, matrix, xlabels, ylabels):
    plt.figure()

    plt.matshow(matrix, cmap="viridis")
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=90, fontsize=5)
    plt.yticks(np.arange(len(ylabels)), ylabels, fontsize=5)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "black" if matrix[i][j] > 0.5 else "white"
            plt.text(
                j,
                i,
                "{:.5f}".format(matrix[i][j]),
                ha="center",
                va="center",
                color=color,
                fontsize=5,
            )

    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, "{}.pdf".format(name)))


def save_confusion_matrix_figure(path, name, matrix, xlabels, ylabels):
    plt.figure()

    plt.matshow(matrix, cmap="viridis")
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=90, fontsize=5)
    plt.yticks(np.arange(len(ylabels)), ylabels, fontsize=5)
    plt.xlabel("Predictions")
    plt.ylabel("Targets")

    for i in range(matrix.shape[0]):
        row_sum = sum(matrix[i])
        for j in range(matrix.shape[1]):
            count = int(matrix[i][j])
            percentage = count / row_sum * 100
            color = "black" if percentage > 50 else "white"
            plt.text(
                j,
                i,
                "{:0.1f}% ({:d})".format(percentage, count),
                ha="center",
                va="center",
                color=color,
                fontsize=5,
            )

    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, "{}.pdf".format(name)))
