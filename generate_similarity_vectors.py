import os
import torch
import numpy as np
import config
from enum import Enum, auto


class Property(Enum):
    ANIMAL = auto()
    DOG = auto()
    CAT = auto()
    FROG = auto()
    BIRD = auto()
    DEER = auto()
    HORSE = auto()

weight = {
    Property.ANIMAL: 1,
    Property.DOG: 1,
    Property.CAT: 1,
    Property.FROG: 1,
    Property.BIRD: 1,
    Property.DEER: 1,
    Property.HORSE: 1,
}


cat_property_set = [Property.ANIMAL, Property.CAT]
dog_property_set = [Property.ANIMAL, Property.DOG]
frog_property_set = [Property.ANIMAL, Property.FROG]
bird_property_set = [Property.ANIMAL, Property.BIRD]
deer_property_set = [Property.ANIMAL, Property.DEER]
horse_property_set = [Property.ANIMAL, Property.HORSE]

property_set_dict = {
    "cat": cat_property_set,
    "dog": dog_property_set,
    "frog": frog_property_set,
    "bird": bird_property_set,
    "deer": deer_property_set,
    "horse": horse_property_set,
}


def measure_similarity(class_a, class_b) -> float:
    property_set_a = property_set_dict[class_a]
    property_set_b = property_set_dict[class_b]
    intersection = [weight[p] for p in property_set_a if p in property_set_b]
    similarity = sum(intersection)
    return similarity


def create_similarity_vector(class_name) -> torch.Tensor:
    num_classes = len(config.CLASS_LIST)
    similarity_vector = torch.zeros((num_classes,), dtype=torch.float)
    for i, other_class_name in enumerate(config.CLASS_LIST):
        similarity_vector[i] = measure_similarity(class_name, other_class_name)

    # normalize similarity vector
    similarity_vector /= similarity_vector.sum()

    return similarity_vector


def create_similarity_matrix(filename="cifar10") -> torch.Tensor:
    similarity_vector_list = []
    for class_name in config.CLASS_LIST:
        similarity_vector = create_similarity_vector(class_name)
        similarity_vector_list.append(similarity_vector)

    similarity_matrix = torch.stack(similarity_vector_list)
  
    if not os.path.isdir(config.SIMILARITY_VECTORS_PATH):
        os.makedirs(config.SIMILARITY_VECTORS_PATH)

    torch.save(
        similarity_matrix,
        os.path.join(config.SIMILARITY_VECTORS_PATH, "{}.th".format(filename)),
    )


if __name__ == "__main__":
    create_similarity_matrix(filename=config.SIMILARITY_VECTORS_FN)