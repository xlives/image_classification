ALL_CLASS_LIST = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
CLASS_LIST = ["bird", "cat", "deer", "dog", "frog", "horse"]

DATA_PATH = "data/"
FIGURES_PATH = "figures/"

SIMILARITY_VECTORS_PATH = "similarity_vectors"
SIMILARITY_VECTORS_FN = "cifar10"

CHECKPOINTS_PATH = "trained_models/checkpoints_wp"
CLEAN_CHECKPOINTS_PATH = True

AUGMENT_DATA = True
BATCH_SIZE = 128
VALIDATION_SIZE = 0.1

NUM_EPOCHS = 200
PATIENCE = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

USE_PROGRESSIVE_LEARNING = True
T_INITIAL = 0.5
DECAY_RATE = 0.01
