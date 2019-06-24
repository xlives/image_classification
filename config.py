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

CHECKPOINTS_PATH = "trained_models/checkpoints_normal"
CLEAN_CHECKPOINTS_PATH = True

BATCH_SIZE = 128
VALIDATION_SIZE = 0.1

NUM_EPOCHS = 200
PATIENCE = 10
LEARNING_RATE = 0.001

USE_PROGRESSIVE_LEARNING = True
T_INITIAL = 0.1
DECAY_RATE = 0.01
