from pathlib import Path
import tensorflow as tf


# Rutas

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "datasets"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR = ROOT_DIR / "logs"


# Dataset

PATCH_SIZE = (256, 256)
IMAGE_CHANNELS = 4
MASK_CHANNELS = 1
NUM_CLASSES = 1
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# Entrenamiento

BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


# Modelo

MODEL_NAME = "ForestCanopySegmentation"
BACKBONE = "MobileNetV3Small"
ACTIVATION = "sigmoid"


# Exportación

KERAS_MODEL = MODELS_DIR / "forest_segmentation.keras"
TFLITE_MODEL = MODELS_DIR / "forest_segmentation.tflite"


# TensorBoard

TENSORBOARD_LOGS = LOGS_DIR / "tensorboard"


# Checkpoints

CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
BEST_MODEL = CHECKPOINT_DIR / "best_model.keras"

# Optimizador

OPTIMIZER = "adam"

# Función de pérdida

LOSS = "bce_dice"

# Métricas

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "dice",
    "iou"
]

# Callbacks

PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6
VERBOSE = 1