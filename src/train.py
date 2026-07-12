import json

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)

from utils.config import (
    BEST_MODEL,
    CHECKPOINT_DIR,
    EPOCHS,
    LEARNING_RATE,
    LOSS,
    MIN_LEARNING_RATE,
    PATIENCE,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    RESULTS_DIR,
    TENSORBOARD_LOGS,
    VERBOSE
)
from dataset_old import DatasetProcessor
from losses import get_loss
from metrics import get_metrics
from model import build_model

def create_callbacks():

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TENSORBOARD_LOGS.mkdir(
        parents=True,
        exist_ok=True
    )

    callbacks = [

        ModelCheckpoint(
            filepath=BEST_MODEL,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=VERBOSE
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=VERBOSE
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=VERBOSE
        ),

        TensorBoard(
            log_dir=TENSORBOARD_LOGS
        )

    ]

    return callbacks

def compile_model():
    model = build_model()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss=get_loss(LOSS),
        metrics=get_metrics()
    )

    return model

def train_model(model, train_dataset, valid_dataset):

    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=EPOCHS,
        callbacks=create_callbacks(),
        verbose=VERBOSE
    )

    return history

def evaluate_model(model, test_dataset):
    results = model.evaluate(
        test_dataset,
        verbose=VERBOSE
    )

    print("\nResultados de evaluación")
    for metric, value in zip(model.metrics_names, results):
        print(f"{metric}: {value:.4f}")

    return results

def save_history(history):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    history_path = RESULTS_DIR / "history.json"
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=4)

    print(f"\nHistorial guardado en:\n{history_path}")

def plot_history(history):
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    metrics = history.history

    for metric, values in metrics.items():
        plt.figure(figsize=(8, 5))
        plt.plot(values)
        val_metric = f"val_{metric}"
        if val_metric in metrics:
            plt.plot(metrics[val_metric])
        
        plt.title(metric.replace("_", " ").title())
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend(["Train", "Validation"])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{metric}.png")
        plt.close()

    print("\nGráficas guardadas correctamente.")

def main():
    processor = DatasetProcessor()
    train_dataset, valid_dataset, test_dataset = processor.prepare()
    model = compile_model()
    history = train_model(
        model,
        train_dataset,
        valid_dataset
    )

    evaluate_model(
        model,
        test_dataset
    )

    save_history(history)
    plot_history(history)
    print("\nEntrenamiento finalizado correctamente.")

if __name__ == "__main__":
    main()