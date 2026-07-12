from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils.config import (
    IMAGE_SIZE,
    KERAS_MODEL
)

def load_segmentation_model():
    model = tf.keras.models.load_model(
        KERAS_MODEL,
        compile=False
    )

    return model

def load_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(
            f"No se pudo leer la imagen:\n{image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def predict_mask(model, image):
    prediction = model.predict(
        image,
        verbose=0
    )
    prediction = (prediction > 0.5).astype(np.uint8)
    return prediction[0]

def save_prediction(mask, output_path):
    mask = np.squeeze(mask)
    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(
        str(output_path),
        mask
    )
    print(f"Máscara guardada en:\n{output_path}")

def visualize_prediction(image, mask):
    image = np.squeeze(image)
    mask = np.squeeze(mask)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Imagen")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Predicción")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    image_path = Path("datasets/test/images/sample.png")
    output_path = Path("results/predictions/prediction.png")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model = load_segmentation_model()
    image = load_image(image_path)
    prediction = predict_mask(
        model,
        image
    )

    save_prediction(
        prediction,
        output_path
    )

    visualize_prediction(
        image,
        prediction
    )


if __name__ == "__main__":
    main()