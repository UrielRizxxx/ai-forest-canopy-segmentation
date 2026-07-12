import tensorflow as tf

from utils.config import (
    BEST_MODEL,
    TFLITE_MODEL
)

def load_model():
    model = tf.keras.models.load_model(
        BEST_MODEL,
        compile=False
    )
    return model

def convert_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(
        model
    )
    tflite_model = converter.convert()
    return tflite_model

def save_model(tflite_model):
    TFLITE_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(TFLITE_MODEL, "wb") as file:
        file.write(tflite_model)
    print(f"Modelo guardado en:\n{TFLITE_MODEL}")

def verify_model():
    interpreter = tf.lite.Interpreter(
        model_path=str(TFLITE_MODEL)
    )
    interpreter.allocate_tensors()

    print("Modelo TensorFlow Lite verificado correctamente.")

def main():
    model = load_model()
    tflite_model = convert_model(model)
    save_model(tflite_model)
    verify_model()
    print("\nConversión finalizada correctamente.")


if __name__ == "__main__":
    main()