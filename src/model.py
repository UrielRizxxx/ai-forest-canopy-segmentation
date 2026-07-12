import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dropout,
    Multiply,
    SeparableConv2D,
    UpSampling2D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV3Small

from utils.config import (
    PATCH_SIZE,
    IMAGE_CHANNELS,
    BACKBONE
)


def conv_block(inputs, filters):
    """
    Bloque convolucional del decoder.
    """

    x = SeparableConv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        depthwise_initializer=HeNormal(),
        pointwise_initializer=HeNormal(),
        depthwise_regularizer=l2(1e-4),
        pointwise_regularizer=l2(1e-4)
    )(inputs)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = SeparableConv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        depthwise_initializer=HeNormal(),
        pointwise_initializer=HeNormal(),
        depthwise_regularizer=l2(1e-4),
        pointwise_regularizer=l2(1e-4)
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def attention_gate(skip, gating, filters):
    """
    Attention Gate.
    """
    theta = Conv2D(
        filters,
        kernel_size=1,
        padding="same"
    )(skip)

    phi = Conv2D(
        filters,
        kernel_size=1,
        padding="same"
    )(gating)

    attention = Add()([
        theta,
        phi
    ])

    attention = Activation("relu")(attention)
    attention = Conv2D(
        filters=1,
        kernel_size=1,
        activation="sigmoid"
    )(attention)

    return Multiply()([
        skip,
        attention
    ])


def decoder_block(inputs, skip, filters):
    """
    Bloque decoder.
    """
    x = UpSampling2D(
        size=(2, 2),
        interpolation="bilinear"
    )(inputs)

    skip = attention_gate(
        skip,
        x,
        filters
    )

    x = Concatenate()([
        x,
        skip
    ])

    x = Dropout(0.20)(x)
    x = conv_block(
        x,
        filters
    )
    return x

def build_encoder():
    """
    Construye el encoder utilizando MobileNetV3Small
    con un adaptador espectral de 4→3 canales.
    """

    inputs = tf.keras.Input(
        shape=(*PATCH_SIZE, IMAGE_CHANNELS),
        name="multispectral_input"
    )

    x = Conv2D(
        filters=3,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer=HeNormal(),
        name="spectral_adapter"
    )(inputs)

    if BACKBONE != "MobileNetV3Small":
        raise ValueError(
            f"Backbone '{BACKBONE}' no soportado."
        )

    backbone = MobileNetV3Small(
        input_shape=(*PATCH_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    backbone.trainable = False

    layer_names = [
        "re_lu",
        "expanded_conv_project_bn",
        "expanded_conv_3_project_bn",
        "expanded_conv_8_project_bn",
        "activation_17"
    ]

    outputs = [
        backbone.get_layer(name).output
        for name in layer_names
    ]

    encoder = tf.keras.Model(
        inputs=backbone.input,
        outputs=outputs,
        name="encoder"
    )

    s1, s2, s3, s4, bridge = encoder(x)

    bridge = Dropout(
        0.30,
        name="bridge_dropout"
    )(bridge)

    return inputs, (s1, s2, s3, s4), bridge

def build_model():
    """
    Construye el modelo completo de segmentación
    basado en Attention U-Net con MobileNetV3Small.
    """
    inputs, skips, bridge = build_encoder()
    s1, s2, s3, s4 = skips

    x = decoder_block(
        bridge,
        s4,
        192
    )

    x = decoder_block(
        x,
        s3,
        128
    )

    x = decoder_block(
        x,
        s2,
        64
    )

    x = decoder_block(
        x,
        s1,
        32
    )

    outputs = Conv2D(
        filters=1,
        kernel_size=1,
        activation="sigmoid",
        padding="same",
        kernel_initializer=HeNormal(),
        name="segmentation_output"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="ForestCanopySegmentation"
    )
    return model

def unfreeze_encoder(model):
    encoder = model.get_layer("encoder")
    encoder.trainable = True
    for layer in encoder.layers[:-30]:
        layer.trainable = False

    return model


def count_parameters(model):
    """
    Imprime el número de parámetros del modelo.
    """
    trainable = np.sum([
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights
    ])

    non_trainable = np.sum([
        tf.keras.backend.count_params(w)
        for w in model.non_trainable_weights
    ])

    total = trainable + non_trainable
    print()
    print("=" * 50)
    print(f"Trainable      : {trainable:,}")
    print(f"Non Trainable  : {non_trainable:,}")
    print(f"Total          : {total:,}")
    print("=" * 50)
    print()


if __name__ == "__main__":
    model = build_model()
    model.summary(
        line_length=120,
        expand_nested=True
    )

    count_parameters(model)