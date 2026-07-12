import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

def binary_crossentropy_loss():

    return BinaryCrossentropy()

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1.0 - dice

def bce_dice_loss(y_true, y_pred):
    bce = BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)

    return bce + dice

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(
        y_pred,
        1e-7,
        1.0 - 1e-7
    )

    bce = -(
        y_true * tf.math.log(y_pred)
        +
        (1 - y_true) * tf.math.log(1 - y_pred)
    )

    weight = alpha * tf.pow(
        1 - y_pred,
        gamma
    )
    loss = weight * bce

    return tf.reduce_mean(loss)

def get_loss(name="bce_dice"):
    losses = {
        "bce": binary_crossentropy_loss(),
        "dice": dice_loss,
        "bce_dice": bce_dice_loss,
        "focal": focal_loss
    }

    return losses[name]