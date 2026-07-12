from pathlib import Path

import numpy as np
import tensorflow as tf

from utils.config import (
    PROCESSED_DATASET_DIR,
    PATCH_SIZE,
    IMAGE_CHANNELS,
    BATCH_SIZE,
    AUTOTUNE,
    SEED
)


class DatasetLoader:
    """
    Crea los datasets de TensorFlow a partir de los
    patches almacenados en formato .npy.
    """

    def __init__(self) -> None:

        self.train_images = PROCESSED_DATASET_DIR / "train" / "images"
        self.train_masks = PROCESSED_DATASET_DIR / "train" / "masks"

        self.validation_images = (
            PROCESSED_DATASET_DIR / "validation" / "images"
        )

        self.validation_masks = (
            PROCESSED_DATASET_DIR / "validation" / "masks"
        )

        self.test_images = (
            PROCESSED_DATASET_DIR / "test" / "images"
        )

        self.test_masks = (
            PROCESSED_DATASET_DIR / "test" / "masks"
        )

    def load_paths(
        self,
        image_dir: Path,
        mask_dir: Path
    ) -> tuple[list[Path], list[Path]]:
        images = sorted(image_dir.glob("*.npy"))
        masks = sorted(mask_dir.glob("*.npy"))
        if len(images) != len(masks):
            raise RuntimeError(
                "La cantidad de imágenes y máscaras no coincide."
            )
        return images, masks

    @staticmethod
    def _load_numpy(
        image_path,
        mask_path
    ):
        image = np.load(image_path.decode())
        mask = np.load(mask_path.decode())
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        mask = np.expand_dims(
            mask,
            axis=-1
        )

        return image, mask

    def parse(
        self,
        image_path,
        mask_path
    ):
        image, mask = tf.numpy_function(
            self._load_numpy,
            [image_path, mask_path],
            [tf.float32, tf.float32]
        )

        image.set_shape(
            (
                PATCH_SIZE[0],
                PATCH_SIZE[1],
                IMAGE_CHANNELS
            )
        )

        mask.set_shape(
            (
                PATCH_SIZE[0],
                PATCH_SIZE[1],
                1
            )
        )
        return image, mask

    def augment(
        self,
        image,
        mask
    ):
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
        return image, mask

    def create_dataset(
        self,
        image_dir: Path,
        mask_dir: Path,
        training: bool
    ) -> tf.data.Dataset:

        image_paths, mask_paths = self.load_paths(
            image_dir,
            mask_dir
        )
        dataset = tf.data.Dataset.from_tensor_slices(

            (
                [str(path) for path in image_paths],
                [str(path) for path in mask_paths]
            )

        )
        dataset = dataset.map(
            self.parse,
            num_parallel_calls=AUTOTUNE
        )

        if training:
            dataset = dataset.shuffle(
                buffer_size=len(image_paths),
                seed=SEED
            )

            dataset = dataset.map(
                self.augment,
                num_parallel_calls=AUTOTUNE
            )

        dataset = dataset.batch(
            BATCH_SIZE
        )

        dataset = dataset.prefetch(
            AUTOTUNE
        )
        return dataset

    def train(self):
        return self.create_dataset(
            self.train_images,
            self.train_masks,
            training=True
        )

    def validation(self):
        return self.create_dataset(
            self.validation_images,
            self.validation_masks,
            training=False
        )

    def test(self):
        return self.create_dataset(
            self.test_images,
            self.test_masks,
            training=False
        )