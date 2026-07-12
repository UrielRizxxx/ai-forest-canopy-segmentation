from pathlib import Path
from typing import List, Tuple
import shutil

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split


# Configuración

class DatasetConfig:
    IMAGE_SIZE = (512, 512)
    BATCH_SIZE = 8
    IMAGE_CHANNELS = 3
    MASK_CHANNELS = 1
    VALID_EXTENSIONS = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff"
    )

    TRAIN_SIZE = 0.70
    VALIDATION_SIZE = 0.15
    TEST_SIZE = 0.15
    RANDOM_STATE = 42
    AUTOTUNE = tf.data.AUTOTUNE


config = DatasetConfig()


# Data augmentation

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.RandomRotate90(p=0.5)
])


# Rutas

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "datasets"

RAW_IMAGES = DATASET_DIR / "raw" / "images"
RAW_MASKS = DATASET_DIR / "raw" / "masks"

PROCESSED_IMAGES = DATASET_DIR / "processed" / "images"
PROCESSED_MASKS = DATASET_DIR / "processed" / "masks"

TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_MASKS = DATASET_DIR / "train" / "masks"

VALID_IMAGES = DATASET_DIR / "validation" / "images"
VALID_MASKS = DATASET_DIR / "validation" / "masks"

TEST_IMAGES = DATASET_DIR / "test" / "images"
TEST_MASKS = DATASET_DIR / "test" / "masks"

#---------------------------
# procesamiento del dataset
class DatasetProcessor:

    def __init__(self):

        self.image_size = config.IMAGE_SIZE
        self.batch_size = config.BATCH_SIZE

        self.raw_images = RAW_IMAGES
        self.raw_masks = RAW_MASKS

        self.processed_images = PROCESSED_IMAGES
        self.processed_masks = PROCESSED_MASKS

        self.train_images = TRAIN_IMAGES
        self.train_masks = TRAIN_MASKS

        self.valid_images = VALID_IMAGES
        self.valid_masks = VALID_MASKS

        self.test_images = TEST_IMAGES
        self.test_masks = TEST_MASKS

        self.create_directories()


    def create_directories(self):

        directories = [

            self.processed_images,
            self.processed_masks,

            self.train_images,
            self.train_masks,

            self.valid_images,
            self.valid_masks,

            self.test_images,
            self.test_masks

        ]

        for directory in directories:

            directory.mkdir(
                parents=True,
                exist_ok=True
            )

    def get_files(self, folder: Path):
        files = []
        for extension in config.VALID_EXTENSIONS:

            files.extend(folder.glob(f"*{extension}"))
            files.extend(folder.glob(f"*{extension.upper()}"))

        return sorted(files)

    def verify_dataset(self):
        if not self.raw_images.exists():
            raise FileNotFoundError(
                f"No existe la carpeta:\n{self.raw_images}"
            )

        if not self.raw_masks.exists():
            raise FileNotFoundError(
                f"No existe la carpeta:\n{self.raw_masks}"
            )

        images = self.get_files(self.raw_images)
        masks = self.get_files(self.raw_masks)

        print(f"Imágenes encontradas : {len(images)}")
        print(f"Máscaras encontradas : {len(masks)}")

        if len(images) == 0:
            raise Exception("No hay imágenes.")

        if len(masks) == 0:
            raise Exception("No hay máscaras.")

        if len(images) != len(masks):
            raise Exception(
                "La cantidad de imágenes y máscaras no coincide."
            )

        print("Dataset verificado correctamente.\n")

    def dataset_info(self):

        images = self.get_files(self.raw_images)

        print("-" * 40)
        print("Información del Dataset")
        print("-" * 40)

        print(f"Total imágenes : {len(images)}")
        print(f"Tamaño entrada : {config.IMAGE_SIZE}")
        print(f"Batch Size     : {config.BATCH_SIZE}")
        print(f"Train          : {config.TRAIN_SIZE * 100:.0f}%")
        print(f"Validation     : {config.VALIDATION_SIZE * 100:.0f}%")
        print(f"Test           : {config.TEST_SIZE * 100:.0f}%")

        print("-" * 40)

    def load_image(self, image_path: Path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"No se pudo leer la imagen:\n{image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_AREA
        )

        image = image.astype(np.float32) / 255.0
        return image

    def load_mask(self, mask_path: Path):
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise ValueError(f"No se pudo leer la máscara:\n{mask_path}")

        mask = cv2.resize(
            mask,
            self.image_size,
            interpolation=cv2.INTER_NEAREST
        )
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        return mask

    def save_image(self, image, output_path: Path):
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )
        cv2.imwrite(
            str(output_path),
            image
        )

    def save_mask(self, mask, output_path: Path):
        mask = np.squeeze(mask)
        mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(
            str(output_path),
            mask
        )

    def is_corrupted(self, file_path: Path):

        try:
            image = cv2.imread(str(file_path))
            if image is None:
                return True
            if image.size == 0:
                return True
            return False

        except Exception:
            return True

    def remove_corrupted(self):
        print("\nBuscando archivos corruptos...\n")
        removed = 0
        images = self.get_files(self.raw_images)
        masks = self.get_files(self.raw_masks)

        for image in images:

            if self.is_corrupted(image):
                print(f"Imagen eliminada: {image.name}")
                image.unlink()
                removed += 1

        for mask in masks:

            if self.is_corrupted(mask):
                print(f"Máscara eliminada: {mask.name}")
                mask.unlink()
                removed += 1

        print(f"\nArchivos eliminados: {removed}")

    def convert_to_png(self):
        print("\nConvirtiendo imágenes a PNG...")

        images = self.get_files(self.raw_images)
        masks = self.get_files(self.raw_masks)

        for image_path in images:

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            output = self.processed_images / f"{image_path.stem}.png"

            cv2.imwrite(str(output), image)

        for mask_path in masks:

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE
            )

            if mask is None:
                continue

            output = self.processed_masks / f"{mask_path.stem}.png"

            cv2.imwrite(str(output), mask)

        print("Conversión finalizada.")

    def resize_dataset(self):

        print("\nRedimensionando imágenes...")

        images = self.get_files(self.processed_images)
        masks = self.get_files(self.processed_masks)

        for image_path in images:

            image = cv2.imread(str(image_path))

            image = cv2.resize(
                image,
                self.image_size,
                interpolation=cv2.INTER_AREA
            )

            cv2.imwrite(str(image_path), image)

        for mask_path in masks:

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE
            )

            mask = cv2.resize(
                mask,
                self.image_size,
                interpolation=cv2.INTER_NEAREST
            )

            cv2.imwrite(str(mask_path), mask)

        print("Redimensionamiento finalizado.")

    def binarize_masks(self):

        print("\nBinarizando máscaras...")

        masks = self.get_files(self.processed_masks)

        for mask_path in masks:

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE
            )

            _, mask = cv2.threshold(
                mask,
                127,
                255,
                cv2.THRESH_BINARY
            )

            cv2.imwrite(str(mask_path), mask)

        print("Máscaras listas.")

    def preprocess_dataset(self):

        self.remove_corrupted()
        self.convert_to_png()
        self.resize_dataset()
        self.binarize_masks()

        print("\nPreprocesamiento terminado.")

    def split_dataset(self):
        print("\nDividiendo dataset...")

        images = self.get_files(self.processed_images)
        masks = self.get_files(self.processed_masks)

        image_dict = {image.stem: image for image in images}
        mask_dict = {mask.stem: mask for mask in masks}

        common_names = sorted(
            list(set(image_dict.keys()) & set(mask_dict.keys()))
        )

        image_files = [image_dict[name] for name in common_names]
        mask_files = [mask_dict[name] for name in common_names]

        train_images, temp_images, train_masks, temp_masks = train_test_split(
            image_files,
            mask_files,
            test_size=(1 - config.TRAIN_SIZE),
            random_state=config.RANDOM_STATE,
            shuffle=True
        )

        valid_images, test_images, valid_masks, test_masks = train_test_split(
            temp_images,
            temp_masks,
            test_size=0.5,
            random_state=config.RANDOM_STATE,
            shuffle=True
        )

        self.copy_files(train_images, self.train_images)
        self.copy_files(train_masks, self.train_masks)

        self.copy_files(valid_images, self.valid_images)
        self.copy_files(valid_masks, self.valid_masks)

        self.copy_files(test_images, self.test_images)
        self.copy_files(test_masks, self.test_masks)

        print("\nDataset dividido correctamente.\n")

        print(f"Train      : {len(train_images)}")
        print(f"Validation : {len(valid_images)}")
        print(f"Test       : {len(test_images)}")

    def copy_files(self, files, destination):

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        for file in files:

            shutil.copy2(
                file,
                destination / file.name
            )

    def create_tf_dataset(self, image_folder, mask_folder):

        image_paths = self.get_files(image_folder)
        mask_paths = self.get_files(mask_folder)

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                [str(p) for p in image_paths],
                [str(p) for p in mask_paths]
            )
        )

        dataset = dataset.map(
            self.parse_data,
            num_parallel_calls=config.AUTOTUNE
        )

        dataset = dataset.batch(
            self.batch_size
        )

        dataset = dataset.cache()

        dataset = dataset.prefetch(
            config.AUTOTUNE
        )

        return dataset
    
    def read_image(self, image_path):

        image = tf.io.read_file(image_path)

        image = tf.image.decode_png(
            image,
            channels=3
        )

        image = tf.image.resize(
            image,
            self.image_size
        )

        image = tf.cast(
            image,
            tf.float32
        ) / 255.0

        return image

    def read_mask(self, mask_path):

        mask = tf.io.read_file(mask_path)

        mask = tf.image.decode_png(
            mask,
            channels=1
        )

        mask = tf.image.resize(
            mask,
            self.image_size,
            method="nearest"
        )

        mask = tf.where(
            mask > 127,
            1.0,
            0.0
        )

        return tf.cast(
            mask,
            tf.float32
        )

    def parse_data(self, image_path, mask_path):
        image = self.read_image(image_path)
        mask = self.read_mask(mask_path)
        return image, mask

    def augment_dataset(self, dataset):

        def augment(image, mask):

            if tf.random.uniform(()) > 0.5:

                image = tf.image.flip_left_right(image)
                mask = tf.image.flip_left_right(mask)

            if tf.random.uniform(()) > 0.5:

                image = tf.image.random_brightness(
                    image,
                    max_delta=0.2
                )

            return image, mask

        return dataset.map(
            augment,
            num_parallel_calls=config.AUTOTUNE
        )

    def prepare(self):
        self.verify_dataset()
        self.clear_processed_data()
        self.remove_corrupted()
        self.convert_to_png()
        self.resize_dataset()
        self.binarize_masks()
        self.split_dataset()

        train_dataset = self.create_tf_dataset(
            self.train_images,
            self.train_masks
        )

        valid_dataset = self.create_tf_dataset(
            self.valid_images,
            self.valid_masks
        )

        test_dataset = self.create_tf_dataset(
            self.test_images,
            self.test_masks
        )

        train_dataset = self.augment_dataset(
            train_dataset
        )

        self.summary()

        return (
            train_dataset,
            valid_dataset,
            test_dataset
        )
    
    def clear_processed_data(self):

        folders = [

            self.processed_images,
            self.processed_masks,

            self.train_images,
            self.train_masks,

            self.valid_images,
            self.valid_masks,

            self.test_images,
            self.test_masks

        ]

        for folder in folders:

            if not folder.exists():
                continue

            for file in folder.iterdir():

                if file.is_file():
                    file.unlink()

        print("Carpetas limpiadas.")

    def visualize_sample(self, dataset):

        import matplotlib.pyplot as plt

        for images, masks in dataset.take(1):
            image = images[0].numpy()
            mask = masks[0].numpy().squeeze()
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Imagen")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap="gray")
            plt.title("Máscara")
            plt.axis("off")

            plt.show()

    def summary(self):

        train_images = len(self.get_files(self.train_images))
        valid_images = len(self.get_files(self.valid_images))
        test_images = len(self.get_files(self.test_images))

        total = train_images + valid_images + test_images

        print("\n" + "=" * 50)
        print("Resumen del Dataset")
        print("=" * 50)
        print(f"Total        : {total}")
        print(f"Train        : {train_images}")
        print(f"Validation   : {valid_images}")
        print(f"Test         : {test_images}")
        print("=" * 50)

if __name__ == "__main__":
    dataset = DatasetProcessor()
    train_dataset, valid_dataset, test_dataset = dataset.prepare()
    dataset.visualize_sample(train_dataset)