from pathlib import Path
import random
import shutil

from utils.config import (
    PROCESSED_DATASET_DIR,
    TRAIN_SIZE,
    VALIDATION_SIZE,
    TEST_SIZE,
    SEED
)


class DatasetSplitter:
    """
    Divide el dataset en entrenamiento,
    validación y prueba.
    """

    def __init__(self) -> None:
        random.seed(SEED)
        self.images_dir = PROCESSED_DATASET_DIR / "images"
        self.masks_dir = PROCESSED_DATASET_DIR / "masks"

    def _create_directories(self) -> None:

        for split in (
            "train",
            "validation",
            "test"
        ):

            (PROCESSED_DATASET_DIR / split / "images").mkdir(
                parents=True,
                exist_ok=True
            )

            (PROCESSED_DATASET_DIR / split / "masks").mkdir(
                parents=True,
                exist_ok=True
            )

    def _group_images(self) -> dict:
        groups = {}
        for image_path in sorted(
            self.images_dir.glob("*.npy")
        ):
            name = image_path.stem
            original = "_".join(name.split("_")[:-2])
            groups.setdefault(original, []).append(image_path)
        return groups

    def _split_groups(
        self,
        groups: dict
    ):
        keys = list(groups.keys())
        random.shuffle(keys)
        total = len(keys)
        train_end = int(total * TRAIN_SIZE)
        valid_end = train_end + int(
            total * VALIDATION_SIZE
        )

        return (
            keys[:train_end],
            keys[train_end:valid_end],
            keys[valid_end:]
        )

    def _copy_group(
        self,
        group_keys,
        split
    ):
        for key in group_keys:
            for image in self.groups[key]:
                mask = self.masks_dir / image.name

                shutil.copy2(
                    image,
                    PROCESSED_DATASET_DIR /
                    split /
                    "images" /
                    image.name
                )

                shutil.copy2(
                    mask,
                    PROCESSED_DATASET_DIR /
                    split /
                    "masks" /
                    mask.name
                )

    def split(self):
        self._create_directories()
        self.groups = self._group_images()
        train, validation, test = self._split_groups(
            self.groups
        )

        self._copy_group(
            train,
            "train"
        )

        self._copy_group(
            validation,
            "validation"
        )

        self._copy_group(
            test,
            "test"
        )
        print()
        print("=" * 60)
        print("Dataset dividido correctamente")
        print()
        print(f"Train: {len(train)} escenas")
        print(f"Validation: {len(validation)} escenas")
        print(f"Test: {len(test)} escenas")
        print("=" * 60)


if __name__ == "__main__":
    DatasetSplitter().split()