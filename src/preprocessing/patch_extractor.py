from pathlib import Path

import numpy as np


class PatchExtractor:

    """
    Extrae patches de imágenes y máscaras.
    """

    def __init__(
        self,
        patch_size: int,
        stride: int
    ) -> None:

        self.patch_size = patch_size
        self.stride = stride

    def apply_padding(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        """
        Aplica padding para cubrir completamente la imagen.
        """

        height, width = image.shape[:2]

        pad_height = (
            self.patch_size - (height % self.patch_size)
        ) % self.patch_size

        pad_width = (
            self.patch_size - (width % self.patch_size)
        ) % self.patch_size

        image = np.pad(
            image,
            (
                (0, pad_height),
                (0, pad_width),
                (0, 0)
            ),
            mode="constant"
        )

        mask = np.pad(
            mask,
            (
                (0, pad_height),
                (0, pad_width)
            ),
            mode="constant"
        )

        return image, mask

    def extract(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int]]]:

        """
        Extrae patches de una imagen y su máscara.
        """

        image, mask = self.apply_padding(
            image,
            mask
        )

        image_patches = []

        mask_patches = []

        coordinates = []

        height, width = image.shape[:2]

        for y in range(
            0,
            height - self.patch_size + 1,
            self.stride
        ):

            for x in range(
                0,
                width - self.patch_size + 1,
                self.stride
            ):

                image_patch = image[
                    y:y + self.patch_size,
                    x:x + self.patch_size
                ]

                mask_patch = mask[
                    y:y + self.patch_size,
                    x:x + self.patch_size
                ]

                image_patches.append(
                    image_patch
                )

                mask_patches.append(
                    mask_patch
                )

                coordinates.append(
                    (x, y)
                )

        return (
            image_patches,
            mask_patches,
            coordinates
        )

    def save_patch(
        self,
        image_patch: np.ndarray,
        mask_patch: np.ndarray,
        image_path: Path,
        mask_path: Path
    ) -> None:

        """
        Guarda un patch de imagen y su máscara.
        """

        image_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        mask_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            image_path,
            image_patch
        )

        np.save(
            mask_path,
            mask_patch
        )