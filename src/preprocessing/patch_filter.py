from pathlib import Path
import random

import numpy as np


class PatchFilter:

    """
    Filtra los patches según la cantidad de bosque.
    """

    def __init__(
        self,
        minimum_forest_ratio: float,
        empty_patch_ratio: float
    ) -> None:

        self.minimum_forest_ratio = minimum_forest_ratio
        self.empty_patch_ratio = empty_patch_ratio

    def forest_ratio(
        self,
        mask: np.ndarray
    ) -> float:

        """
        Calcula el porcentaje de bosque.
        """

        forest_pixels = np.count_nonzero(mask)

        total_pixels = mask.size

        return forest_pixels / total_pixels

    def filter(
        self,
        image_patches: list[np.ndarray],
        mask_patches: list[np.ndarray],
        coordinates: list[tuple[int, int]]
    ) -> tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[tuple[int, int]]
    ]:

        """
        Filtra los patches conservando todos los que contienen
        bosque y una pequeña muestra de los vacíos.
        """

        forest_samples = []

        empty_samples = []

        for image, mask, coordinate in zip(
            image_patches,
            mask_patches,
            coordinates
        ):

            sample = (
                image,
                mask,
                coordinate
            )

            if self.forest_ratio(mask) >= self.minimum_forest_ratio:

                forest_samples.append(sample)

            else:

                empty_samples.append(sample)

        number_empty = int(
            len(empty_samples) * self.empty_patch_ratio
        )

        empty_samples = random.sample(
            empty_samples,
            k=min(number_empty, len(empty_samples))
        )

        samples = forest_samples + empty_samples

        images = [sample[0] for sample in samples]

        masks = [sample[1] for sample in samples]

        coordinates = [sample[2] for sample in samples]

        return (
            images,
            masks,
            coordinates
        )

    def statistics(
        self,
        original: int,
        remaining: int
    ) -> dict[str, int]:

        """
        Devuelve estadísticas del filtrado.
        """

        return {
            "original_patches": original,
            "remaining_patches": remaining,
            "removed_patches": original - remaining
        }