from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


class Rasterizer:
    """
    Convierte las geometrías del bosque en una máscara binaria.
    """
    def __init__(
        self,
        geometries: gpd.GeoDataFrame
    ) -> None:

        self.geometries = geometries

    def get_geometries_for_image(
        self,
        image: rasterio.io.DatasetReader
    ) -> gpd.GeoDataFrame:

        """
        Obtiene únicamente las geometrías que intersectan
        con la imagen.
        """

        image_bounds = image.bounds

        selected = self.geometries.cx[
            image_bounds.left:image_bounds.right,
            image_bounds.bottom:image_bounds.top
        ]

        return selected

    def rasterize_mask(
        self,
        image: rasterio.io.DatasetReader
    ) -> np.ndarray:
        """
        Genera la máscara binaria para una imagen.
        """
        geometries = self.get_geometries_for_image(
            image
        )

        if geometries.empty:

            return np.zeros(
                (
                    image.height,
                    image.width
                ),
                dtype=np.uint8
            )
        mask = rasterize(

            shapes=[
                (geometry, 1)
                for geometry in geometries.geometry
            ],
            out_shape=(
                image.height,
                image.width
            ),
            transform=image.transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )

        return mask

    def save_mask(
        self,
        mask: np.ndarray,
        output_path: Path
    ) -> None:

        """
        Guarda la máscara en formato NumPy.
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            output_path,
            mask
        )