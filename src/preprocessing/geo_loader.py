from pathlib import Path
from typing import Any

import geopandas as gpd
import rasterio


class GeoLoader:
    """
    Carga y valida el dataset Open-Canopy.
    """
    def __init__(
        self,
        dataset_path: Path
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "2023" / "spot"
        self.parquet_path = self.dataset_path / "forest_mask.parquet"
        self.geojson_path = self.dataset_path / "geometries.geojson"

    def validate_dataset(self) -> None:

        """
        Verifica que todos los archivos necesarios existan.
        """

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"No existe el dataset:\n{self.dataset_path}"
            )

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"No existe la carpeta de imágenes:\n{self.images_dir}"
            )

        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"No existe:\n{self.parquet_path}"
            )

        if not self.geojson_path.exists():
            raise FileNotFoundError(
                f"No existe:\n{self.geojson_path}"
            )

    def get_image_paths(self) -> list[Path]:

        """
        Devuelve todas las imágenes SPOT disponibles.
        """

        image_paths = sorted(
            self.images_dir.glob("*.tif")
        )

        if not image_paths:
            raise RuntimeError(
                "No se encontraron imágenes TIFF."
            )

        return image_paths

    def load_geometries(self) -> gpd.GeoDataFrame:
        """
        Carga las geometrías del bosque.
        """
        return gpd.read_parquet(
            self.parquet_path
        )

    def load_image(
        self,
        image_path: Path
    ) -> rasterio.io.DatasetReader:

        """
        Abre un GeoTIFF.
        """

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        return rasterio.open(image_path)

    def get_image_info(
        self,
        image_path: Path
    ) -> dict[str, Any]:

        """
        Obtiene la información de un GeoTIFF.
        """

        with rasterio.open(image_path) as src:

            return {
                "filename": image_path.name,
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "crs": src.crs,
                "bounds": src.bounds,
                "resolution": src.res,
                "transform": src.transform,
                "dtype": src.dtypes
            }