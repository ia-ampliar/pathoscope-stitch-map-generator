import cv2
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from pathlib import Path
from typing import Optional, List, Union
import time
import logging

from src.config.config import Config

def imread(filepath: Union[str, Path]) -> Optional[np.ndarray]:
    """Reads an image from the given filepath using OpenCV."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Image file not found: {filepath}")
    return cv2.imread(str(filepath))


def average_images(image_folder: Union[str, Path]) -> np.ndarray:
    """
    Reads all JPG images from the specified folder, calculates the pixel-wise average,
    and returns the average image.

    Args:
        image_folder (Union[str, Path]): Path to the folder containing images.

    Returns:
        np.ndarray: The average image.

    Raises:
        ValueError: If no images are found or if the first image cannot be read.
    """
    image_folder = Path(image_folder)
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    images: List[Path] = [
        f for ext in Config.SUPPORTED_IMAGE_EXTENSIONS
        for f in image_folder.glob(f"*{ext}")
    ]

    if not images:
        raise ValueError(f"No images found in folder: {image_folder}")

    # Reads images in parallel
    with ThreadPoolExecutor() as executor:
        images_data = list(executor.map(imread, images))

    # Filter non readed images
    images_data = [img for img in images_data if img is not None]

    if not images_data:
        raise ValueError("No images could be read.")

    # Read the first image to get dimensions
    first_image: Optional[np.ndarray] = imread(images[0])
    if first_image is None:
        raise ValueError(f"Error reading the first image: {images[0]}")

    avg_image: np.ndarray = np.zeros(first_image.shape, np.float64)

    for img in images_data:
        if img.shape != first_image.shape:
            img = cv2.resize(img, (first_image.shape[1], first_image.shape[0]))
        avg_image += img.astype(np.float64)

    avg_image /= len(images)
    avg_image = np.array(np.round(avg_image), dtype=np.uint8)

    return avg_image


def normalize_image(
    image: np.ndarray,
    normalization_image: np.ndarray,
    brightness_factor: float = Config.DEFAULT_BRIGHTNESS_FACTOR,
    epsilon: float = Config.DEFAULT_EPSILON,
) -> np.ndarray:
    """
    Normalizes an image using a normalization image.

    Args:
        image (np.ndarray): The image to normalize.
        normalization_image (np.ndarray): The normalization image.
        brightness_factor (float): Brightness adjustment factor.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        np.ndarray: The normalized image.
    """
    image_f: np.ndarray = image.astype(np.float32)
    norm_f: np.ndarray = normalization_image.astype(np.float32)
    normalized: np.ndarray = image_f / (norm_f + epsilon) * 255.0
    normalized *= brightness_factor
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    return normalized


def move_tiles_to_raw(tiles_dir: Union[str, Path]) -> None:
    """
    Moves all loose files within the 'tiles' directory to the 'raw' subfolder.

    Args:
        tiles_dir (Union[str, Path]): Path to the directory containing the tiles.

    Raises:
        FileNotFoundError: If the tiles directory does not exist.
    """
    
    if not tiles_dir.exists():
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")

    Config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in tiles_dir.iterdir():
        if file_path.is_file():
            shutil.move(str(file_path), str(Config.RAW_DIR / file_path.name))

def process_and_save(img_path, norm_image, brightness_factor):
    img = imread(img_path)
    if img is None:
        return
    normalized = normalize_image(img, norm_image, brightness_factor=brightness_factor)
    cv2.imwrite(str(Config.NORMALIZED_DIR / img_path.name), normalized)

def run_preprocessing(
    tiles_dir: Union[str, Path],
    brightness_factor: float = Config.DEFAULT_BRIGHTNESS_FACTOR,
) -> Path:
    """
    Runs the complete preprocessing pipeline:
      1. Moves original files to the 'raw' directory.
      2. Calculates the average image for normalization.
      3. Normalizes each image and saves it to the 'normalized' directory.

    Args:
        tiles_dir (Union[str, Path]): Path to the directory containing the tiles.
        brightness_factor (float): Brightness adjustment factor.

    Returns:
        Path: Path to the directory with normalized images.

    Raises:
        ValueError: If no images are found in the 'raw' directory.
    """

    # Create directories if they don't exist
    Config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    Config.AVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    Config.NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

    # Get all images in the raw directory
    image_paths: List[Path] = [
        f for ext in Config.SUPPORTED_IMAGE_EXTENSIONS
        for f in Config.TILES_DIR.glob(f"*{ext}")
    ]

    if not image_paths:
        raise ValueError(f"No images found in directory: {Config.TILES_DIR}")

    # Calculate the normalization image (average image)
    norm_image: np.ndarray = average_images(Config.TILES_DIR)
    norm_image_path: Path = Config.AVERAGE_DIR / "normalize.jpg"
    cv2.imwrite(str(norm_image_path), norm_image)

    # Normalize and save images using joblib
    Parallel(n_jobs=-1)(
        delayed(process_and_save)(img_path, norm_image, brightness_factor)
        for img_path in image_paths
    )

    return Config.NORMALIZED_DIR

if __name__ == "__main__":

    start_time = time.perf_counter()
    
    run_preprocessing(Config.TILES_DIR)

    # Configura o logging básico
    logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.DEBUG)

    # Suprime avisos do matplotlib
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")