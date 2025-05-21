import gc
import json
import logging
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import zarr
from joblib import Parallel, delayed

from src.config.config import Config

from .registry import get_detector

logger = logging.getLogger(__name__)


def process_tile(algorithm, tile, valid_tiles):
    # Inicializa o detector dentro da função para evitar problemas de pickle
    detector = get_detector(algorithm)

    tile_name = Path(tile["name"]).stem
    if not valid_tiles.get(tile_name, False):
        return tile_name, None, None, False, tile["coordinates"]

    img_path = Path(tile["path"]) / tile["name"]  # Corrigido o caminho da imagem
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error(f"[!] Erro ao abrir imagem: {img_path}")
        return tile_name, None, None, False, tile["coordinates"]

    keypoints, descriptors = detector.detect_and_compute(img)

    if keypoints and len(keypoints) > 4:
        kp_array = np.array(
            [
                [
                    kp.pt[0],
                    kp.pt[1],
                    kp.size,
                    kp.angle,
                    kp.response,
                    kp.octave,
                    kp.class_id,
                ]
                for kp in keypoints
            ],
            dtype=np.float32,
        )
        descriptors = descriptors.astype(np.uint8)
        registered = True
    else:
        kp_array = np.zeros((0, 7), dtype=np.float32)
        descriptors = np.zeros((0, 32), dtype=np.uint8)
        registered = False

    # Executa limpeza
    del detector, img, keypoints
    gc.collect()

    # Retorna sempre 5 valores (pode ter None, mas garante desempacotamento correto)
    return tile_name, kp_array, descriptors, registered, tile["coordinates"]


def save_to_zarr(zarr_store, results):
    for tile_name, kp_array, descriptors, registered, coords in results:
        if kp_array is None or descriptors is None:
            logger.info(f"[!] Tile '{tile_name}' ignorado (keypoints/descriptors None)")
            continue  # Ignora o tile se os keypoints ou descritores estiverem ausentes

        group = zarr_store.create_group(tile_name, overwrite=True)
        zarr.array(
            kp_array,
            store=group.store,
            path=f"{group.path}/keypoints",
            chunks=kp_array.shape,
            dtype=np.float32,
        )
        zarr.array(
            descriptors,
            store=group.store,
            path=f"{group.path}/descriptors",
            chunks=descriptors.shape,
            dtype=np.uint8,
        )
        group.attrs["registered"] = registered
        group.attrs["coordinates"] = coords


def detect():
    start = perf_counter()

    with open(Config.METADATA_FILE) as f:
        dataset_metadata = json.load(f)

    with open(Config.VALID_TILES_FILE) as f:
        valid_tiles = json.load(f)

    tiles = dataset_metadata

    if Config.DETECTION_N_JOBS == 1:
        results = [
            process_tile(Config.DETECTION_ALGORITHM, tile, valid_tiles)
            for tile in tiles
        ]
    else:
        results = Parallel(n_jobs=Config.DETECTION_N_JOBS)(
            delayed(process_tile)(Config.DETECTION_ALGORITHM, tile, valid_tiles)
            for tile in tiles
        )

    zarr_store = zarr.open(Config.KEYPOINTS_ZARR_STORE, mode="w")
    save_to_zarr(zarr_store, results)

    end = perf_counter()
    logger.info(f"Features extraídas e salvas em {Config.KEYPOINTS_ZARR_STORE}")
    logger.info(f"Tempo total: {end - start:.2f} segundos")


if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.INFO)
    detect()
