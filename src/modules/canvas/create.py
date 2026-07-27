import re
import time
from pathlib import Path
from typing import Tuple

import cv2
import dask.array as da
import numpy as np

from src.config.config import Config
from src.utils.coordinates import extract_coordinates


def calculate_canvas_size(
    tile_size: Tuple[int, int], grid_size: Tuple[int, int], gap: int
) -> Tuple[int, int]:
    tile_width, tile_height = tile_size
    n_cols, n_rows = grid_size
    total_width = n_cols * tile_width + (n_cols + 1) * gap
    total_height = n_rows * tile_height + (n_rows + 1) * gap
    return total_width, total_height


def create_blank_canvas_shape_from_tiles(
    resized_dir: Path, gap: int
) -> Tuple[int, int, int]:
    image_files = list(resized_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {resized_dir}")

    pattern = re.compile(Config.COORDINATES_PATTERN)
    coords = [extract_coordinates(f.name, pattern) for f in image_files]
    xs, ys = zip(*coords)
    max_x = max(xs)
    max_y = max(ys)
    # Coordenadas são base-0, então a quantidade de colunas/linhas é max + 1
    grid_size = (max_x + 1, max_y + 1)

    sample_img = cv2.imread(str(image_files[0]))
    if sample_img is None:
        raise FileNotFoundError(f"Erro ao carregar imagem de amostra: {image_files[0]}")
    tile_height, tile_width = sample_img.shape[:2]

    canvas_width, canvas_height = calculate_canvas_size(
        (tile_width, tile_height), grid_size, gap
    )

    return canvas_height, canvas_width, 3  # (altura, largura, canais)


def create():
    Config.CANVAS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    CHUNK_SIZE = Config.CANVAS_CHUNK_SIZE

    # Determina o shape do canvas final
    canvas_shape = create_blank_canvas_shape_from_tiles(
        Config.TILES_DIR, Config.CANVAS_GAP
    )
    print("Shape do canvas:", canvas_shape)

    # Criação do Dask array
    darr = da.full(
        canvas_shape, fill_value=255, dtype=np.uint8, chunks=Config.CANVAS_CHUNK_SIZE
    )

    # Criação do arquivo memmap
    memmap_img = np.memmap(
        Config.BLANK_CANVAS_PATH, dtype=np.uint8, mode="w+", shape=canvas_shape
    )

    # Preenche o memmap com o conteúdo do Dask array (vazio por enquanto)
    for i in range(0, canvas_shape[0], CHUNK_SIZE[0]):
        for j in range(0, canvas_shape[1], CHUNK_SIZE[1]):
            bloco = darr[i : i + CHUNK_SIZE[0], j : j + CHUNK_SIZE[1]].compute()
            memmap_img[i : i + CHUNK_SIZE[0], j : j + CHUNK_SIZE[1]] = bloco

    memmap_img.flush()
    print(f"Memmap salvo em: {Config.BLANK_CANVAS_PATH}")

    # Salva o shape do canvas para uso posterior
    np.save(Config.CANVAS_OUTPUT_PATH / "canvas_shape.npy", canvas_shape)


if __name__ == "__main__":
    start_time = time.perf_counter()
    create()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")
