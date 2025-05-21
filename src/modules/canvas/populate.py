import re
import time
from pathlib import Path

import cv2
import numpy as np
from tifffile import imwrite

from src.config.config import Config
from src.utils.coordinates import extract_coordinates


def populate():
    # Carrega shape e memmap do canvas
    canvas_shape = np.load(Config.CANVAS_SHAPE_PATH)
    canvas_memmap = np.memmap(
        Config.BLANK_CANVAS_PATH, dtype=np.uint8, mode="r+", shape=tuple(canvas_shape)
    )

    pattern = re.compile(Config.COORDINATES_PATTERN)

    # Lista todos os tiles disponíveis
    tile_paths = list(Config.TILES_DIR.glob("*.jpg"))
    if not tile_paths:
        raise RuntimeError("Nenhum tile encontrado em 'resized/'.")

    # Lê tamanho de tile a partir do primeiro
    sample_tile = cv2.imread(str(tile_paths[0]))
    if sample_tile is None:
        raise RuntimeError("Erro ao ler imagem de exemplo.")
    tile_h, tile_w, _ = sample_tile.shape

    # Insere cada tile
    for tile_path in tile_paths:
        filename = tile_path.name
        x_idx, y_idx = extract_coordinates(filename, pattern)

        # Converte para índices base-0
        x_idx -= 1
        y_idx -= 1

        # Calcula a posição no canvas com GAP nas bordas
        start_y = Config.CANVAS_GAP + y_idx * (tile_h + Config.CANVAS_GAP)
        start_x = Config.CANVAS_GAP + x_idx * (tile_w + Config.CANVAS_GAP)

        # Lê o tile atual
        tile_img = cv2.imread(str(tile_path))
        if tile_img is None:
            print(f"[AVISO] Não foi possível ler o tile {filename}. Pulando.")
            continue

        # Conversão de BGR para RGB
        tile_img = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)

        # Insere o tile no canvas
        canvas_memmap[start_y : start_y + tile_h, start_x : start_x + tile_w] = tile_img

    # Garante que tudo foi salvo no disco
    canvas_memmap.flush()

    # Salva o TIFF atualizado
    imwrite(
        Config.CANVAS_POPULATED_PATH,
        data=canvas_memmap,
        tile=(240, 240),
        bigtiff=True,
        resolutionunit="MICROMETER",
        compression="JPEG",
        dtype=np.uint8,
    )

    print(f"Canvas salvo em: {Config.CANVAS_POPULATED_PATH}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    populate()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")
