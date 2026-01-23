# src/modules/canvas/create_geom.py

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import cv2
import time

from src.config.config import Config
from src.utils.io import load_positions_pickle
from src.utils.images import list_tile_images, infer_tile_shape

logger = logging.getLogger(__name__)

# Nó do grafo/topologia: (x, y) no grid
Node = Tuple[int, int]
# Posição global em pixels: (X, Y)
PosXY = Tuple[float, float]
Positions = Dict[Node, PosXY]

def compute_canvas_shape_from_positions(
    positions: Positions,
    tile_w: int,
    tile_h: int,
    channels: int = 3,
) -> Tuple[Tuple[int, int, int], Tuple[float, float, float, float]]:
    """
    Calcula o shape do canvas (H, W, C) necessário para acomodar todos os tiles
    com base nas posições globais (X,Y) dos cantos superiores esquerdos.

    Retorna:
      - canvas_shape: (H, W, C)
      - bbox: (min_x, min_y, max_x, max_y) para debug
    """
    if not positions:
        raise ValueError("Positions vazio: não há como calcular canvas.")

    xs = [xy[0] for xy in positions.values()]
    ys = [xy[1] for xy in positions.values()]

    min_x = float(min(xs))
    min_y = float(min(ys))
    max_x = float(max(xs))
    max_y = float(max(ys))

    # (max - min) dá a extensão ocupada pelos cantos; soma tile_w/tile_h para cobrir o tile inteiro
    canvas_w = int(np.ceil((max_x - min_x) + tile_w))
    canvas_h = int(np.ceil((max_y - min_y) + tile_h))

    canvas_shape = (canvas_h, canvas_w, channels)

    logger.info(
        f"[GEOM] bbox: minX={min_x:.2f}, minY={min_y:.2f}, maxX={max_x:.2f}, maxY={max_y:.2f} "
        f"=> canvas_shape={canvas_shape}"
    )
    return canvas_shape, (min_x, min_y, max_x, max_y)


def create_blank_canvas_geom(
    force_recompute: bool = False,
) -> Tuple[Path, Tuple[int, int, int]]:
    """
    Cria um canvas branco (memmap) no disco para o mosaico geométrico.

    Fonte de verdade para o tamanho:
      - Config.GLOBAL_POS_FILE (positions globais)
      - Um tile de amostra em Config.TILES_DIR (para tile_w/tile_h)
    
    Salva:
      - Config.BLANK_CANVAS_GEOM_PATH  (memmap .dat)
      - canvas_shape.npy dentro de Config.CANVAS_OUTPUT_PATH

    Retorna:
      (memmap_path, canvas_shape)
    """
    memmap_path = Config.BLANK_CANVAS_GEOM_PATH
    shape_path = Config.CANVAS_GEOM_SHAPE_PATH

    # Cache simples
    if memmap_path.exists() and shape_path.exists() and not force_recompute:
        canvas_shape_arr = np.load(shape_path)
        canvas_shape = tuple(int(x) for x in canvas_shape_arr.tolist())
        logger.info(f"[GEOM] Canvas já existe. Reusando: {memmap_path}")
        logger.info(f"[GEOM] Shape carregado de: {shape_path} => {canvas_shape}")
        return memmap_path, canvas_shape  # type: ignore

    # 1) carregar positions
    positions = load_positions_pickle(Config.GLOBAL_POS_FILE)
    logger.info(f"[GEOM] Positions carregadas: {len(positions)}")

    # 2) escolher um tile de amostra (extensão-agnóstico)
    tiles = list_tile_images(Config.TILES_DIR)
    if not tiles:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {Config.TILES_DIR}")

    sample_tile = tiles[0]
    tile_h, tile_w, channels = infer_tile_shape(sample_tile)
    if channels != 3:
        raise ValueError(
            f"Tile de amostra não é RGB (3 canais). Encontrado channels={channels} em {sample_tile}"
        )
    logger.info(f"[GEOM] Amostra: {sample_tile.name} => tile (w={tile_w}, h={tile_h}, c={channels})")

    # 3) calcular canvas_shape por bbox geométrico
    canvas_shape, _bbox = compute_canvas_shape_from_positions(
        positions=positions,
        tile_w=tile_w,
        tile_h=tile_h,
        channels=3,  # sempre RGB
    )

    # 4) criar o canvas branco usando dask + memmap em chunks
    Config.CANVAS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    # Remover memmap antigo se for forçar recomputação
    if memmap_path.exists() and force_recompute:
        memmap_path.unlink()

    # 5) criar o memmap do tamanho do canvas
    canvas_memmap = np.memmap(
        memmap_path,
        dtype=np.uint8,
        mode="w+",
        shape=canvas_shape
    )

    # 6) preencher o memmap em CHUNKS para não explodir RAM em canvas gigantes
    chunk_h, chunk_w, chunk_c = Config.CANVAS_CHUNK_SIZE
    H, W, C = canvas_shape

    if chunk_c != 3:
        raise ValueError(f"Config.CANVAS_CHUNK_SIZE deve ter 3 no canal. Encontrado: {Config.CANVAS_CHUNK_SIZE}")

    fill_value = int(getattr(Config, "CANVAS_FILL_VALUE", 255))

    logger.info(f"[GEOM] Preenchendo canvas branco em chunks: chunk={Config.CANVAS_CHUNK_SIZE}, fill={fill_value}")

    for y in range(0, H, chunk_h):
        y_end = min(y + chunk_h, H)
        for x in range(0, W, chunk_w):
            x_end = min(x + chunk_w, W)

            canvas_memmap[y:y_end, x:x_end, :] = fill_value

    canvas_memmap.flush()

    # 7) salvar shape
    np.save(shape_path, np.array(canvas_shape, dtype=np.int64))

    logger.info(f"[GEOM] Canvas branco criado em: {memmap_path}")
    logger.info(f"[GEOM] Canvas shape salvo em: {shape_path} => {canvas_shape}")

    return memmap_path, canvas_shape



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


def main() -> None:
    memmap_path, canvas_shape = create_blank_canvas_geom(force_recompute=True)
    logger.info(f"[GEOM] Finalizado. memmap={memmap_path}, shape={canvas_shape}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] - %(message)s")
    start_time = time.perf_counter()
    
    main()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")

