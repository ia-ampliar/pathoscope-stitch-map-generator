import argparse
import re
import time

import cv2
import numpy as np
from tifffile import imwrite

from src.config.config import Config
from src.utils.coordinates import extract_coordinates


def save_jpg_scaled(image: np.ndarray, output_path, scale: float):
    """Salva uma versão reduzida do canvas em JPG."""
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Converte RGB para BGR para salvar com OpenCV
    resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), resized_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Canvas JPG ({new_w}x{new_h}) salvo em: {output_path}")


def populate(scale: float = None):
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

        # Coordenadas já são base-0, calcula posição no canvas com GAP nas bordas
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

    # Salva versão JPG reduzida se --scale foi fornecido
    if scale is not None:
        save_jpg_scaled(canvas_memmap, Config.CANVAS_POPULATED_JPG_PATH, scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preenche o canvas com os tiles posicionados."
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Fator de escala (0.0 a 1.0) para salvar uma versão JPG reduzida do canvas.",
    )
    args = parser.parse_args()

    start_time = time.perf_counter()
    populate(scale=args.scale)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")
