import re
import time

import cv2
import numpy as np
import tifffile
import zarr

from src.config.config import Config
from src.utils.coordinates import extract_coordinates


def draw_line(imagem, pt1, pt2, color):
    cv2.line(imagem, pt1, pt2, color=color, thickness=1)
    cv2.circle(imagem, pt1, radius=5, color=color, thickness=1)
    cv2.circle(imagem, pt2, radius=5, color=color, thickness=1)


def load_keypoints(zarr_store, tile_name):
    group = zarr_store[tile_name]
    kp_array = group["keypoints"][:]
    keypoints = []
    for row in kp_array:
        x, y, size, angle, response, octave, class_id = row
        keypoints.append(cv2.KeyPoint(float(x), float(y), float(size)))
    return keypoints


def main():
    print("[INICIO] Desenhando matches a partir de arquivos Zarr...")

    canvas_shape = np.load(Config.CANVAS_SHAPE_PATH)
    dtype = np.uint8
    canvas_memmap = np.memmap(
        Config.BLANK_CANVAS_PATH, dtype=dtype, mode="r+", shape=tuple(canvas_shape)
    )
    canvas = np.array(canvas_memmap)

    match_files = list(Config.MATCHING_ZARR_PATH.glob("*.zarr"))
    print(f"[INFO] Total de arquivos de matches encontrados: {len(match_files)}")

    zarr_store = zarr.open(Config.KEYPOINTS_ZARR_STORE, mode="r")

    # Define tamanho de tile a partir de uma imagem real
    sample_tile_path = next(Config.TILES_DIR.glob("*.jpg"), None)
    if sample_tile_path is None:
        raise RuntimeError("Nenhum tile encontrado para inferir tamanho.")
    sample_tile = cv2.imread(str(sample_tile_path))
    tile_h, tile_w = sample_tile.shape[:2]

    pattern = re.compile(Config.COORDINATES_PATTERN)

    for match_file in match_files:
        group = zarr.open(match_file, mode="r")["matches"]
        tile_a = group.attrs["tile_a"]
        tile_b = group.attrs["tile_b"]
        matches = group["matches"][: Config.CANVAS_MAX_MATCHES_TO_DRAW]

        if matches.size == 0:
            continue

        xa, ya = extract_coordinates(tile_a, pattern)
        xb, yb = extract_coordinates(tile_b, pattern)

        offset_a = (
            (xa - 1) * (tile_w + Config.CANVAS_GAP) + Config.CANVAS_GAP,
            (ya - 1) * (tile_h + Config.CANVAS_GAP) + Config.CANVAS_GAP,
        )
        offset_b = (
            (xb - 1) * (tile_w + Config.CANVAS_GAP) + Config.CANVAS_GAP,
            (yb - 1) * (tile_h + Config.CANVAS_GAP) + Config.CANVAS_GAP,
        )

        kps_a = load_keypoints(zarr_store, tile_a)
        kps_b = load_keypoints(zarr_store, tile_b)

        for idx_a, idx_b in matches:
            pt1 = kps_a[idx_a].pt
            pt2 = kps_b[idx_b].pt

            pt1_offset = (
                int(round(pt1[0] + offset_a[0])),
                int(round(pt1[1] + offset_a[1])),
            )
            pt2_offset = (
                int(round(pt2[0] + offset_b[0])),
                int(round(pt2[1] + offset_b[1])),
            )

            color = tuple(np.random.randint(0, 256, size=3).tolist())  # RGB aleatório
            draw_line(canvas, pt1_offset, pt2_offset, color)

    tifffile.imwrite(
        Config.CANVAS_WITH_DRAW_MATCHES_PATH,
        data=canvas,
        tile=(240, 240),
        bigtiff=True,
        resolutionunit="MICROMETER",
        compression="JPEG",
        dtype=np.uint8,
    )

    print(
        f"[FIM] Matches desenhados e imagem salva em: {Config.CANVAS_WITH_DRAW_MATCHES_PATH}"
    )


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")
