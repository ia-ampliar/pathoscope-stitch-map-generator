import re
import time

import cv2
import numpy as np
import zarr
from joblib import Parallel, delayed

from src.config.config import Config
from src.utils.coordinates import extract_coordinates

from .registry import get_matcher


def load_keypoints_and_descriptors(zarr_store, tile_name: str):
    print(f"Carregando keypoints e descritores de: {tile_name}")
    group = zarr_store[tile_name]
    kp_array = group["keypoints"][:]
    descriptors = group["descriptors"][:]

    keypoints = [
        cv2.KeyPoint(
            x=float(row[0]),
            y=float(row[1]),
            size=float(row[2]),
            angle=float(row[3]),
            response=float(row[4]),
            octave=int(row[5]),
            class_id=int(row[6]),
        )
        for row in kp_array
    ]

    print(
        f"Tile {tile_name}: {len(keypoints)} keypoints, descritores shape = {descriptors.shape}"
    )
    return keypoints, descriptors


def match_pair(tile_a, tile_b) -> int:
    matcher = get_matcher(
        Config.MATCHER,
        algorithm=Config.DETECTION_ALGORITHM,
        ratio_thresh=Config.MATCHING_RATIO_THRESH,
    )
    store = zarr.open(Config.KEYPOINTS_ZARR_STORE, mode="r")

    try:
        kp1, desc1 = load_keypoints_and_descriptors(store, tile_a)
        kp2, desc2 = load_keypoints_and_descriptors(store, tile_b)
    except Exception as e:
        print(f"Falha ao carregar dados: {tile_a} <-> {tile_b}. Erro: {e}")
        return 0

    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        print(f"Descritores vazios: {tile_a} ou {tile_b}")
        return 0

    print(f"[MATCHING] {tile_a} <-> {tile_b}")
    raw_matches = matcher.match(kp1, desc1, kp2, desc2)
    matches = raw_matches[:100]

    if len(matches) == 0:
        return 0

    match_filename = f"{tile_a}__{tile_b}.zarr"
    output_path = Config.MATCHING_ZARR_PATH / match_filename
    match_zarr_store = zarr.open(output_path, mode="w")
    group = match_zarr_store.create_group("matches", overwrite=True)

    zarr.array(
        np.array([(m.queryIdx, m.trainIdx) for m in matches]),
        store=group.store,
        path=f"{group.path}/matches",
        chunks=(100, 2),
        dtype=int,
    )

    group.attrs["tile_a"] = tile_a
    group.attrs["tile_b"] = tile_b

    print(f"[SALVO] {output_path}")
    return 1


def match():
    print(f"[INÍCIO] Abertura do Zarr em: {Config.KEYPOINTS_ZARR_STORE}")
    Config.MATCHING_ZARR_PATH.mkdir(parents=True, exist_ok=True)

    store = zarr.open(Config.KEYPOINTS_ZARR_STORE, mode="r")
    tile_names = list(store.group_keys())
    print(f"Tiles encontrados no Zarr: {len(tile_names)}")

    pattern = re.compile(Config.COORDINATES_PATTERN)
    tile_coords = {tile: extract_coordinates(tile, pattern) for tile in tile_names}
    coord_to_tile = {v: k for k, v in tile_coords.items()}

    tile_pairs = set()
    for tile_name, (x, y) in tile_coords.items():
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor_coords = (x + dx, y + dy)
            neighbor_tile = coord_to_tile.get(neighbor_coords)
            if neighbor_tile:
                pair = tuple(sorted((tile_name, neighbor_tile)))
                tile_pairs.add(pair)

    print(f"Total de pares únicos de vizinhos: {len(tile_pairs)}")

    total_salvos = Parallel(n_jobs=Config.MATCHING_N_JOBS)(
        delayed(match_pair)(tile_a, tile_b) for tile_a, tile_b in sorted(tile_pairs)
    )

    print(f"\n[FIM] Total de matches salvos: {sum(total_salvos)}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    match()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")
