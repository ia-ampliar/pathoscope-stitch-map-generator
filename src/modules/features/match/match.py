import re
import time

import cv2
import numpy as np
import zarr
from joblib import Parallel, delayed
import json

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

    # Verifica se os descritores foram carregados corretamente
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        print(f"Descritores vazios: {tile_a} ou {tile_b}")
        return 0

    # Realiza o matching inicial (KNN ou Brute Force, com ratio test embutido no matcher)
    raw_matches = matcher.match(kp1, desc1, kp2, desc2)
    raw_match_count = len(raw_matches)

    # Mínimo de matches para prosseguir
    if raw_match_count < Config.MIN_MATCHES:
        print(f"[REJEITADO] {tile_a} <-> {tile_b} — matches insuficientes: {raw_match_count}")
        return 0

    # --- ABORDAGEM MEDIANA (docs/tile.py) ---
    # Deslocamento (dy, dx) de cada correspondência
    dy_list = [kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1] for m in raw_matches]
    dx_list = [kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0] for m in raw_matches]

    med_y = float(np.median(dy_list))
    med_x = float(np.median(dx_list))

    # Inliers: matches cujo deslocamento está dentro de ±N_PIXELS da mediana
    inlier_flags = [
        abs(dy - med_y) <= Config.N_PIXELS and abs(dx - med_x) <= Config.N_PIXELS
        for dy, dx in zip(dy_list, dx_list)
    ]
    good_matches = [m for m, ok in zip(raw_matches, inlier_flags) if ok]
    fraction_within = len(good_matches) / raw_match_count

    # Rejeita o par se menos de 50 % dos matches converge para a mesma translação
    if fraction_within <= 0.5:
        print(
            f"[REJEITADO] {tile_a} <-> {tile_b} — "
            f"apenas {fraction_within:.1%} dos matches dentro de ±{Config.N_PIXELS}px da mediana"
        )
        return 0

    # Translação final = mediana dos deslocamentos (robusta a outliers pontuais)
    dy = med_y
    dx = med_x

    # RMSE de dispersão dos inliers em torno da mediana (substitui ransac_rmse)
    inlier_dy = [dy_list[i] for i, ok in enumerate(inlier_flags) if ok]
    inlier_dx = [dx_list[i] for i, ok in enumerate(inlier_flags) if ok]
    errs = [(d - med_y) ** 2 + (e - med_x) ** 2 for d, e in zip(inlier_dy, inlier_dx)]
    median_spread = float(np.mean(errs) ** 0.5) if errs else 0.0

    print(
        f"[MEDIANA] {tile_a} <-> {tile_b} — "
        f"dy={dy:.2f}px, dx={dx:.2f}px | "
        f"inliers={len(good_matches)}/{raw_match_count} ({fraction_within:.1%}) | "
        f"spread={median_spread:.2f}px"
    )

    # Matriz de translação homogênea 3×3
    matrix = np.array(
        [
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # --- SALVAMENTO NO ZARR ---
    match_filename = f"{tile_a}__{tile_b}.zarr"
    output_path = Config.MATCHING_ZARR_PATH / match_filename
    match_zarr_store = zarr.open(output_path, mode="w")
    group = match_zarr_store.create_group("matches", overwrite=True)

    zarr.array(
        np.array([(m.queryIdx, m.trainIdx) for m in good_matches]),
        store=group.store,
        path=f"{group.path}/matches",
        chunks=(100, 2),
        dtype=int,
    )

    # Metadados — contrato mantido para geograph.py / globalpos.py
    group.attrs["tile_a"] = tile_a
    group.attrs["tile_b"] = tile_b
    group.attrs["translation_matrix"] = matrix.tolist()
    group.attrs["inlier_count"] = len(good_matches)
    group.attrs["raw_match_count"] = int(raw_match_count)
    group.attrs["ransac_rmse"] = median_spread   # chave mantida; valor = spread mediana

    print(f"[SALVO] {output_path} com matriz de transformação.")
    return 1


def match():
    print(f"[INÍCIO] Abertura do Zarr em: {Config.KEYPOINTS_ZARR_STORE}")
    Config.MATCHING_ZARR_PATH.mkdir(parents=True, exist_ok=True)

    # Carregar tiles válidos
    with open(Config.VALID_TILES_FILE) as f:
        valid_tiles = json.load(f)
    
    store = zarr.open(Config.KEYPOINTS_ZARR_STORE, mode="r")
    all_tile_names = list(store.group_keys())
    
    # Filtrar apenas tiles válidos
    tile_names = [tile for tile in all_tile_names if valid_tiles.get(tile, False)]
    print(f"Tiles encontrados no Zarr: {len(all_tile_names)}")
    print(f"Tiles válidos para matching: {len(tile_names)}")

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
