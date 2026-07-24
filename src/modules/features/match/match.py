import re
import time
from functools import lru_cache

import cv2
import numpy as np
import zarr
from joblib import Parallel, delayed
import json
import logging

from src.config.config import Config
from src.utils.coordinates import extract_coordinates

from .registry import get_matcher

logger = logging.getLogger(__name__)


# --- Singletons por processo (worker) ---------------------------------------
# Em execução paralela (joblib/loky), cada worker é um processo separado e
# inicializa seu próprio store/matcher sob demanda. Isso evita reabrir o store
# Zarr e reinstanciar o matcher a cada par de tiles.
_STORE = None
_MATCHER = None


def _get_store():
    """Abre o store Zarr de features uma única vez por processo."""
    global _STORE
    if _STORE is None:
        _STORE = zarr.open(Config.KEYPOINTS_ZARR_STORE, mode="r")
    return _STORE


def _get_matcher():
    """Instancia o matcher uma única vez por processo."""
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = get_matcher(
            Config.MATCHER,
            algorithm=Config.DETECTION_ALGORITHM,
            ratio_thresh=Config.MATCHING_RATIO_THRESH,
        )
    return _MATCHER


def load_keypoints_and_descriptors(zarr_store, tile_name: str):
    logger.debug(f"Carregando keypoints e descritores de: {tile_name}")
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

    logger.debug(
        f"Tile {tile_name}: {len(keypoints)} keypoints, descritores shape = {descriptors.shape}"
    )
    return keypoints, descriptors


@lru_cache(maxsize=None)
def _load_kp_desc_cached(tile_name: str):
    """Cache por processo: cada tile é lido no máximo uma vez por worker.

    Como cada tile participa de até 4 pares vizinhos, o cache elimina a
    releitura redundante de keypoints/descritores do mesmo tile.
    """
    return load_keypoints_and_descriptors(_get_store(), tile_name)


def match_pair(tile_a, tile_b) -> int:
    matcher = _get_matcher()

    try:
        kp1, desc1 = _load_kp_desc_cached(tile_a)
        kp2, desc2 = _load_kp_desc_cached(tile_b)
    except Exception as e:
        logger.error(f"Falha ao carregar dados: {tile_a} <-> {tile_b}. Erro: {e}")
        return 0

    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        logger.warning(f"Descritores vazios: {tile_a} ou {tile_b}")
        return 0

    # print(f"[MATCHING] {tile_a} <-> {tile_b}")
    # Realiza o matching inicial (ex: KNN ou Brute Force)
    raw_matches = matcher.match(kp1, desc1, kp2, desc2)

    # Quantidade total de matches encontrados
    raw_match_count = len(raw_matches)
    
    # É necessário um mínimo de pontos para estimativa robusta [7, 8]
    if len(raw_matches) < Config.MATCHING_MIN_MATCHES:
        logger.warning(f"Matches insuficientes para RANSAC: {len(raw_matches)}")
        return 0

    # --- INÍCIO DO PROCESSO RANSAC ---
    # 1. Extrair as coordenadas (x, y) dos pontos correspondentes [4]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in raw_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in raw_matches]).reshape(-1, 1, 2)

    # 2. Estimar a Transformação Afim com RANSAC para remover outliers [2, 3]
    # O threshold de 5.0 define a tolerância de erro de reprojeção em pixels

    # --- HOMOGRAFIA ---
    # matrix, mask = cv2.findHomography(
    #     src_pts, 
    #     dst_pts, 
    #     method=cv2.RANSAC, 
    #     ransacReprojThreshold=5.0
    # )

    # --- TRANSLAÇÃO FORÇADA ---
    src_xy = src_pts.reshape(-1, 2).astype(np.float32)
    dst_xy = dst_pts.reshape(-1, 2).astype(np.float32)

    M, mask = cv2.estimateAffinePartial2D(
        src_xy,
        dst_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=Config.RANSAC_REPROJ_THRESHOLD,
        maxIters=Config.RANSAC_MAX_ITERS,
        confidence=Config.RANSAC_CONFIDENCE,
        refineIters=Config.RANSAC_REFINE_ITERS,
    )

    if M is None or mask is None:
        logger.warning(f"Falha ao estimar translação robusta para {tile_a} e {tile_b}")
        return 0

    dx = float(M[0, 2])
    dy = float(M[1, 2])

    # 3) Construir matriz homogênea 3x3 APENAS de translação
    matrix = np.array(
        [
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # 4) Filtrar apenas os "inliers"
    matches_mask = mask.ravel().tolist()
    good_matches = [raw_matches[i] for i in range(len(raw_matches)) if matches_mask[i]]

    # RMSE de reprojeção nos inliers 
    inlier_src = src_xy[mask.ravel() == 1]
    inlier_dst = dst_xy[mask.ravel() == 1]

    # aplica M: [a b tx; c d ty]
    pred = (inlier_src @ M[:, :2].T) + M[:, 2]
    err = inlier_dst - pred
    ransac_rmse = float((err[:, 0] ** 2 + err[:, 1] ** 2).mean() ** 0.5)

    logger.info(f"[RANSAC] Filtrados {len(good_matches)} inliers de {len(raw_matches)} matches totais.")

    if len(good_matches) == 0:
        return 0

    # --- SALVAMENTO NO ZARR ---
    match_filename = f"{tile_a}__{tile_b}.zarr"
    output_path = Config.MATCHING_ZARR_PATH / match_filename
    match_zarr_store = zarr.open(output_path, mode="w")
    group = match_zarr_store.create_group("matches", overwrite=True)

    # Salva apenas os índices dos matches validados pelo RANSAC 
    zarr.array(
        np.array([(m.queryIdx, m.trainIdx) for m in good_matches]),
        store=group.store,
        path=f"{group.path}/matches",
        chunks=(100, 2),
        dtype=int,
    )

    # Armazena a matriz resultante como metadado para uso na costura (canvas.populate) 
    group.attrs["tile_a"] = tile_a
    group.attrs["tile_b"] = tile_b
    group.attrs["translation_matrix"] = matrix.tolist() # Converter para lista para JSON
    group.attrs["inlier_count"] = len(good_matches)
    group.attrs["raw_match_count"] = int(raw_match_count)
    group.attrs["ransac_rmse"] = float(ransac_rmse)

    logger.info(f"[SALVO] {output_path} com matriz de transformação.")
    return 1


def match():
    # Fixar seeds para reprodutibilidade do RANSAC e operações numpy
    cv2.setRNGSeed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

    logger.info(f"[INÍCIO] Abertura do Zarr em: {Config.KEYPOINTS_ZARR_STORE}")
    Config.MATCHING_ZARR_PATH.mkdir(parents=True, exist_ok=True)

    # Carregar tiles válidos
    with open(Config.VALID_TILES_FILE) as f:
        valid_tiles = json.load(f)
    
    store = zarr.open(Config.KEYPOINTS_ZARR_STORE, mode="r")
    all_tile_names = list(store.group_keys())
    
    # Filtrar apenas tiles válidos
    tile_names = [tile for tile in all_tile_names if valid_tiles.get(tile, False)]
    logger.info(f"Tiles encontrados no Zarr: {len(all_tile_names)}")
    logger.info(f"Tiles válidos para matching: {len(tile_names)}")

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

    logger.info(f"Total de pares únicos de vizinhos: {len(tile_pairs)}")

    # --- Retomada incremental ---
    # Pula pares cujo arquivo de output zarr já existe e é não-vazio.
    pairs_sorted = sorted(tile_pairs)
    pairs_to_process = []
    skipped = 0
    for tile_a, tile_b in pairs_sorted:
        match_filename = f"{tile_a}__{tile_b}.zarr"
        output_path = Config.MATCHING_ZARR_PATH / match_filename
        if output_path.exists():
            skipped += 1
        else:
            pairs_to_process.append((tile_a, tile_b))

    if skipped:
        logger.info(f"Retomada incremental: {skipped} pares já processados. Pulando.")
    logger.info(f"Pares a processar: {len(pairs_to_process)} de {len(tile_pairs)} totais.")

    if not pairs_to_process:
        logger.info("Todos os pares já foram processados. Nada a fazer.")
        return

    total_salvos = Parallel(n_jobs=Config.MATCHING_N_JOBS)(
        delayed(match_pair)(tile_a, tile_b) for tile_a, tile_b in pairs_to_process
    )

    logger.info(f"[FIM] Total de matches salvos nesta execução: {sum(total_salvos)}")


if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.INFO)
    start_time = time.perf_counter()
    match()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")
