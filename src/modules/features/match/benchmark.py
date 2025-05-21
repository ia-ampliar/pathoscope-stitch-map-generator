import argparse
import time
from pathlib import Path

import cv2
import zarr

from src.config.config import Config

from .match import (
    load_keypoints_and_descriptors,
)
from .registry import get_matcher

MAX_MATCHES_TO_DRAW = 500


def main():
    print("[TESTE] Desenhando matches filtrados para inspeção visual...")

    parser = argparse.ArgumentParser(
        description="Comparar dois tiles e desenhar seus matches."
    )
    parser.add_argument(
        "tile_a",
        type=str,
        help="Nome do arquivo do primeiro tile (ex: '00057_x5_y5_zp1.jpg')",
    )
    parser.add_argument(
        "tile_b",
        type=str,
        help="Nome do arquivo do segundo tile (ex: '00058_x6_y5_zp1.jpg')",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="bf",
        help="Nome do matcher a ser usado (ex: 'bf' ou 'bf_lowe')",
    )
    args = parser.parse_args()

    tile_a = Path(args.tile_a).stem
    tile_b = Path(args.tile_b).stem

    zarr_store = zarr.open(Config.KEYPOINTS_ZARR_STORE, mode="r")
    kp_a, desc_a = load_keypoints_and_descriptors(zarr_store, tile_a)
    kp_b, desc_b = load_keypoints_and_descriptors(zarr_store, tile_b)

    if desc_a is None or desc_b is None:
        print("[ERRO] Descritores não encontrados.")
        return

    img_a = cv2.imread(str(Config.BASE_DIR / "tiles" / f"{tile_a}.jpg"))
    img_b = cv2.imread(str(Config.BASE_DIR / "tiles" / f"{tile_b}.jpg"))

    if img_a is None or img_b is None:
        print("[ERRO] Não foi possível carregar as imagens.")
        return

    matcher = get_matcher(args.matcher)
    matches = matcher.match(kp_a, desc_a, kp_b, desc_b)

    if not matches:
        print("[INFO] Nenhum match encontrado.")
        return

    matches = sorted(matches, key=lambda m: m.distance)[:MAX_MATCHES_TO_DRAW]
    print(f"[INFO] {len(matches)} melhores matches selecionados para desenho.")

    img_matches = cv2.drawMatches(
        img_a,
        kp_a,
        img_b,
        kp_b,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    output_dir = Config.BASE_DIR / "tmp" / "matching_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"matches_{tile_a}_{tile_b}.jpg"
    cv2.imwrite(str(output_path), img_matches)

    print(f"[FIM] Matches desenhados e imagem salva em: {output_path}")


if __name__ == "__main__":
    # python -m src.modules.features.match.benchmark "path/to/tile1" "path/to/tile2"
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")
