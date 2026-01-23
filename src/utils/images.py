from pathlib import Path
from typing import Dict, Tuple, List

import cv2

from src.config.config import Config

# Nó do grafo/topologia: (x, y) no grid
Node = Tuple[int, int]
# Posição global em pixels: (X, Y)
PosXY = Tuple[float, float]
Positions = Dict[Node, PosXY]


# Extensões suportadas (você disse que o principal é .jpg, mas vamos ser genéricos)
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def list_tile_images(tiles_dir: Path) -> List[Path]:
    """
    Lista imagens em tiles_dir de forma extensão-agnóstica:
      - varre glob("*")
      - filtra por extensões suportadas
      - ordena para ter determinismo
    """
    if not tiles_dir.exists():
        raise FileNotFoundError(f"TILES_DIR não encontrado: {tiles_dir}")

    paths = [p for p in tiles_dir.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    paths.sort()
    return paths


def infer_tile_shape(sample_path: Path) -> Tuple[int, int, int]:
    """
    Lê um tile de amostra e retorna (h, w, c).
    Você disse que nunca é grayscale, então assumimos c=3 quando img.ndim==3.
    """
    img = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {sample_path}")

    h, w, c = img.shape  # esperado c=3
    return h, w, c