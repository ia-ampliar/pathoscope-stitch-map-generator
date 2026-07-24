"""
conftest.py — Fixtures compartilhadas para a suíte de testes.

Fornece um dataset sintético (grid 3x3 de tiles com sobreposição conhecida)
para testar o pipeline end-to-end sem dependência de dados reais.
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config.config import Config


@pytest.fixture()
def synthetic_dataset(tmp_path: Path, monkeypatch):
    """Cria um dataset sintético 3x3 de tiles com sobreposição de ~25%.

    Cada tile é 200x200 pixels, com deslocamento real de 150px entre vizinhos
    (overlap de 50px). Conteúdo: ruído aleatório com seed fixa para gerar
    features reais via SIFT/ORB.

    A fixture configura Config para apontar para os diretórios temporários
    e retorna um dict com metadados do grid.
    """
    rng = np.random.RandomState(123)
    tile_size = 200
    overlap = 50
    step = tile_size - overlap  # 150
    grid_n = 3

    # Canvas completo do qual os tiles serão recortados
    canvas_h = tile_size + (grid_n - 1) * step  # 200 + 2*150 = 500
    canvas_w = canvas_h
    canvas = rng.randint(30, 220, (canvas_h, canvas_w, 3), dtype=np.uint8)

    # Adicionar texturas (círculos, linhas) para melhorar features
    for _ in range(50):
        cx, cy = rng.randint(0, canvas_w), rng.randint(0, canvas_h)
        r = rng.randint(5, 30)
        color = tuple(int(c) for c in rng.randint(0, 255, 3))
        cv2.circle(canvas, (cx, cy), r, color, -1)

    # Criar diretórios
    tiles_src = tmp_path / "output" / "tiles" / "src"
    tiles_norm = tmp_path / "output" / "tiles" / "normalized"
    for d in [
        tiles_src, tiles_norm,
        tmp_path / "output" / "features",
        tmp_path / "output" / "matches",
        tmp_path / "output" / "metadata",
        tmp_path / "output" / "result",
        tmp_path / "output" / "tmp" / "canvas",
        tmp_path / "output" / "tmp" / "classified",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Gerar tiles
    tile_names = []
    positions_real = {}
    idx = 1
    for row in range(1, grid_n + 1):
        for col in range(1, grid_n + 1):
            y0 = (row - 1) * step
            x0 = (col - 1) * step
            tile_img = canvas[y0 : y0 + tile_size, x0 : x0 + tile_size].copy()
            name = f"{idx:05d}_x{col}_y{row}_zp1.jpg"
            # Salvar em src e normalized (já "normalizado")
            cv2.imwrite(str(tiles_src / name), tile_img)
            cv2.imwrite(str(tiles_norm / name), tile_img)
            tile_names.append(name)
            positions_real[(col, row)] = (float(x0), float(y0))
            idx += 1

    # Monkeypatch Config para usar tmp_path
    monkeypatch.setattr(Config, "BASE_DIR", tmp_path / "output")
    monkeypatch.setattr(Config, "TILES_DIR", tiles_src)
    monkeypatch.setattr(Config, "NORMALIZED_DIR", tiles_norm)
    monkeypatch.setattr(Config, "FEATURES_DIR", tmp_path / "output" / "features")
    monkeypatch.setattr(Config, "KEYPOINTS_ZARR_STORE", tmp_path / "output" / "features" / "features.zarr")
    monkeypatch.setattr(Config, "MATCHING_ZARR_PATH", tmp_path / "output" / "matches")
    monkeypatch.setattr(Config, "METADATA_FILE", tmp_path / "output" / "metadata" / "dataset.json")
    monkeypatch.setattr(Config, "VALID_TILES_FILE", tmp_path / "output" / "metadata" / "valid_tiles.json")
    monkeypatch.setattr(Config, "TOPOLOGY_GRAPH_FILE", tmp_path / "output" / "result" / "graph_topology.gpickle")
    monkeypatch.setattr(Config, "GEOMETRIC_GRAPH_FILE", tmp_path / "output" / "result" / "graph_geometric.gpickle")
    monkeypatch.setattr(Config, "GEOMETRIC_GRAPH_WEIGHTS_FILE", tmp_path / "output" / "result" / "graph_geometric_weights.jpg")
    monkeypatch.setattr(Config, "GLOBAL_POS_FILE", tmp_path / "output" / "result" / "global_positions.pkl")
    monkeypatch.setattr(Config, "GRAPH_FILE", tmp_path / "output" / "result" / "graph.jpg")
    monkeypatch.setattr(Config, "CANVAS_OUTPUT_PATH", tmp_path / "output" / "tmp" / "canvas")
    monkeypatch.setattr(Config, "BLANK_CANVAS_GEOM_PATH", tmp_path / "output" / "tmp" / "canvas" / "canvas_geom.dat")
    monkeypatch.setattr(Config, "CANVAS_GEOM_SHAPE_PATH", tmp_path / "output" / "tmp" / "canvas" / "canvas_geom_shape.npy")
    monkeypatch.setattr(Config, "CANVAS_PREVIEW_PATH", tmp_path / "output" / "tmp" / "canvas" / "mosaic_preview.jpg")
    monkeypatch.setattr(Config, "CANVAS_GEOM_PATH", tmp_path / "output" / "tmp" / "canvas" / "mosaic_geom.tif")
    monkeypatch.setattr(Config, "CLASSIFIED_DIR", tmp_path / "output" / "tmp" / "classified")
    # Usar 1 job para evitar problemas de fork com monkeypatch
    monkeypatch.setattr(Config, "DETECTION_N_JOBS", 1)
    monkeypatch.setattr(Config, "MATCHING_N_JOBS", 1)

    return {
        "tmp_path": tmp_path,
        "tile_names": tile_names,
        "positions_real": positions_real,
        "tile_size": tile_size,
        "step": step,
        "grid_n": grid_n,
    }
