"""
Testes para src/utils/contracts.py — validação de artefatos entre etapas.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.utils.contracts import (
    load_metadata,
    load_valid_tiles,
    load_positions,
    load_graph,
    validate_match_zarr,
)


class TestLoadMetadata:
    def test_valid(self, tmp_path: Path):
        data = [
            {"name": "tile.jpg", "path": "/tmp", "coordinates": [1, 2]},
        ]
        p = tmp_path / "dataset.json"
        p.write_text(json.dumps(data))
        result = load_metadata(p)
        assert len(result) == 1

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Metadados"):
            load_metadata(tmp_path / "nao_existe.json")

    def test_wrong_type(self, tmp_path: Path):
        p = tmp_path / "dataset.json"
        p.write_text(json.dumps({"not": "a list"}))
        with pytest.raises(ValueError, match="esperado lista"):
            load_metadata(p)

    def test_missing_keys(self, tmp_path: Path):
        p = tmp_path / "dataset.json"
        p.write_text(json.dumps([{"name": "a.jpg"}]))
        with pytest.raises(ValueError, match="chaves ausentes"):
            load_metadata(p)

    def test_bad_coordinates(self, tmp_path: Path):
        p = tmp_path / "dataset.json"
        p.write_text(json.dumps([{"name": "a.jpg", "path": "/x", "coordinates": [1, 2, 3]}]))
        with pytest.raises(ValueError, match="2 elementos"):
            load_metadata(p)


class TestLoadValidTiles:
    def test_valid(self, tmp_path: Path):
        p = tmp_path / "valid.json"
        p.write_text(json.dumps({"tile_a": True, "tile_b": False}))
        result = load_valid_tiles(p)
        assert result["tile_a"] is True

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_valid_tiles(tmp_path / "nope.json")


class TestLoadPositions:
    def test_valid(self, tmp_path: Path):
        positions = {(1, 1): (0.0, 0.0), (2, 1): (150.0, 0.0)}
        p = tmp_path / "pos.pkl"
        with p.open("wb") as f:
            pickle.dump(positions, f)
        result = load_positions(p)
        assert len(result) == 2

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_positions(tmp_path / "nope.pkl")

    def test_empty_dict(self, tmp_path: Path):
        p = tmp_path / "pos.pkl"
        with p.open("wb") as f:
            pickle.dump({}, f)
        with pytest.raises(ValueError, match="vazio"):
            load_positions(p)


class TestLoadGraph:
    def test_valid(self, tmp_path: Path):
        import networkx as nx

        G = nx.Graph()
        G.add_node((1, 1), label="tile_a")
        p = tmp_path / "graph.gpickle"
        with p.open("wb") as f:
            pickle.dump(G, f)
        result = load_graph(p)
        assert result.number_of_nodes() == 1

    def test_empty_graph(self, tmp_path: Path):
        import networkx as nx

        G = nx.Graph()
        p = tmp_path / "graph.gpickle"
        with p.open("wb") as f:
            pickle.dump(G, f)
        with pytest.raises(ValueError, match="vazio"):
            load_graph(p)
