"""
Testes de integração por etapa usando a fixture sintética 3x3.

Valida que cada etapa produz o artefato esperado e que o dado de saída
é consumível pela etapa seguinte.
"""

import json
from pathlib import Path

import pytest


class TestFetchStage:
    def test_produces_metadata(self, synthetic_dataset):
        from src.config.config import Config
        from src.modules.tile.fetch.fetch import extract_and_save_metadata

        count = extract_and_save_metadata(Config.NORMALIZED_DIR, Config.METADATA_FILE)
        assert count == 9  # 3x3 grid
        assert Config.METADATA_FILE.exists()

        with open(Config.METADATA_FILE) as f:
            data = json.load(f)
        assert len(data) == 9
        assert "coordinates" in data[0]
        assert len(data[0]["coordinates"]) == 2


class TestClassifyStage:
    def test_produces_valid_tiles(self, synthetic_dataset):
        from src.config.config import Config
        from src.modules.tile.fetch.fetch import extract_and_save_metadata
        from src.modules.tile.classify.classifier import run_classification

        # Pré-requisito: fetch
        extract_and_save_metadata(Config.NORMALIZED_DIR, Config.METADATA_FILE)

        run_classification(parallel=False)

        assert Config.VALID_TILES_FILE.exists()
        with open(Config.VALID_TILES_FILE) as f:
            valid_tiles = json.load(f)
        # Tiles sintéticos têm conteúdo (ruído+círculos) => devem ser válidos
        assert len(valid_tiles) == 9
        assert any(v for v in valid_tiles.values()), "Pelo menos 1 tile deve ser válido"


class TestDetectStage:
    def test_produces_features_zarr(self, synthetic_dataset):
        from src.config.config import Config
        from src.modules.tile.fetch.fetch import extract_and_save_metadata
        from src.modules.tile.classify.classifier import run_classification
        from src.modules.features.detect.detect import detect

        # Pré-requisitos
        extract_and_save_metadata(Config.NORMALIZED_DIR, Config.METADATA_FILE)
        run_classification(parallel=False)

        detect()

        assert Config.KEYPOINTS_ZARR_STORE.exists()

    def test_incremental_skips_existing(self, synthetic_dataset):
        """Reexecutar detect() não reprocessa tiles já no zarr."""
        from src.config.config import Config
        from src.modules.tile.fetch.fetch import extract_and_save_metadata
        from src.modules.tile.classify.classifier import run_classification
        from src.modules.features.detect.detect import detect
        import zarr

        extract_and_save_metadata(Config.NORMALIZED_DIR, Config.METADATA_FILE)
        run_classification(parallel=False)

        detect()  # primeira execução
        store = zarr.open(Config.KEYPOINTS_ZARR_STORE, mode="r")
        count_first = len(list(store.group_keys()))

        detect()  # segunda execução — deve pular tudo
        store2 = zarr.open(Config.KEYPOINTS_ZARR_STORE, mode="r")
        count_second = len(list(store2.group_keys()))

        assert count_first == count_second


class TestMatchStage:
    def test_produces_match_zarrs(self, synthetic_dataset):
        from src.config.config import Config
        from src.modules.tile.fetch.fetch import extract_and_save_metadata
        from src.modules.tile.classify.classifier import run_classification
        from src.modules.features.detect.detect import detect
        from src.modules.features.match.match import match

        # Pré-requisitos
        extract_and_save_metadata(Config.NORMALIZED_DIR, Config.METADATA_FILE)
        run_classification(parallel=False)
        detect()

        match()

        match_files = list(Config.MATCHING_ZARR_PATH.glob("*.zarr"))
        # Um grid 3x3 tem 12 pares vizinhos (4-conectados)
        assert len(match_files) > 0, "Deveria ter pelo menos 1 match zarr"
