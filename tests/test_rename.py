"""
Testes para src/modules/tile/rename/rename.py — diagnóstico e padronização
da nomenclatura dos tiles.
"""

import json
from pathlib import Path

import pytest

from src.config.config import Config
from src.modules.tile.rename.rename import (
    analyze_filename,
    analyze_directory,
    build_rename_plan,
    apply_rename_plan,
    run_rename,
)


def _touch(directory: Path, name: str) -> Path:
    p = directory / name
    p.write_bytes(b"fake-image-bytes")
    return p


class TestAnalyzeFilename:
    def test_canonical(self):
        pattern, x, y, z = analyze_filename("00001_x2_y3_zp1")
        assert pattern == "canonical"
        assert (x, y, z) == (2, 3, 1)

    def test_xy_labeled(self):
        pattern, x, y, z = analyze_filename("scan_x10_y20")
        assert pattern == "xy_labeled"
        assert (x, y) == (10, 20)

    def test_xy_labeled_with_z(self):
        pattern, x, y, z = analyze_filename("img_x5_y6_z3")
        assert pattern == "xy_labeled"
        assert (x, y, z) == (5, 6, 3)

    def test_col_row(self):
        pattern, x, y, z = analyze_filename("col4_row7")
        assert pattern == "col_row"
        assert (x, y) == (4, 7)

    def test_row_col(self):
        pattern, x, y, z = analyze_filename("row7_col4")
        assert pattern == "row_col"
        assert (x, y) == (4, 7)

    def test_tile_x_y(self):
        pattern, x, y, z = analyze_filename("tile_3_8")
        assert pattern == "tile_x_y"
        assert (x, y) == (3, 8)

    def test_bare_x_y(self):
        pattern, x, y, z = analyze_filename("3_8")
        assert pattern == "bare_x_y"
        assert (x, y) == (3, 8)

    def test_unknown(self):
        pattern, x, y, z = analyze_filename("foto_qualquer")
        assert pattern is None
        assert x is None and y is None

    def test_default_z(self):
        _, _, _, z = analyze_filename("scan_x1_y1", default_z=9)
        assert z == 9


class TestAnalyzeDirectory:
    def test_mixed(self, tmp_path: Path):
        _touch(tmp_path, "00001_x1_y1_zp1.jpg")   # canonical
        _touch(tmp_path, "scan_x2_y1.jpg")         # xy_labeled
        _touch(tmp_path, "col3_row1.png")          # col_row
        _touch(tmp_path, "aleatorio.jpg")          # unknown
        _touch(tmp_path, "notas.txt")              # ignorado (extensão)

        analyses, report = analyze_directory(tmp_path)

        assert report.total == 4  # .txt ignorado
        assert report.canonical == 1
        assert report.renamable == 2
        assert len(report.unknown) == 1


class TestBuildRenamePlan:
    def test_skips_canonical_and_unknown(self, tmp_path: Path):
        _touch(tmp_path, "00001_x1_y1_zp1.jpg")   # canonical -> skip
        _touch(tmp_path, "scan_x2_y1.jpg")         # renomear
        _touch(tmp_path, "aleatorio.jpg")          # unknown -> skip

        analyses, _ = analyze_directory(tmp_path)
        plan = build_rename_plan(analyses)

        assert len(plan) == 1
        new_name = list(plan.values())[0]
        # Prefixo deve começar acima do maior canônico (00001) => 00002
        assert new_name == "00002_x2_y1_zp1.jpg"

    def test_raster_order_prefixes(self, tmp_path: Path):
        # Sem canônicos; ordem raster = (y, x)
        _touch(tmp_path, "x2_y1.jpg")
        _touch(tmp_path, "x1_y1.jpg")
        _touch(tmp_path, "x1_y2.jpg")

        analyses, _ = analyze_directory(tmp_path)
        plan = build_rename_plan(analyses)

        # Ordenado por (y, x): (1,1), (2,1), (1,2)
        by_new = {v: k.name for k, v in plan.items()}
        assert "00001_x1_y1_zp1.jpg" in by_new
        assert "00002_x2_y1_zp1.jpg" in by_new
        assert "00003_x1_y2_zp1.jpg" in by_new


class TestApplyRenamePlan:
    def test_dry_run_does_not_change(self, tmp_path: Path):
        _touch(tmp_path, "scan_x1_y1.jpg")
        analyses, _ = analyze_directory(tmp_path)
        plan = build_rename_plan(analyses)

        apply_rename_plan(plan, tmp_path, dry_run=True)

        assert (tmp_path / "scan_x1_y1.jpg").exists()  # inalterado
        assert not (tmp_path / "00001_x1_y1_zp1.jpg").exists()

    def test_apply_in_place(self, tmp_path: Path):
        _touch(tmp_path, "scan_x1_y1.jpg")
        _touch(tmp_path, "scan_x2_y1.jpg")
        analyses, _ = analyze_directory(tmp_path)
        plan = build_rename_plan(analyses)

        mapping = apply_rename_plan(plan, tmp_path, dry_run=False)

        assert (tmp_path / "00001_x1_y1_zp1.jpg").exists()
        assert (tmp_path / "00002_x2_y1_zp1.jpg").exists()
        assert not (tmp_path / "scan_x1_y1.jpg").exists()
        assert len(mapping) == 2
        # Mapa de auditoria gravado
        assert (tmp_path.parent / "rename_map.json").exists()

    def test_apply_copy_to_dest(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"
        _touch(src, "tile_1_1.jpg")
        analyses, _ = analyze_directory(src)
        plan = build_rename_plan(analyses)

        apply_rename_plan(plan, src, dest_dir=dest, dry_run=False)

        # Original preservado, cópia canônica no destino
        assert (src / "tile_1_1.jpg").exists()
        assert (dest / "00001_x1_y1_zp1.jpg").exists()

    def test_generated_names_match_pipeline_pattern(self, tmp_path: Path):
        import re

        _touch(tmp_path, "row2_col3.jpg")
        analyses, _ = analyze_directory(tmp_path)
        plan = build_rename_plan(analyses)
        for new_name in plan.values():
            assert re.search(Config.COORDINATES_PATTERN, new_name) is not None


class TestRunRename:
    def test_all_canonical_noop(self, tmp_path: Path):
        _touch(tmp_path, "00001_x1_y1_zp1.jpg")
        _touch(tmp_path, "00002_x2_y1_zp1.jpg")

        report = run_rename(directory=tmp_path, apply=True)

        assert report.canonical == 2
        assert report.renamable == 0
        # Nada renomeado
        assert (tmp_path / "00001_x1_y1_zp1.jpg").exists()

    def test_end_to_end_apply(self, tmp_path: Path):
        _touch(tmp_path, "col1_row1.jpg")
        _touch(tmp_path, "col2_row1.jpg")

        run_rename(directory=tmp_path, apply=True)

        assert (tmp_path / "00001_x1_y1_zp1.jpg").exists()
        assert (tmp_path / "00002_x2_y1_zp1.jpg").exists()
