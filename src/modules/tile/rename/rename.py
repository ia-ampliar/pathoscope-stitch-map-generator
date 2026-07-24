"""
rename.py — Diagnóstico e padronização da nomenclatura dos tiles.

O pipeline exige que os nomes dos tiles contenham as coordenadas no formato
que casa com `Config.COORDINATES_PATTERN` (por padrão `.*_x(\\d+)_y(\\d+)_.*`).
O formato canônico adotado pelo projeto é:

    {prefix}_x{X}_y{Y}_zp{Z}.{ext}      ex.: 00001_x1_y1_zp1.jpg

Este módulo:
  1. Detecta a convenção de nomenclatura atualmente usada no diretório de tiles
     (testando um conjunto de padrões conhecidos).
  2. Emite um relatório com a distribuição dos padrões, quantos já estão no
     formato canônico, quantos podem ser renomeados e quais não foram
     reconhecidos.
  3. Renomeia (ou copia) os arquivos reconhecidos para o formato canônico.

Segurança:
  - Por padrão roda em modo *dry-run* (apenas relatório, não altera nada).
  - Use --apply para efetivar. Use --dest para copiar em vez de renomear no
    lugar (não destrutivo).
  - Ao aplicar, grava um mapa old->new em JSON para auditoria/reversão.

Uso (CLI):
    python -m src.modules.tile.rename.rename                 # dry-run em Config.TILES_DIR
    python -m src.modules.tile.rename.rename --apply          # renomeia no lugar
    python -m src.modules.tile.rename.rename --dir <path> --dest <out> --apply
"""

import argparse
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Padrões conhecidos de nomenclatura
# ---------------------------------------------------------------------------
# Cada padrão captura os grupos nomeados 'x' (coluna) e 'y' (linha). São
# testados em ordem de prioridade; o primeiro que casar define a extração.
# O padrão 'canonical' também captura 'prefix' e 'z'.
CANONICAL_RE = re.compile(r"^(?P<prefix>\d+)_x(?P<x>\d+)_y(?P<y>\d+)_zp(?P<z>\d+)$")

KNOWN_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # Já no formato canônico exato.
    ("canonical", CANONICAL_RE),
    # x/y rotulados em qualquer posição: "img_x3_y4", "x3_y4_z1", "..._x03_y10_..."
    ("xy_labeled", re.compile(r"(?:^|[_-])x(?P<x>\d+)[_-]?y(?P<y>\d+)(?:[_-]|$)", re.IGNORECASE)),
    # coluna/linha: "col2_row3", "c2_r3"
    ("col_row", re.compile(r"(?:^|[_-])(?:col|c)(?P<x>\d+)[_-]?(?:row|r)(?P<y>\d+)(?:[_-]|$)", re.IGNORECASE)),
    # linha/coluna: "row3_col2", "r3_c2"
    ("row_col", re.compile(r"(?:^|[_-])(?:row|r)(?P<y>\d+)[_-]?(?:col|c)(?P<x>\d+)(?:[_-]|$)", re.IGNORECASE)),
    # "tile_3_4" / "tile-3-4"
    ("tile_x_y", re.compile(r"tile[_-](?P<x>\d+)[_-](?P<y>\d+)", re.IGNORECASE)),
    # dois números separados: "3_4", "3-4" (ambíguo: assume x_y)
    ("bare_x_y", re.compile(r"^(?P<x>\d+)[_-](?P<y>\d+)$")),
]

# Captura de z-plane opcional em nomes não-canônicos (ex.: "_zp2", "_z3")
_Z_RE = re.compile(r"[_-]z[p]?(?P<z>\d+)(?:[_-]|$)", re.IGNORECASE)


@dataclass
class FileAnalysis:
    """Resultado da análise de um único arquivo."""

    path: Path
    pattern: Optional[str]  # nome do padrão que casou, ou None
    x: Optional[int]
    y: Optional[int]
    z: int

    @property
    def recognized(self) -> bool:
        return self.pattern is not None

    @property
    def is_canonical(self) -> bool:
        return self.pattern == "canonical"


@dataclass
class DirectoryReport:
    """Resumo do diagnóstico de um diretório."""

    directory: Path
    total: int = 0
    pattern_counts: Dict[str, int] = field(default_factory=dict)
    unknown: List[Path] = field(default_factory=list)

    @property
    def canonical(self) -> int:
        return self.pattern_counts.get("canonical", 0)

    @property
    def renamable(self) -> int:
        # Reconhecidos que não são canônicos.
        return sum(
            c for name, c in self.pattern_counts.items() if name != "canonical"
        )

    @property
    def dominant_non_canonical(self) -> Optional[str]:
        non_canon = {
            name: c for name, c in self.pattern_counts.items() if name != "canonical"
        }
        if not non_canon:
            return None
        return max(non_canon, key=non_canon.get)

    def log(self) -> None:
        logger.info("=== Diagnóstico de nomenclatura ===")
        logger.info(f"Diretório           : {self.directory}")
        logger.info(f"Total de imagens    : {self.total}")
        logger.info(f"Já canônicos        : {self.canonical}")
        logger.info(f"Renomeáveis         : {self.renamable}")
        logger.info(f"Não reconhecidos    : {len(self.unknown)}")
        logger.info("Distribuição por padrão:")
        for name, count in sorted(self.pattern_counts.items(), key=lambda kv: -kv[1]):
            logger.info(f"  - {name:<12}: {count}")
        if self.dominant_non_canonical:
            logger.info(f"Convenção detectada : {self.dominant_non_canonical}")
        if self.unknown:
            sample = ", ".join(p.name for p in self.unknown[:5])
            logger.warning(f"Exemplos não reconhecidos (serão ignorados): {sample}")


def _extract_z(stem: str, default_z: int) -> int:
    """Tenta extrair o z-plane do nome; retorna default se ausente."""
    m = _Z_RE.search(stem)
    if m:
        return int(m.group("z"))
    return default_z


def analyze_filename(stem: str, default_z: int = 1) -> Tuple[Optional[str], Optional[int], Optional[int], int]:
    """Analisa o *stem* (nome sem extensão) e retorna (padrão, x, y, z).

    Retorna (None, None, None, default_z) se nenhum padrão conhecido casar.
    """
    for name, pattern in KNOWN_PATTERNS:
        m = pattern.search(stem)
        if not m:
            continue
        groups = m.groupdict()
        x = int(groups["x"])
        y = int(groups["y"])
        if name == "canonical":
            z = int(groups["z"])
        else:
            z = _extract_z(stem, default_z)
        return name, x, y, z
    return None, None, None, default_z


def list_image_files(directory: Path) -> List[Path]:
    """Lista arquivos de imagem no diretório (extensão-agnóstico, ordenado)."""
    if not directory.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {directory}")
    exts = set(Config.SUPPORTED_EXTENSIONS)
    paths = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]
    return sorted(paths)


def analyze_directory(
    directory: Path, default_z: int = 1
) -> Tuple[List[FileAnalysis], DirectoryReport]:
    """Analisa todos os tiles do diretório e produz a lista de análises + relatório."""
    files = list_image_files(directory)
    analyses: List[FileAnalysis] = []
    report = DirectoryReport(directory=directory, total=len(files))

    for path in files:
        pattern, x, y, z = analyze_filename(path.stem, default_z)
        analyses.append(FileAnalysis(path=path, pattern=pattern, x=x, y=y, z=z))
        if pattern is None:
            report.unknown.append(path)
        else:
            report.pattern_counts[pattern] = report.pattern_counts.get(pattern, 0) + 1

    return analyses, report


def _max_canonical_prefix(analyses: List[FileAnalysis]) -> int:
    """Maior prefixo numérico entre arquivos já canônicos (0 se nenhum)."""
    prefixes = []
    for a in analyses:
        if a.is_canonical:
            m = CANONICAL_RE.match(a.path.stem)
            if m:
                prefixes.append(int(m.group("prefix")))
    return max(prefixes) if prefixes else 0


def build_rename_plan(
    analyses: List[FileAnalysis],
    prefix_digits: int = 5,
) -> Dict[Path, str]:
    """Constrói o plano de renomeação {caminho_origem: novo_nome}.

    - Arquivos já canônicos são preservados (não entram no plano).
    - Arquivos não reconhecidos são ignorados (não entram no plano).
    - Os demais recebem prefixo sequencial em ordem raster (y, depois x),
      começando acima do maior prefixo canônico existente (evita colisão).
    """
    start = _max_canonical_prefix(analyses) + 1

    # Reconhecidos e não-canônicos, ordenados por (y, x) para ordem raster.
    renamable = [a for a in analyses if a.recognized and not a.is_canonical]
    renamable.sort(key=lambda a: (a.y, a.x))

    plan: Dict[Path, str] = {}
    for i, a in enumerate(renamable):
        prefix = start + i
        ext = a.path.suffix.lower()
        new_name = f"{prefix:0{prefix_digits}d}_x{a.x}_y{a.y}_zp{a.z}{ext}"
        plan[a.path] = new_name

    return plan


def _validate_canonical(name: str) -> bool:
    """Confere que o nome gerado casa com Config.COORDINATES_PATTERN."""
    return re.search(Config.COORDINATES_PATTERN, name) is not None


def apply_rename_plan(
    plan: Dict[Path, str],
    directory: Path,
    dest_dir: Optional[Path] = None,
    dry_run: bool = True,
) -> Dict[str, str]:
    """Executa (ou simula) o plano de renomeação.

    Args:
        plan: mapa {caminho_origem: novo_nome}.
        directory: diretório de origem (usado para salvar o mapa de auditoria).
        dest_dir: se informado, copia para lá com o novo nome (não destrutivo);
                  caso contrário, renomeia no lugar.
        dry_run: se True (padrão), apenas registra o que seria feito.

    Returns:
        Mapa {nome_antigo: nome_novo} efetivamente aplicado (ou que seria).
    """
    mapping: Dict[str, str] = {}

    # Sanidade: garantir que todos os novos nomes casam com o padrão do pipeline.
    invalid = [n for n in plan.values() if not _validate_canonical(n)]
    if invalid:
        raise ValueError(
            f"Nomes gerados não casam com COORDINATES_PATTERN: {invalid[:3]}"
        )

    if dest_dir is not None:
        dest_dir.mkdir(parents=True, exist_ok=True)

    for src_path, new_name in plan.items():
        target_dir = dest_dir if dest_dir is not None else directory
        target = target_dir / new_name

        if target.exists() and target != src_path:
            logger.error(
                f"Colisão: destino já existe, pulando: {new_name} (origem {src_path.name})"
            )
            continue

        mapping[src_path.name] = new_name
        action = "copiar" if dest_dir is not None else "renomear"
        if dry_run:
            logger.info(f"[dry-run] {action}: {src_path.name} -> {new_name}")
            continue

        if dest_dir is not None:
            shutil.copy2(str(src_path), str(target))
        else:
            # Renomeação em duas fases evita colisões A->B / B->A dentro do lote.
            tmp = src_path.with_name(f".__tmp__{src_path.name}")
            src_path.rename(tmp)
            tmp.rename(target)

    if not dry_run and mapping:
        map_path = directory.parent / "rename_map.json"
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        logger.info(f"Mapa de renomeação salvo em: {map_path}")

    return mapping


def run_rename(
    directory: Optional[Path] = None,
    apply: bool = False,
    dest_dir: Optional[Path] = None,
) -> DirectoryReport:
    """Orquestra diagnóstico + renomeação.

    Args:
        directory: diretório de tiles (padrão: Config.TILES_DIR).
        apply: se False (padrão), roda em dry-run (só relatório).
        dest_dir: se informado, copia em vez de renomear no lugar.

    Returns:
        DirectoryReport com o diagnóstico.
    """
    directory = Path(directory) if directory else Config.TILES_DIR
    default_z = int(getattr(Config, "TILE_DEFAULT_ZP", 1))
    prefix_digits = int(getattr(Config, "TILE_PREFIX_DIGITS", 5))

    analyses, report = analyze_directory(directory, default_z=default_z)
    report.log()

    if report.total == 0:
        logger.warning(f"Nenhuma imagem encontrada em {directory}. Nada a fazer.")
        return report

    plan = build_rename_plan(analyses, prefix_digits=prefix_digits)

    if not plan:
        logger.info(
            "Nenhum arquivo a renomear (todos já canônicos ou não reconhecidos)."
        )
        return report

    logger.info(
        f"{'[dry-run] ' if not apply else ''}"
        f"{len(plan)} arquivo(s) serão {'copiados' if dest_dir else 'renomeados'}."
    )
    apply_rename_plan(plan, directory, dest_dir=dest_dir, dry_run=not apply)

    if not apply:
        logger.info("Modo dry-run: nada foi alterado. Use --apply para efetivar.")

    return report


def main() -> None:
    logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Diagnostica e padroniza a nomenclatura dos tiles para o formato canônico."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Diretório de tiles (padrão: Config.TILES_DIR).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Efetiva a renomeação. Sem esta flag, roda em dry-run (só relatório).",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Diretório de destino. Se informado, copia em vez de renomear no lugar.",
    )
    args = parser.parse_args()

    directory = Path(args.dir) if args.dir else None
    dest_dir = Path(args.dest) if args.dest else None

    run_rename(directory=directory, apply=args.apply, dest_dir=dest_dir)


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")
