"""
contracts.py — Camada de validação de artefatos entre etapas do pipeline.

Cada função carrega um artefato de disco e valida esquema/forma antes de
retorná-lo. Falha cedo com mensagem acionável em vez de propagar KeyError
ou IndexError genéricos em etapas posteriores.

Uso típico:
    from src.utils.contracts import load_metadata, load_valid_tiles, ...
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Tipos reutilizáveis
Node = Tuple[int, int]
PosXY = Tuple[float, float]
Positions = Dict[Node, PosXY]


# ---------------------------------------------------------------------------
# dataset.json (saída do fetch, entrada do detect)
# ---------------------------------------------------------------------------

def load_metadata(path: Path) -> List[Dict[str, Any]]:
    """Carrega e valida o JSON de metadados de tiles (dataset.json).

    Esquema esperado: lista de dicts com chaves 'name', 'path', 'coordinates'.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"[contracts] Metadados não encontrados: {path}. "
            f"Execute a etapa 'fetch' primeiro."
        )

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"[contracts] {path}: esperado lista de dicts, obtido {type(data).__name__}."
        )

    required_keys = {"name", "path", "coordinates"}
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(
                f"[contracts] {path}[{i}]: esperado dict, obtido {type(entry).__name__}."
            )
        missing = required_keys - entry.keys()
        if missing:
            raise ValueError(
                f"[contracts] {path}[{i}]: chaves ausentes: {missing}."
            )
        coords = entry["coordinates"]
        if not (isinstance(coords, (list, tuple)) and len(coords) == 2):
            raise ValueError(
                f"[contracts] {path}[{i}]: 'coordinates' deve ter 2 elementos [x, y], "
                f"obtido {coords!r}."
            )

    logger.debug(f"[contracts] Metadados carregados: {len(data)} tiles de {path}")
    return data


# ---------------------------------------------------------------------------
# valid_tiles.json (saída do classify, entrada do detect/match/graph)
# ---------------------------------------------------------------------------

def load_valid_tiles(path: Path) -> Dict[str, bool]:
    """Carrega e valida o JSON de tiles válidos.

    Esquema esperado: dict {tile_name: bool}.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"[contracts] Arquivo de tiles válidos não encontrado: {path}. "
            f"Execute a etapa 'classify' primeiro."
        )

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"[contracts] {path}: esperado dict, obtido {type(data).__name__}."
        )

    # Validação leve: checar que todos os valores são booleanos
    non_bool = [k for k, v in data.items() if not isinstance(v, bool)]
    if non_bool:
        logger.warning(
            f"[contracts] {path}: {len(non_bool)} entradas com valor não-booleano "
            f"(ex: {non_bool[:3]}). Serão interpretadas via truthiness."
        )

    logger.debug(f"[contracts] Valid tiles carregados: {len(data)} de {path}")
    return data


# ---------------------------------------------------------------------------
# global_positions.pkl (saída do globalpos, entrada do create_geom/populate_geom)
# ---------------------------------------------------------------------------

def load_positions(path: Path) -> Positions:
    """Carrega e valida o pickle de posições globais.

    Esquema esperado: dict { (int, int): (float, float) }.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"[contracts] Posições globais não encontradas: {path}. "
            f"Execute a etapa 'globalpos' primeiro."
        )

    with path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"[contracts] {path}: esperado dict, obtido {type(data).__name__}."
        )

    if not data:
        raise ValueError(f"[contracts] {path}: dicionário de posições está vazio.")

    # Validação de amostra (primeiros 3 itens)
    sample = list(data.items())[:3]
    for node, pos in sample:
        if not (isinstance(node, tuple) and len(node) == 2):
            raise ValueError(
                f"[contracts] {path}: chave inválida {node!r} — esperado tuple (x, y)."
            )
        if not (isinstance(pos, tuple) and len(pos) == 2):
            raise ValueError(
                f"[contracts] {path}: valor inválido para nó {node}: {pos!r} — "
                f"esperado tuple (X, Y) em pixels."
            )

    logger.debug(f"[contracts] Posições carregadas: {len(data)} nós de {path}")
    return data


# ---------------------------------------------------------------------------
# graph pickle (saída do graph/geograph, entrada de etapas subsequentes)
# ---------------------------------------------------------------------------

def load_graph(path: Path, expected_type: str = "any"):
    """Carrega e valida um grafo NetworkX salvo em pickle.

    Args:
        path: caminho do arquivo .gpickle.
        expected_type: 'directed' | 'undirected' | 'any'.
    """
    import networkx as nx

    if not path.exists():
        raise FileNotFoundError(
            f"[contracts] Grafo não encontrado: {path}. "
            f"Execute a etapa de grafo correspondente primeiro."
        )

    with path.open("rb") as f:
        G = pickle.load(f)

    if not isinstance(G, (nx.Graph, nx.DiGraph)):
        raise ValueError(
            f"[contracts] {path}: esperado nx.Graph ou nx.DiGraph, "
            f"obtido {type(G).__name__}."
        )

    if expected_type == "directed" and not isinstance(G, nx.DiGraph):
        raise ValueError(
            f"[contracts] {path}: esperado nx.DiGraph (grafo dirigido), "
            f"obtido {type(G).__name__}."
        )
    if expected_type == "undirected" and isinstance(G, nx.DiGraph):
        raise ValueError(
            f"[contracts] {path}: esperado nx.Graph (não-dirigido), "
            f"obtido nx.DiGraph."
        )

    if G.number_of_nodes() == 0:
        raise ValueError(f"[contracts] {path}: grafo está vazio (0 nós).")

    logger.debug(
        f"[contracts] Grafo carregado de {path}: "
        f"{G.number_of_nodes()} nós, {G.number_of_edges()} arestas"
    )
    return G


# ---------------------------------------------------------------------------
# match zarr (saída do match, lido pelo geograph)
# ---------------------------------------------------------------------------

def validate_match_zarr(zarr_path: Path) -> bool:
    """Valida estrutura mínima de um arquivo .zarr de match.

    Retorna True se a estrutura está íntegra, False caso contrário (com log).
    """
    import zarr as zarr_lib

    try:
        root = zarr_lib.open(str(zarr_path), mode="r")
        grp = root["matches"]
        if "translation_matrix" not in grp.attrs:
            logger.warning(f"[contracts] {zarr_path.name}: sem 'translation_matrix' nos attrs")
            return False
        M = np.array(grp.attrs["translation_matrix"])
        if M.shape != (3, 3):
            logger.warning(f"[contracts] {zarr_path.name}: translation_matrix shape={M.shape}, esperado (3,3)")
            return False
        return True
    except Exception as e:
        logger.warning(f"[contracts] {zarr_path.name}: falha ao validar — {e}")
        return False
