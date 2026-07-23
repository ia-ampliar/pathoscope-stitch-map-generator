# src/modules/mosaic/draw_mosaic.py

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple
import networkx as nx

from src.config.config import Config

from src.modules.graph.graph import load_graph



logger = logging.getLogger(__name__)

Node = Tuple[int, int]
PosXY = Tuple[float, float]
Positions = Dict[Node, PosXY]


def load_global_positions(path: Path) -> Positions:
    """
    Carrega o arquivo global_positions.pkl gerado pelo módulo globalpos.py.

    O conteúdo esperado é:
      dict { (x, y): (X, Y) }
    onde:
      - (x, y) são coordenadas do nó (grid)
      - (X, Y) são coordenadas globais em pixels (já normalizadas)
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo global_positions.pkl não encontrado em: {path}")

    with path.open("rb") as f:
        positions = pickle.load(f)

    if not isinstance(positions, dict):
        raise ValueError(f"Conteúdo inválido em {path}: esperado dict, obtido {type(positions)}")

    logger.info(f"Posições globais carregadas de: {path} (total={len(positions)})")
    return positions


def load_geometric_graph(path: Path) -> nx.DiGraph:
    """
    Carrega o grafo geométrico salvo em disco (pickle), usando a função load_graph do graph.py.

    Esperado:
      - nx.DiGraph
      - nós com atributo "label" (ex: "00003_x3_y1_zp1")
      - arestas com dx/dy (não usamos aqui ainda, mas faz parte do grafo)
    """
    G = load_graph(path)

    if not isinstance(G, nx.DiGraph):
        raise ValueError(f"Grafo em {path} não é nx.DiGraph. Tipo encontrado: {type(G)}")

    logger.info(
        f"Grafo geométrico carregado de: {path} (nós={G.number_of_nodes()}, arestas={G.number_of_edges()})"
    )
    return G


def resolve_tile_paths_from_labels(
    G_geo: nx.DiGraph,
    positions: Positions,
    tiles_dir: Path,
) -> Dict[Node, Path]:
    """
    Resolve o caminho da imagem de cada nó usando o atributo "label" do nó no grafo geométrico.

    Estratégia:
      - Para cada nó presente em positions:
          - pega label = G_geo.nodes[node]["label"]
          - procura em tiles_dir por arquivos com padrão: f"{label}.*"
          - exige:
              - exatamente 1 arquivo encontrado
              - caso 0: erro (tile faltando)
              - caso >1: erro (ambiguidade) [você disse que não ocorre]

    Retorna:
      dict { node: Path(arquivo_da_imagem) }
    """
    if not tiles_dir.exists():
        raise FileNotFoundError(f"TILES_DIR não encontrado: {tiles_dir}")

    node_to_path: Dict[Node, Path] = {}

    for node in positions.keys():
        if node not in G_geo.nodes:
            raise KeyError(
                f"Nó {node} existe em positions mas não existe no grafo geométrico."
            )

        label = G_geo.nodes[node].get("label")
        if not label:
            raise KeyError(f"Nó {node} não possui atributo 'label' no grafo geométrico.")

        matches = list(tiles_dir.glob(f"{label}.*"))

        if len(matches) == 0:
            raise FileNotFoundError(
                f"Nenhuma imagem encontrada para label='{label}' em {tiles_dir} (padrão '{label}.*')."
            )

        if len(matches) > 1:
            raise ValueError(
                f"Ambiguidade: {len(matches)} arquivos encontrados para label='{label}' em {tiles_dir}: {matches}"
            )

        node_to_path[node] = matches[0]

    logger.info(f"Mapeamento node -> imagem concluído (total={len(node_to_path)}).")
    return node_to_path


def main() -> None:
    """
    draw_mosaic - passo 2:
      - carrega positions globais
      - carrega grafo geométrico (para obter label por nó)
      - resolve caminho das imagens em TILES_DIR via label.*
    """
    positions = load_global_positions(Config.GLOBAL_POS_FILE)

    G_geo = load_geometric_graph(Config.GEOMETRIC_GRAPH_FILE)

    node_to_path = resolve_tile_paths_from_labels(
        G_geo=G_geo,
        positions=positions,
        tiles_dir=Config.TILES_DIR,
    )

    # Debug: mostrar 3 exemplos
    sample = list(node_to_path.items())[:3]
    logger.info(f"Amostra node->path: {sample}")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] - %(message)s")
    main()
