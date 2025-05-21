import json
import logging
import re
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch

from src.config.config import Config
from src.utils.coordinates import extract_coordinates

logger = logging.getLogger(__name__)


def load_valid_tiles(path: Path) -> Dict[str, bool]:
    """
    Carrega o dicionário de tiles válidos a partir de um arquivo JSON.

    Parâmetros:
        path (Path): Caminho do arquivo JSON

    Retorna:
        Dict[str, bool]: Dicionário com o nome dos tiles e sua validade
    """
    with open(path) as f:
        return json.load(f)


def build_graph(valid_tiles: Dict[str, bool], pattern: re.Pattern) -> nx.Graph:
    """
    Constrói um grafo onde cada nó representa um tile.
    Conecta os nós vizinhos com base em coordenadas (4 vizinhos).

    Parâmetros:
        valid_tiles (Dict[str, bool]): Dicionário de tiles válidos
        pattern (re.Pattern): Expressão regular para extração de coordenadas

    Retorna:
        nx.Graph: Grafo com nós e conexões de vizinhança
    """
    G = nx.Graph()

    # Adiciona os nós ao grafo
    for tile_name, is_valid in valid_tiles.items():
        coord = extract_coordinates(tile_name, pattern)
        if coord:
            G.add_node(coord, valid=is_valid, label=tile_name)

    # Conecta os vizinhos (topo, baixo, esquerda, direita)
    for x, y in G.nodes:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (x + dx, y + dy)
            if neighbor in G.nodes:
                G.add_edge((x, y), neighbor)

    return G


def plot_graph(G: nx.Graph, output_path: Path, spacing: int = 100):
    """
    Gera e salva a imagem do grafo com visualização top-down.
    Nós são coloridos conforme a validade do tile.

    Parâmetros:
        G (nx.Graph): Grafo a ser plotado
        output_path (Path): Caminho de saída da imagem
        spacing (int): Espaçamento visual entre os nós
    """
    pos = {
        (x, y): (x * spacing, -y * spacing) for (x, y) in G.nodes
    }  # Inverte Y para exibir top-down
    node_colors = ["blue" if G.nodes[n]["valid"] else "red" for n in G.nodes]
    labels = {n: G.nodes[n]["label"] for n in G.nodes}

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, edge_color="#aaaaaa")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=4)

    # Legenda
    legend_elements = [
        Patch(facecolor="blue", edgecolor="black", label="Registrado"),
        Patch(facecolor="red", edgecolor="black", label="Não registrado"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    plt.title(
        "Grafo de Vizinhança dos Tiles (Azul = Registrado, Vermelho = Não Registrado)"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_graph():
    pattern = re.compile(Config.COORDINATES_PATTERN)
    valid_tiles = load_valid_tiles(Config.VALID_TILES_FILE)
    graph = build_graph(valid_tiles, pattern)
    plot_graph(graph, Config.GRAPH_FILE)
    logger.info(f"Grafo criado em: {Config.GRAPH_FILE}")


if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.DEBUG)
    generate_graph()
