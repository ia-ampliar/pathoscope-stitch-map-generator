import json
import logging
import re
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import pickle
from matplotlib.patches import Patch

from src.config.config import Config
from src.utils.coordinates import extract_coordinates


logger = logging.getLogger(__name__)

def save_graph(G: nx.Graph, path: Path) -> None:
    """
    Persiste um grafo do NetworkX em disco usando o formato 'gpickle'.

    Por que gpickle?
    - Salva o grafo completo (nós, arestas e atributos) sem precisar conversões.
    - Ideal para pipeline interno (rápido e simples).
    - Mantém tipos Python (ex.: nós como tuplas (x, y)).

    Parâmetros:
        G (nx.Graph): Grafo a ser salvo.
        path (Path): Caminho do arquivo de saída (ex.: Config.TOPOLOGY_GRAPH_FILE).

    Efeito:
        - Cria o diretório pai se não existir.
        - Salva o grafo no caminho informado.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Usando pickle diretamente para maior controle
    with open(path, "wb") as f:
        pickle.dump(G, f)

    logger.info(f"Grafo salvo em: {path}")


def load_graph(path: Path) -> nx.Graph:
    """
    Carrega um grafo do NetworkX a partir de um arquivo 'gpickle'.

    Parâmetros:
        path (Path): Caminho do arquivo (.gpickle).

    Retorna:
        nx.Graph: O grafo carregado do disco.

    Observações:
        - Se o arquivo não existir, essa função levanta FileNotFoundError.
          (Isso é bom para falhar cedo e você decidir como tratar no fluxo principal.)
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de grafo não encontrado: {path}")

    # Usando pickle diretamente para maior controle
    with open(path, "rb") as f:
        G = pickle.load(f)
    
    logger.info(f"Grafo carregado de: {path}")
    return G


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


def get_or_build_topological_graph(
    valid_tiles: Dict[str, bool],
    pattern: re.Pattern,
    force_rebuild: bool = False,
) -> nx.Graph:
    """
    Retorna o grafo topológico persistido em disco, se existir.
    Caso contrário (ou se force_rebuild=True), constrói um novo,
    salva em disco e retorna.

    Parâmetros:
        valid_tiles (Dict[str, bool]): Dicionário de tiles válidos.
        pattern (re.Pattern): Regex para extração das coordenadas.
        force_rebuild (bool): Se True, ignora o grafo salvo e recria.

    Retorna:
        nx.Graph: Grafo topológico pronto para uso.
    """
    graph_path = Config.TOPOLOGY_GRAPH_FILE

    # Caso 1: usar grafo salvo em disco
    if graph_path.exists() and not force_rebuild:
        logger.info("Grafo topológico encontrado em disco. Carregando...")
        return load_graph(graph_path)

    # Caso 2: construir novo grafo
    logger.info("Construindo novo grafo topológico...")
    G = build_graph(valid_tiles, pattern)

    # Persistir em disco
    save_graph(G, graph_path)

    return G


def generate_graph():
    # Gera o grafo de tiles e salva a imagem
    pattern = re.compile(Config.COORDINATES_PATTERN)

    # Carrega os tiles válidos
    valid_tiles = load_valid_tiles(Config.VALID_TILES_FILE)

    # Obtém ou constrói o grafo topológico
    graph = get_or_build_topological_graph(valid_tiles, pattern)

    # Gera a imagem do grafo
    plot_graph(graph, Config.GRAPH_FILE)

    # logger.info(f"Grafo criado em: {Config.GRAPH_FILE}")


if __name__ == "__main__":
    # Configura o logging básico
    logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.DEBUG)

    # Suprime avisos do matplotlib
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    generate_graph()
