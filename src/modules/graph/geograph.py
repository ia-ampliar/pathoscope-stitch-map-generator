from typing import Optional, Tuple

import time
import numpy as np
import zarr
import networkx as nx
from pathlib import Path

import logging
import matplotlib.pyplot as plt

from src.modules.graph.graph import load_valid_tiles, load_graph, save_graph

from src.config.config import Config


logger = logging.getLogger(__name__)

def _find_match_zarr_path(
    matches_dir: Path, 
    tile_a: str, 
    tile_b: str
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Procura o arquivo .zarr do match entre dois tiles.

    Retorna:
        (path, direction)
        - path: caminho do .zarr encontrado ou None
        - direction:
            "A__B" se encontrou tile_a__tile_b.zarr (transformação A -> B)
            "B__A" se encontrou tile_b__tile_a.zarr (transformação B -> A)
            None se não encontrou
    """
    path_a_b = matches_dir / f"{tile_a}__{tile_b}.zarr"
    if path_a_b.exists():
        return path_a_b, "A__B"

    path_b_a = matches_dir / f"{tile_b}__{tile_a}.zarr"
    if path_b_a.exists():
        return path_b_a, "B__A"

    return None, None


def build_geometric_graph_translation(
    G_topo: nx.Graph,
    matches_dir: Path,
) -> nx.DiGraph:
    """
    Constrói um grafo geométrico (dirigido) a partir do grafo topológico (vizinhança),
    lendo os arquivos .zarr de match e extraindo apenas a translação (dx, dy).

    A lógica:
    - Para cada aresta (u, v) do grafo topológico:
        - Recupera os labels (nomes dos tiles) em u e v.
        - Procura o arquivo .zarr do par:
            - Se existir "tile_u__tile_v.zarr", a translação lida é u -> v.
            - Se existir "tile_v__tile_u.zarr", a translação lida é v -> u,
              então invertemos o sinal para obter u -> v.
        - Adiciona arestas dirigidas nos dois sentidos:
            u -> v com (dx, dy)
            v -> u com (-dx, -dy)
    - Se não existir .zarr para o par, NÃO cria arestas (como você pediu).

    Parâmetros:
        G_topo (nx.Graph): grafo topológico já criado/carregado (nós (x,y) com attrs label/valid).
        matches_dir (Path): diretório contendo os .zarr (ex: Path("output/matches")).

    Retorna:
        nx.DiGraph: grafo geométrico dirigido com pesos (dx, dy) nas arestas.
    """

    if not matches_dir.exists():
        raise FileNotFoundError(f"Diretório de matches não encontrado: {matches_dir}")

    # Grafo geométrico: dirigido, pois translação tem sentido (u->v != v->u)
    G_geo = nx.DiGraph()

    # Copiar nós e atributos do grafo topológico (mantém label/valid, etc.)
    for node, attrs in G_topo.nodes(data=True):
        G_geo.add_node(node, **attrs)

    # Percorrer todas as vizinhanças do grafo topológico
    for u, v in G_topo.edges():
        # Labels (nomes dos tiles) precisam existir no nó
        tile_u = G_topo.nodes[u].get("label")
        tile_v = G_topo.nodes[v].get("label")

        if not tile_u or not tile_v:
            logger.warning(
                f"Aresta ({u}, {v}) ignorada: nó sem atributo 'label' (tile_u={tile_u}, tile_v={tile_v})."
            )
            continue

        # Encontrar o .zarr correspondente ao par
        zarr_path, direction = _find_match_zarr_path(matches_dir, tile_u, tile_v)

        # Regra solicitada: se não existe .zarr do par, NÃO cria a aresta
        if zarr_path is None:
            logger.warning(
                f"Match .zarr não encontrado para o par: {tile_u} <-> {tile_v}. Aresta geométrica não criada."
            )
            continue

        # Abrir o zarr e ler a matriz de translação
        root = zarr.open(str(zarr_path), mode="r")
        matches_group = root["matches"]
        group = zarr.open_group(str(zarr_path), mode="r")

        if "translation_matrix" not in matches_group.attrs:
            logger.warning(
                f"Arquivo {zarr_path.name} não contém attrs['translation_matrix']. Aresta não criada."
            )
            continue

        M = np.array(matches_group.attrs["translation_matrix"], dtype=np.float64)

        # Esperamos uma matriz 3x3 homogênea (translação pura)
        # [1 0 dx]
        # [0 1 dy]
        # [0 0  1]
        dx = float(M[0, 2])
        dy = float(M[1, 2])

        # Se o arquivo era tile_u__tile_v (A__B), então a translação lida é u -> v.
        # Para obter u -> v, invertemos o sinal.
        if direction == "A__B":
            dx, dy = -dx, -dy

        # Criar as duas direções no grafo geométrico
        # u -> v
        G_geo.add_edge(
            u,
            v,
            dx=dx,
            dy=dy,
            match_file=str(zarr_path),
            match_direction=direction,  # útil para debug
        )

        # v -> u (inverso exato da translação)
        G_geo.add_edge(
            v,
            u,
            dx=-dx,
            dy=-dy,
            match_file=str(zarr_path),
            match_direction=direction,
        )

    logger.info(
        f"Grafo geométrico criado: {G_geo.number_of_nodes()} nós, {G_geo.number_of_edges()} arestas dirigidas."
    )
    return G_geo


def plot_geometric_graph_with_weights(
    G_geo: nx.DiGraph,
    output_path: Path,
    show_arrows: bool = False,
    decimals: int = 1,
) -> None:
    """
    Plota o grafo geométrico com os pesos (dx, dy) nas arestas.

    Estratégia padrão (show_arrows=False):
      - Converte para grafo não-dirigido para evitar duplicar (u->v e v->u).
      - Mostra apenas 1 label por par.

    Se show_arrows=True:
      - Desenha o DiGraph com setas.
      - Ainda assim aplica um filtro simples de labels para não duplicar tudo.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Posição dos nós: baseada nas coordenadas do próprio nó (x, y)
    # Inverte Y para ficar “de cima pra baixo” como no seu plot topológico.
    pos = {node: (node[0], -node[1]) for node in G_geo.nodes()}

    plt.figure(figsize=(10, 8))

    if not show_arrows:
        # Visualização sem duplicar direções
        G_und = G_geo.to_undirected()

        # Cria labels de aresta pegando um sentido “canônico” (u->v se existir)
        edge_labels = {}
        for u, v in G_und.edges():
            # escolhe um sentido consistente para pegar (dx, dy)
            if G_geo.has_edge(u, v):
                dx = G_geo.edges[u, v]["dx"]
                dy = G_geo.edges[u, v]["dy"]
            else:
                # se por algum motivo só existir v->u, invertimos
                dx = -G_geo.edges[v, u]["dx"]
                dy = -G_geo.edges[v, u]["dy"]

            edge_labels[(u, v)] = f"({dx:.{decimals}f},{dy:.{decimals}f})"

        nx.draw(
            G_und,
            pos,
            with_labels=True,
            node_size=900,
            font_size=8,
        )
        nx.draw_networkx_edge_labels(
            G_und,
            pos,
            edge_labels=edge_labels,
            font_size=7,
        )

    else:
        # Visualização com setas (pode ficar mais carregado)
        # Para reduzir duplicação visual, só rotula arestas num critério simples.
        edge_labels = {}
        for u, v in G_geo.edges():
            dx = G_geo.edges[u, v]["dx"]
            dy = G_geo.edges[u, v]["dy"]

            # filtro anti-duplicação simples: só rotula se (dx > 0) ou (dx==0 e dy > 0)
            if (dx > 0) or (dx == 0 and dy > 0):
                edge_labels[(u, v)] = f"({dx:.{decimals}f},{dy:.{decimals}f})"

        nx.draw(
            G_geo,
            pos,
            with_labels=True,
            node_size=900,
            font_size=8,
            arrows=True,
            arrowsize=15,
        )
        nx.draw_networkx_edge_labels(
            G_geo,
            pos,
            edge_labels=edge_labels,
            font_size=7,
        )

    plt.title("Geometric graph (translation dx, dy)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_geometric_graph():

    # Obtém ou constrói o grafo topológico
    G_topo = load_graph(Config.TOPOLOGY_GRAPH_FILE)

    # Constrói o grafo geométrico a partir do topológico
    G_geo = build_geometric_graph_translation(G_topo, matches_dir=Config.MATCHING_ZARR_PATH)

    # Loga informações sobre o grafo geométrico criado
    logger.info(f"Grafo geométrico criado com {G_geo.number_of_nodes()} nós e {G_geo.number_of_edges()} arestas.")

    # Salva o grafo geométrico em disco
    geo_graph_path = Config.GEOMETRIC_GRAPH_FILE
    save_graph(G_geo, geo_graph_path)

    plot_geometric_graph_with_weights(
        G_geo,
        output_path=Config.GEOMETRIC_GRAPH_WEIGHTS_FILE,
        show_arrows=False, 
    )


if __name__ == "__main__":

    generate_geometric_graph()
    start_time = time.perf_counter()
    
    # Configura o logging básico
    logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.DEBUG)

    # Suprime avisos do matplotlib
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")

    
