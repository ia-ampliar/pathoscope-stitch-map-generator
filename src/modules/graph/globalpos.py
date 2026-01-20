# src/modules/graph/globalpos.py

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import networkx as nx
from collections import deque

from src.config.config import Config


logger = logging.getLogger(__name__)

# % Tipos (apenas para deixar o código mais legível)
Node = Tuple[int, int]              # seus nós são coordenadas (x, y) do grid topológico
PosXY = Tuple[float, float]         # posição global em pixels (X, Y)
Positions = Dict[Node, PosXY]       # mapa de posições globais por nó


def save_positions(positions: Positions, path: Path) -> None:
    """
    Salva o dicionário de posições globais em disco usando pickle.

    positions:
        dict { node: (X, Y) }
        - node é uma tupla (x, y) do grid
        - (X, Y) são floats em pixels

    path:
        caminho do arquivo .pkl (ex.: Config.GLOBAL_POS_FILE)

    Observação:
        Usar pickle evita dor de cabeça com serialização de tuplas (JSON não suporta tupla direto).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(positions, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Posições globais salvas em: {path}")


def load_positions(path: Path) -> Positions:
    """
    Carrega o dicionário de posições globais salvo via pickle.

    Retorna:
        positions: dict { node: (X, Y) }
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de posições globais não encontrado: {path}")

    with path.open("rb") as f:
        positions = pickle.load(f)

    # Segurança leve: garantir que é um dict (evita confusão se carregar arquivo errado)
    if not isinstance(positions, dict):
        raise ValueError(f"Conteúdo inválido em {path}: esperado dict, obtido {type(positions)}")

    logger.info(f"Posições globais carregadas de: {path}")
    return positions

def choose_root_node(G_geo: nx.DiGraph, strategy: str = "center_valid") -> Node:
    """
    Escolhe um nó âncora (root) para iniciar o cálculo de posições globais.

    Estratégias:
    - "first":
        pega o primeiro nó iterável do grafo.
    - "first_valid":
        pega o primeiro nó com atributo node["valid"] == True (se existir).
        Se não houver, cai para "first".
    - "center":
        escolhe o nó mais "central" no grid (minimiza distância ao centro).
    - "center_valid" (padrão):
        como "center", mas restringe a nós com valid=True quando possível.
        Se não houver nós válidos, cai para "center".

    Por que "center_valid" é bom?
    - O erro acumulado na soma de translações tende a crescer com a distância.
      Começar no centro reduz drift nas bordas.

    Retorna:
        Node: tupla (x, y) do nó escolhido.
    """
    nodes = list(G_geo.nodes())
    if not nodes:
        raise ValueError("Grafo geométrico não possui nós.")

    def is_valid(n: Node) -> bool:
        return bool(G_geo.nodes[n].get("valid", True))

    if strategy == "first":
        return nodes[0]

    if strategy == "first_valid":
        for n in nodes:
            if is_valid(n):
                return n
        return nodes[0]

    # "center" e "center_valid" escolhem o nó mais próximo do centro do grid
    # Centro do grid = média das coordenadas dos nós (x_mean, y_mean)
    xs = [n[0] for n in nodes]
    ys = [n[1] for n in nodes]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)

    def dist2_to_center(n: Node) -> float:
        dx = n[0] - x_mean
        dy = n[1] - y_mean
        return dx * dx + dy * dy

    if strategy == "center":
        return min(nodes, key=dist2_to_center)

    if strategy == "center_valid":
        valid_nodes = [n for n in nodes if is_valid(n)]
        if valid_nodes:
            return min(valid_nodes, key=dist2_to_center)
        return min(nodes, key=dist2_to_center)

    raise ValueError(
        f"Estratégia inválida para root: '{strategy}'. Use: "
        "first, first_valid, center, center_valid."
    )

def compute_global_positions(
    G_geo: nx.DiGraph,
    root: Optional[Node] = None,
    root_strategy: str = "center_valid",
) -> Tuple[Positions, Node, list]:
    """
    Calcula posições globais (X,Y) para cada nó do grafo geométrico a partir das arestas
    dirigidas com pesos (dx,dy), usando uma busca em largura (BFS).

    Parâmetros:
        G_geo:
            Grafo geométrico dirigido. Cada aresta u->v deve ter atributos:
            - "dx" (float)
            - "dy" (float)
        root:
            Nó âncora (x,y). Se None, escolhe via choose_root_node(strategy=root_strategy).
        root_strategy:
            Estratégia usada para escolher root quando root=None.

    Retorna:
        positions:
            dict { node: (X, Y) } em pixels.
            O root sempre terá (0,0).
        root:
            nó usado como âncora.
        unreachable:
            lista de nós do grafo que não receberam posição (componentes desconectados).

    Observação:
        - Em grafos com ciclos, podem existir caminhos diferentes até o mesmo nó com
          pequenas inconsistências. Aqui usamos a primeira posição encontrada (BFS).
          Isso é suficiente para a próxima etapa (render), e depois podemos refinar
          com ajuste global se necessário.
    """
    if G_geo.number_of_nodes() == 0:
        raise ValueError("Grafo geométrico vazio (sem nós).")

    if root is None:
        root = choose_root_node(G_geo, strategy=root_strategy)

    # Posições globais em pixels
    positions: Positions = {root: (0.0, 0.0)}

    # BFS simples (fila)
    queue = deque([root])
    visited = set([root])

    while queue:
        u = queue.popleft()
        Xu, Yu = positions[u]

        # Para cada vizinho alcançável por aresta u -> v
        for v in G_geo.successors(u):
            # Garantir que existem dx/dy
            edge_data = G_geo.edges[u, v]
            dx = float(edge_data.get("dx", 0.0))
            dy = float(edge_data.get("dy", 0.0))

            if v not in positions:
                # Primeira vez que alcançamos v: define posição global
                positions[v] = (Xu + dx, Yu + dy)

            if v not in visited:
                visited.add(v)
                queue.append(v)

    # Nós que ficaram sem posição (componentes desconectados do root)
    unreachable = [n for n in G_geo.nodes() if n not in positions]

    logger.info(
        f"Posições globais calculadas: {len(positions)}/{G_geo.number_of_nodes()} nós alcançados a partir do root={root}."
    )
    if unreachable:
        logger.warning(f"Nós não alcançados (sem posição): {len(unreachable)}")

    return positions, root, unreachable

def normalize_positions(positions: Positions) -> Tuple[Positions, PosXY]:
    """
    Normaliza as posições globais para garantir que todas fiquem em coordenadas >= 0.

    Por quê?
      Ao propagar translações, é comum algumas posições ficarem negativas.
      Para renderizar um mosaico em um canvas (imagem), é mais fácil ter
      todas as coordenadas positivas (origem no canto superior esquerdo).

    Como funciona:
      - Encontra minX e minY
      - Se minX < 0, soma (-minX) em todas as posições X
      - Se minY < 0, soma (-minY) em todas as posições Y

    Retorna:
      positions_shifted:
        dict { node: (X_shifted, Y_shifted) }
      offset:
        (offX, offY) que foi aplicado em todas as posições
    """
    if not positions:
        raise ValueError("Positions está vazio, nada para normalizar.")

    xs = [xy[0] for xy in positions.values()]
    ys = [xy[1] for xy in positions.values()]

    min_x = min(xs)
    min_y = min(ys)

    off_x = -min_x if min_x < 0 else 0.0
    off_y = -min_y if min_y < 0 else 0.0

    shifted: Positions = {}
    for node, (X, Y) in positions.items():
        shifted[node] = (X + off_x, Y + off_y)

    return shifted, (off_x, off_y)


def generate_global_positions(
    force_recompute: bool = False,
    root: Optional[Node] = None,
    root_strategy: str = "center_valid",
    normalize: bool = True,
) -> Positions:
    """
    Orquestra a geração das posições globais:

    1) Carrega o grafo geométrico salvo em Config.GEOMETRIC_GRAPH_FILE
    2) Calcula posições globais com BFS
    3) Normaliza (opcional)
    4) Salva em Config.GLOBAL_POS_FILE (pickle)
    5) Retorna o dicionário final

    Parâmetros:
        force_recompute:
            Se False e o arquivo de posições já existir, carrega do disco.
            Se True, recalcula tudo e sobrescreve.
        root:
            Nó âncora opcional. Se None, escolhe via root_strategy.
        root_strategy:
            Estratégia para escolher root quando root=None.
        normalize:
            Se True, aplica normalize_positions() antes de salvar.

    Retorna:
        Positions: dict { node: (X, Y) } pronto para uso no draw_mosaic.py
    """
    pos_path = Config.GLOBAL_POS_FILE

    # Cache: se já existe e não quer recomputar, carrega e retorna
    if pos_path.exists() and not force_recompute:
        logger.info("Posições globais já existem em disco. Carregando...")
        return load_positions(pos_path)

    # Carregar grafo geométrico do disco (pickle)
    geo_path = Config.GEOMETRIC_GRAPH_FILE
    if not geo_path.exists():
        raise FileNotFoundError(
            f"Grafo geométrico não encontrado em: {geo_path}. "
            "Gere primeiro o graph_geometric.gpickle."
        )

    with geo_path.open("rb") as f:
        G_geo = pickle.load(f)

    if not isinstance(G_geo, nx.DiGraph):
        raise ValueError(f"Arquivo {geo_path} não contém um nx.DiGraph válido.")

    # Computar posições globais por BFS
    positions, used_root, unreachable = compute_global_positions(
        G_geo, root=root, root_strategy=root_strategy
    )

    logger.info(f"Root utilizado: {used_root}")

    if unreachable:
        logger.warning(
            f"Atenção: {len(unreachable)} nós não foram alcançados a partir do root e ficaram sem posição."
        )

    # Normalizar posições (opcional)
    if normalize:
        positions, offset = normalize_positions(positions)
        logger.info(f"Offset aplicado na normalização: {offset}")

    # Salvar e retornar
    save_positions(positions, pos_path)
    logger.info(f"Total de nós com posição salva: {len(positions)}")

    return positions


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] - %(message)s"
    )
    generate_global_positions(force_recompute=True)
    logger.info("Posições globais geradas com sucesso.")