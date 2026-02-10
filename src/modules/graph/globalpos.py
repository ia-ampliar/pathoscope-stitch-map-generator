# src/modules/graph/globalpos.py

from __future__ import annotations

import time
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

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

def log_solver_diagnostics(
    G_geo: nx.DiGraph,
    positions: dict,
    root,
    unreachable: list,
    residuals: np.ndarray,
    weights_base: np.ndarray,
    weights_final: np.ndarray,
    logger,
):
    """
    Gera logs diagnósticos do solver global de posições.

    Esta função NÃO altera resultados. Serve apenas para inspeção,
    validação e debug do comportamento do solver robusto.
    """

    n_nodes_total = G_geo.number_of_nodes()
    n_nodes_solved = len(positions)
    n_nodes_unreachable = len(unreachable)

    n_edges_total = G_geo.number_of_edges()
    n_edges_used = len(weights_base) - 1

    logger.info("=== Solver Global: Diagnóstico ===")
    logger.info(f"Nós totais            : {n_nodes_total}")
    logger.info(f"Nós resolvidos        : {n_nodes_solved}")
    logger.info(f"Nós inalcançáveis     : {n_nodes_unreachable}")
    logger.info(f"Arestas totais        : {n_edges_total}")
    logger.info(f"Arestas usadas        : {n_edges_used}")

    if residuals.size == 0:
        logger.warning("Nenhum resíduo disponível para análise.")
        return

    # Estatísticas de resíduos
    mean_r = float(np.mean(residuals))
    median_r = float(np.median(residuals))
    p90_r = float(np.percentile(residuals, 90))
    p95_r = float(np.percentile(residuals, 95))
    max_r = float(np.max(residuals))

    logger.info(
        "Resíduos (pixels) | "
        f"mean={mean_r:.3f}, "
        f"median={median_r:.3f}, "
        f"p90={p90_r:.3f}, "
        f"p95={p95_r:.3f}, "
        f"max={max_r:.3f}"
    )

    # Análise de pesos
    mean_w_base = float(np.mean(weights_base))
    mean_w_final = float(np.mean(weights_final))

    attenuated = np.sum(weights_final < 0.5 * weights_base)
    heavily_attenuated = np.sum(weights_final < 0.1 * weights_base)

    logger.info(
        "Pesos | "
        f"mean_base={mean_w_base:.4f}, "
        f"mean_final={mean_w_final:.4f}"
    )
    logger.info(
        f"Arestas atenuadas (Huber)     : {attenuated}/{n_edges_used}"
    )
    logger.info(
        f"Arestas fortemente atenuadas : {heavily_attenuated}/{n_edges_used}"
    )

    # Sanity checks
    rx, ry = positions.get(root, (None, None))
    if rx is not None:
        logger.info(f"Root position (esperado ~0,0): x={rx:.6f}, y={ry:.6f}")

    # Checagem numérica
    all_vals = np.array(list(positions.values()), dtype=np.float64)
    if not np.all(np.isfinite(all_vals)):
        logger.warning("Posições globais contêm NaN ou inf.")


def _build_system(
    u_arr: np.ndarray,
    v_arr: np.ndarray,
    dx_arr: np.ndarray,
    dy_arr: np.ndarray,
    w_arr: np.ndarray,
    root_i: int,
    n_nodes: int,
):
    n_constraints = int(len(u_arr))               # só arestas
    m = n_constraints + 1                         # +1 âncora
    anchor_row = m - 1

    rows = []
    cols = []
    vals = []
    bx = np.zeros(m, dtype=np.float64)
    by = np.zeros(m, dtype=np.float64)

    # Linhas das restrições (x_v - x_u = dx ; y_v - y_u = dy)
    for i in range(n_constraints):
        ui = int(u_arr[i]); vi = int(v_arr[i])
        rows += [i, i]
        cols += [vi, ui]
        vals += [1.0, -1.0]
        bx[i] = float(dx_arr[i])
        by[i] = float(dy_arr[i])

    # Âncora: x_root = 0 e y_root = 0
    rows.append(anchor_row); cols.append(int(root_i)); vals.append(1.0)
    bx[anchor_row] = 0.0
    by[anchor_row] = 0.0

    A = coo_matrix((vals, (rows, cols)), shape=(m, n_nodes)).tocsr()
    w_base = np.concatenate([w_arr.astype(np.float64), np.array([1.0], dtype=np.float64)])

    return A, bx, by, w_base, n_constraints


def _build_undirected_constraint_graph(n_nodes: int, u_idx: np.ndarray, v_idx: np.ndarray, keep: np.ndarray) -> nx.Graph:
    """
    Grafo não-dirigido de conectividade induzido pelas restrições mantidas (keep=True).
    Nós são índices [0..n_nodes-1] (índices locais do solver).
    """
    H = nx.Graph()
    H.add_nodes_from(range(n_nodes))
    kept = np.where(keep)[0]
    edges = [(int(u_idx[j]), int(v_idx[j])) for j in kept]
    H.add_edges_from(edges)
    return H


def compute_global_positions_robust_ls(
    G_geo: nx.DiGraph,
    root: Optional[Node] = None,  
    root_strategy: str = "center_valid",
    max_iters: int = 8,
    huber_k: float = 2.5,
    min_weight: float = 1e-6,
    residual_gate_px: float = 200.0,
    gate_after_iter: int = 1,
) -> Tuple[Positions, Node, list]:
    
    """
    Estima as posições globais dos tiles a partir de um grafo geométrico dirigido
    utilizando um solver de mínimos quadrados robusto (IRLS + loss de Huber).

    Esta função formula o problema como um sistema global de restrições do tipo:

        x_v - x_u ≈ dx
        y_v - y_u ≈ dy

    para cada aresta geométrica (u -> v) do grafo, resolvendo simultaneamente todas as
    posições de forma consistente, explorando ciclos do grafo e reduzindo drift.

    A robustez é obtida por:
    - ponderação das arestas (weight) baseada na qualidade do match
    - reponderação iterativa (IRLS) com loss de Huber, reduzindo o impacto de outliers

    O sistema é resolvido separadamente para os eixos X e Y, compartilhando a mesma
    estrutura de pesos.

    Apenas o componente conexo que contém o nó raiz é resolvido. Nós fora desse
    componente são retornados como inalcançáveis.

    Parâmetros
    ----------
    G_geo : nx.DiGraph
        Grafo geométrico dirigido.
        Cada aresta deve conter os atributos:
            - dx (float): deslocamento em X de u para v
            - dy (float): deslocamento em Y de u para v
            - weight (float): peso base da restrição (qualidade do match)

    root : Optional[Node], default=None
        Nó utilizado como âncora do sistema de coordenadas globais.
        Se None, o nó é escolhido automaticamente conforme root_strategy.

    root_strategy : str, default="center_valid"
        Estratégia para escolha automática do nó raiz, reutilizando a lógica existente
        em choose_root_node (ex.: centro do grid, nó válido, etc.).

    max_iters : int, default=8
        Número máximo de iterações do algoritmo IRLS (Iteratively Reweighted Least Squares).
        Valores típicos entre 5 e 10 são suficientes na prática.

    huber_k : float, default=2.5
        Parâmetro da loss de Huber.
        Define o limiar entre comportamento quadrático (inliers) e linear (outliers).
        Valores maiores tornam o solver menos agressivo com outliers.

    min_weight : float, default=1e-6
        Peso mínimo para que uma aresta seja considerada no sistema.
        Arestas com weight <= min_weight são ignoradas.

    Retorna
    -------
    positions : Dict[Node, Tuple[float, float]]
        Dicionário mapeando cada nó resolvido para sua posição global (x, y).

    used_root : Node
        Nó efetivamente utilizado como âncora do sistema.

    unreachable : list
        Lista de nós do grafo geométrico que não pertencem ao componente conexo
        do nó raiz e, portanto, não tiveram posição global estimada.

    Observações
    -----------
    - A posição do nó raiz é fixada em (0, 0) para eliminar a liberdade de translação
      global do sistema (gauge freedom).
    - A função não modifica o grafo de entrada.
    - Este método é significativamente mais robusto que BFS em regiões com:
        * ciclos
        * matches ruidosos
        * tiles com pouco conteúdo visual (ex.: áreas muito brancas)
    """

    if G_geo.number_of_nodes() == 0:
        raise ValueError("Grafo geométrico vazio (sem nós).")

    if root is None:
        root = choose_root_node(G_geo, strategy=root_strategy)

    # Filtrar componente do root
    G_und = G_geo.to_undirected()
    component = nx.node_connected_component(G_und, root)
    nodes = list(component)
    idx = {n: i for i, n in enumerate(nodes)}

    unreachable = [n for n in G_geo.nodes() if n not in component]

    # Extrair as restrições (uma vez)
    u_list, v_list, dx_list, dy_list, w_list = [], [], [], [], []
    meta_list = []  # lista paralela: 1 item por restrição/aresta

    # Preencher listas com arestas válidas
    for u, v, data in G_geo.edges(data=True):
        if u not in idx or v not in idx:
            continue
        w = float(data.get("weight", 0.0))
        if w <= min_weight:
            continue
        u_list.append(idx[u])
        v_list.append(idx[v])
        dx_list.append(float(data.get("dx", 0.0)))
        dy_list.append(float(data.get("dy", 0.0)))
        w_list.append(w)
        meta_list.append((
            u,  # nó original (ex.: (x,y))
            v,  # nó original
            data.get("match_file", ""),  # caminho do .zarr
        ))

    if len(u_list) == 0:
        raise ValueError("Nenhuma restrição válida encontrada (arestas com peso > min_weight).")
    

    u_arr = np.asarray(u_list, dtype=np.int32)
    v_arr = np.asarray(v_list, dtype=np.int32)
    dx_arr = np.asarray(dx_list, dtype=np.float64)
    dy_arr = np.asarray(dy_list, dtype=np.float64)
    w_arr  = np.asarray(w_list,  dtype=np.float64)
    meta_arr = np.asarray(meta_list, dtype=object)

    root_i = idx[root]

    # =========================================================
    # Helper interno: monta A, bx, by, w_base com âncora
    # =========================================================

    A, bx, by, w_base, n_constraints = _build_system(
        u_arr, v_arr, dx_arr, dy_arr, w_arr, root_i, len(nodes)
    )

    # Loop IRLS + Huber + gating
    w_total = w_base.copy()
    gated_once = False

    for it in range(max_iters):
        Wsqrt = np.sqrt(w_total)

        Ax = A.multiply(Wsqrt[:, None])
        bxw = bx * Wsqrt
        solx = lsqr(Ax, bxw)[0]

        Ay = A.multiply(Wsqrt[:, None])
        byw = by * Wsqrt
        soly = lsqr(Ay, byw)[0]

        rx = (A @ solx) - bx
        ry = (A @ soly) - by
        r = np.sqrt(rx * rx + ry * ry) + 1e-12

        # -------------------------
        # GATING (uma única vez)
        # -------------------------
        # Remove restrições com resíduo muito alto (outliers extremos)
        if (
            (not gated_once)
            and (it == gate_after_iter)
            and (residual_gate_px is not None)
            and (residual_gate_px > 0)
        ):
            r_edges = r[:n_constraints]  # NÃO inclui âncora
            keep = r_edges <= residual_gate_px
            num_drop = int((~keep).sum())

            # --- regra: não deixar nó com grau < 2 (no conjunto de restrições) ---
            min_deg = 2

            # graus considerando apenas arestas mantidas
            deg = np.zeros(len(nodes), dtype=np.int32)
            for ui, vi in zip(u_arr[keep], v_arr[keep]):
                deg[ui] += 1
                deg[vi] += 1

            # candidatos a restaurar (os que seriam removidos), ordenados por menor resíduo primeiro
            dropped = np.where(~keep)[0]
            order = np.argsort(r_edges[dropped])  # menor resíduo = "menos ruim"
            dropped_sorted = dropped[order]

            changed = True
            while changed:
                changed = False

                # nós que estão fracos
                weak_nodes = np.where(deg < min_deg)[0]
                if weak_nodes.size == 0:
                    break
                weak_set = set(weak_nodes.tolist())

                # tenta restaurar arestas que conectem nós fracos
                for j in dropped_sorted:
                    if keep[j]:
                        continue

                    ui = int(u_arr[j]); vi = int(v_arr[j])

                    # só restaura se ajuda algum nó fraco
                    if (ui not in weak_set) and (vi not in weak_set):
                        continue

                    # restaura esta restrição
                    keep[j] = True
                    deg[ui] += 1
                    deg[vi] += 1
                    changed = True

                    # atualiza weak_set dinamicamente (para cortar cedo)
                    if deg[ui] >= min_deg and ui in weak_set:
                        weak_set.remove(ui)
                    if deg[vi] >= min_deg and vi in weak_set:
                        weak_set.remove(vi)

                    if not weak_set:
                        break

            dropped_idx = np.where(~keep)[0]

            # Ordena removidas por resíduo (maior primeiro)
            order = np.argsort(r_edges[dropped_idx])[::-1]
            topk = dropped_idx[order[:10]]

            logger.warning("[GATING] top restrições removidas (pior -> melhor):")
            for j in topk:
                u_node, v_node, match_file = meta_arr[j]
                logger.warning(
                    f"  r={r_edges[j]:8.2f}px | "
                    f"{u_node} -> {v_node} | "
                    f"dx={dx_arr[j]:.2f}, dy={dy_arr[j]:.2f} | "
                    f"w={w_arr[j]:.4f} | "
                    f"file={match_file}"
                )

            if num_drop > 0:
                logger.warning(
                    f"[GATING] removendo {num_drop}/{n_constraints} restrições com resíduo > {residual_gate_px:.1f}px"
                )

                # filtra arrays (somente arestas)
                u_arr = u_arr[keep]
                v_arr = v_arr[keep]
                dx_arr = dx_arr[keep]
                dy_arr = dy_arr[keep]
                w_arr = w_arr[keep]
                meta_arr = meta_arr[keep]

                # reconstrói sistema com conjunto filtrado
                A, bx, by, w_base, n_constraints = _build_system(
                    u_arr, v_arr, dx_arr, dy_arr, w_arr, root_i, len(nodes)
                )
                w_total = w_base.copy()
                gated_once = True

                # resolve novamente na próxima iteração com sistema limpo
                continue

            gated_once = True  # não tinha o que cortar, mas não tenta de novo

        # Huber weights
        hub = np.ones_like(r)
        mask = r > huber_k
        hub[mask] = huber_k / r[mask]

        # atualiza pesos (mantendo base * robusto)
        w_total = w_base * hub

    # Construir positions de saída
    positions = {node: (float(solx[idx[node]]), float(soly[idx[node]])) for node in nodes}

    log_solver_diagnostics(
        G_geo=G_geo,
        positions=positions,
        root=root,
        unreachable=unreachable,
        residuals=r,
        weights_base=w_base,
        weights_final=w_total,
        logger=logger,
    )
    
    return positions, root, unreachable


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
    # positions, used_root, unreachable = compute_global_positions(
    #     G_geo, root=root, root_strategy=root_strategy
    # )

    # Solver least squares robusto (IRLS + Huber)
    positions, used_root, unreachable = compute_global_positions_robust_ls(
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
    

    start_time = time.perf_counter()
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] - %(message)s"
    )
    generate_global_positions(force_recompute=True)
    logger.info("Posições globais geradas com sucesso.")

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")