**globalpos.py — Documentação Detalhada

**Visão Geral**
- **Arquivo:** src/modules/graph/globalpos.py
- **Propósito:** Estimar posições globais (X,Y) dos tiles a partir do grafo geométrico (`G_geo`). Fornece duas estratégias: propagação simples por BFS (`compute_global_positions`) e um solver robusto de mínimos quadrados com reponderação iterativa (IRLS + Huber) (`compute_global_positions_robust_ls`).
- **Saídas principais:** arquivo de posições em disco (`Config.GLOBAL_POS_FILE`) salvo via `save_positions` e retorno de dicionário `Positions` para uso em renderização do mosaico.

**Contexto e motivação**
- O grafo geométrico contém arestas direcionadas u->v com atributos `dx`, `dy` e `weight` que representam o deslocamento estimado (em pixels) e a confiança do match.
- Objetivo: combinar todas as estimativas locais (arestas) em um conjunto consistente de coordenadas globais, reduzindo drift e efeitos de outliers.
- Aplicações: posicionamento dos tiles em um canvas para montagem do mosaico.

**Tipos importantes**
- `Node = Tuple[int, int]` — nó representado por coordenadas topológicas (x, y).
- `PosXY = Tuple[float, float]` — posição global em pixels (X, Y).
- `Positions = Dict[Node, PosXY]` — mapa resultante de posições por nó.

**Funções públicas e seu comportamento**

- `save_positions(positions: Positions, path: Path) -> None`
  - Serializa `positions` em disco usando `pickle` (protocol HIGHEST_PROTOCOL).
  - Cria diretórios pais quando necessário.

- `load_positions(path: Path) -> Positions`
  - Carrega dicionário salvo com `pickle` e valida se o conteúdo é `dict`.

- `choose_root_node(G_geo: nx.DiGraph, strategy: str = "center_valid") -> Node`
  - Seleciona um nó âncora (root) para âncora espacial.
  - Estratégias: `first`, `first_valid`, `center`, `center_valid`.
  - `center_valid` (padrão): prefere nós com `node["valid"] == True` e escolhe o mais próximo do centro do conjunto de nós.
  - Racional: escolher um root central minimiza erro acumulado nas bordas.

- `compute_global_positions(G_geo, root=None, root_strategy="center_valid") -> (positions, root, unreachable)`
  - Propaga posições por BFS a partir do `root`: para cada aresta u->v, define position[v] = position[u] + (dx, dy) na primeira visita.
  - Simples, rápida, porém sensível a inconsistências quando há ciclos e outliers nas arestas.
  - Retorna `unreachable` com nós desconectados do componente do root.

- `compute_global_positions_robust_ls(G_geo, root=None, root_strategy="center_valid", max_iters=8, huber_k=2.5, min_weight=1e-6, residual_gate_px=200.0, gate_after_iter=1) -> (positions, root, unreachable)`
  - Solver robusto baseado em formulação linear:
    - Para cada restrição (u->v): x_v - x_u ≈ dx ; y_v - y_u ≈ dy
    - Monta sistema A x = b (separado para X e Y), adicionando uma linha de âncora x_root = 0.
    - Resolve por LS com reponderação iterativa (IRLS) aplicando loss de Huber para reduzir impacto de outliers.
    - Opcionalmente aplica um passo de "gating" para remover arestas com resíduo excessivo (> residual_gate_px) após a 1ª iteração, garantindo que não deixe nós com grau < 2.
  - Retorna posições robustas, root efetivo e lista de `unreachable`.

**Detalhes numéricos e implementação**

- `_build_system(...)`:
  - Monta a matriz esparsa A (coo_matrix -> csr) com shape (m, n_nodes), onde m = n_constraints + 1 (linha extra de âncora).
  - Cada aresta i produz duas colunas na linha i: +1 em coluna v, -1 em coluna u.
  - bx e by contêm os vetores de deslocamento (dx, dy) para cada restrição; última linha força x_root = 0, y_root = 0.
  - Retorna também `w_base` (pesos base com 1.0 adicional para a linha de âncora) e `n_constraints`.

- Estrutura dos vetores usados no solver:
  - `u_arr`, `v_arr`: índices locais dos nós (0..n_nodes-1) correspondentes ao subconjunto do componente do root.
  - `dx_arr`, `dy_arr`: deslocamentos em float.
  - `w_arr`: pesos base das arestas (ex.: qualidade do match).
  - `meta_arr`: metadados por aresta (u_node, v_node, match_file) usados para logs.

- Loop IRLS principal (compute_global_positions_robust_ls):
  - Inicializa `w_total = w_base`.
  - Para cada iteração:
    - Computa Wsqrt = sqrt(w_total) e aplica multiplicação linha/coluna em A para resolver Ax = b ponderado.
    - Resolve separadamente para X e Y usando `scipy.sparse.linalg.lsqr` (retorna vetor solução `solx`, `soly`).
    - Calcula resíduos r = sqrt(rx^2 + ry^2) onde rx = A @ solx - bx.
    - Se `gate_after_iter` e `residual_gate_px` habilitados, faz GATING UMA VEZ:
      - Marca arestas com resíduo > gate como removidas (`keep=False`).
      - Para evitar desconectar nós (grau < 2), tenta restaurar arestas removidas por ordem de menor resíduo.
      - Reconstroi o sistema com arestas mantidas e reinicia iterações no novo sistema.
    - Aplica pesos Huber: hub = 1 se r <= huber_k, else hub = huber_k / r.
    - Atualiza `w_total = w_base * hub` para próxima iteração.

- Observações sobre gating e conservação de grau
  - Gating é agressivo: remove arestas com resíduos muito altos, mas o código previne graus fracos restaurando algumas arestas removidas para manter conectividade mínima (min_deg=2).
  - Após o gating, reconstrói A, bx, by e reinicia o processo.

**Diagnósticos e logs**
- `log_solver_diagnostics(...)` produz informações úteis:
  - número de nós, nós solucionados, nós inalcançáveis, arestas totais e usadas
  - estatísticas de resíduos (mean, median, p90, p95, max)
  - comparação entre média de pesos base e finais (após Huber)
  - lista das top restrições removidas no gating com detalhes (resíduo, dx, dy, peso, arquivo de match)
  - checagem de NaN/inf nas posições

**Fluxo orquestrado**
- `generate_global_positions(force_recompute=False, root=None, root_strategy='center_valid', normalize=True)`:
  - Se existir `Config.GLOBAL_POS_FILE` e `force_recompute=False`, carrega e retorna.
  - Caso contrário, carrega `Config.GEOMETRIC_GRAPH_FILE` (pickle contendo `nx.DiGraph`).
  - Executa `compute_global_positions_robust_ls` para resolver posições do componente do root.
  - Opcionalmente normaliza posições com `normalize_positions` (shift para coordenadas >= 0).
  - Salva com `save_positions` e retorna o dicionário final.

**normalize_positions(positions)**
- Ajusta todas as posições para que minX, minY >= 0 retornando também o offset aplicado.
- Útil para construir um canvas de desenho com origem no canto superior esquerdo.

**Comportamento em casos limites**
- Se `G_geo` estiver vazio -> ValueError.
- Se após filtragem não existir restrição válida -> ValueError.
- Nós fora do componente do root ficam em `unreachable` e não recebem posição.

**Complexidade e desempenho**
- Montagem do sistema e solução LS é O(n_edges * log n) dependente do solver esparso; `lsqr` é iterativo e eficiente para matrizes esparsas.
- Paralelismo não é usado dentro do solver; custo principal é a construção de A (vetores) e chamadas a `lsqr` por iteração (max_iters vezes).
- Para grafos grandes, é recomendado reduzir o subconjunto a componentes relevantes ou aumentar recursos.

**Sugestões de melhoria / cuidados**
- Expor `n_jobs` ou usar solvers mais eficientes se o grafo tiver dezenas de milhares de arestas.
- Considerar normalização/escala de pesos antes do IRLS para maior estabilidade numérica.
- Log mais granular (DEBUG) com amostragem de arestas ao invés de imprimir todas as topK em logs.
- Tornar `min_deg` um parâmetro configurável para diferentes densidades de captura.
- Oferecer opção de resolver apenas sub-regiões (clustering) e depois registrar as transformações entre clusters.

**Como usar**
- Gerar posições e salvar em disco:

  from src.modules.graph.globalpos import generate_global_positions
  positions = generate_global_positions(force_recompute=True)

- Carregar posições já geradas:

  from src.modules.graph.globalpos import load_positions
  positions = load_positions(Config.GLOBAL_POS_FILE)

- Executar em linha de comando (módulo):

  python -m src.modules.graph.globalpos

**Arquivos relevantes e integração**
- `Config.GEOMETRIC_GRAPH_FILE` — arquivo que contém o grafo geométrico (pickle). Gerado por `geograph.py`.
- `Config.GLOBAL_POS_FILE` — caminho onde as posições globais são salvas.
- O resultado `Positions` é utilizado por módulos de renderização (ex.: draw_mosaic, populate_geom, canvas).

**Resumo**
- `globalpos.py` contém tanto uma solução rápida (BFS) quanto uma solução robusta (IRLS + Huber) para estimar posições globais de tiles a partir de restrições de deslocamento.
- O solver robusto é apropriado quando há ciclos, outliers e inconsistências nas arestas; o gating remove restrições com resíduo muito alto mantendo conectividade mínima.
- O módulo foi escrito com atenção a logs diagnósticos e segurança no salvamento/carregamento de resultados.

---

Salvei este documento em [docs/GLOBALPOS_DETAILED.md](docs/GLOBALPOS_DETAILED.md). Deseja que eu abra o arquivo para revisão ou rode `generate_global_positions(force_recompute=True)` agora?