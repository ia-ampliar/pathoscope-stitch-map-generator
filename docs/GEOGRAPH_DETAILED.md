# Documentação Detalhada: `geograph.py`

**Arquivo:** `src/modules/graph/geograph.py`

**Propósito:** Construir o **grafo geométrico** a partir do grafo topológico e dos resultados de matching, adicionando informações de deslocamento (dx, dy) e aplicando validação robusta de transformações.

---

## 1. Visão geral

O módulo `geograph.py` é responsável pela **etapa de construção do grafo geométrico** no pipeline de stitching. Ele:

1. Carrega o **grafo topológico** (vizinhança espacial entre tiles).
2. Lê os arquivos `.zarr` de matching (um por par de tiles).
3. Extrai **matrizes de transformação** (translação dx, dy) e métricas de confiança.
4. Valida transformações contra **gates direcionais** (MIN_MAIN, MAX_MAIN, MAX_ORTHO).
5. Constrói **grafo dirigido** com arestas ponderadas pelas transformações.
6. Aplica **recuperação de conectividade** para garantir grau mínimo em cada nó.
7. Visualiza o grafo com labels de transformações.

**Contexto no pipeline:**
- Entrada: `topology.gpickle` (grafo topológico) + `output/matches/*.zarr` (resultados de matching).
- Saída: `geometric.gpickle` (grafo geométrico) + `geometric_weights.png` (visualização).
- Próxima etapa: propagação de posições globais (`globalpos.py`).

---

## 2. Importações e dependências

```python
from typing import Optional, Tuple

import time
import numpy as np
import zarr
import networkx as nx
from pathlib import Path
import math

import logging
import matplotlib.pyplot as plt

from src.modules.graph.graph import load_valid_tiles, load_graph, save_graph
from src.config.config import Config
```

**Papel de cada import:**
- `typing`: anotações de tipo.
- `numpy`, `zarr`: manipulação de dados e leitura de resultados de matching.
- `networkx`: construção e manipulação de grafos.
- `math`, `matplotlib`: cálculos geométricos e visualização.
- Funções auxiliares: carregamento/salvamento de grafos (reutilizadas de `graph.py`).
- `Config`: injeção de configurações (thresholds, limits).

---

## 3. Função `_find_match_zarr_path(matches_dir: Path, tile_a: str, tile_b: str) -> Tuple[Optional[Path], Optional[str]]`

### 3.1 Assinatura e responsabilidades

```python
def _find_match_zarr_path(
    matches_dir: Path, 
    tile_a: str, 
    tile_b: str
) -> Tuple[Optional[Path], Optional[str]]:
```

**Parâmetros:**
- `matches_dir` (Path): diretório contendo arquivos `.zarr` de matches.
- `tile_a`, `tile_b` (str): nomes dos tiles em um par vizinho.

**Retorna:** tupla `(path, direction)`
- `path` (Path ou None): caminho do arquivo `.zarr` encontrado.
- `direction` (str ou None): indicador de qual arquivo foi encontrado.
  - `"A__B"`: arquivo `tile_a__tile_b.zarr` (match A → B).
  - `"B__A"`: arquivo `tile_b__tile_a.zarr` (match B → A).
  - `None`: nenhum arquivo encontrado.

**Responsabilidade:** localizar arquivo de match correspondente ao par, ignorando ordem.

### 3.2 Implementação

```python
path_a_b = matches_dir / f"{tile_a}__{tile_b}.zarr"
if path_a_b.exists():
    return path_a_b, "A__B"

path_b_a = matches_dir / f"{tile_b}__{tile_a}.zarr"
if path_b_a.exists():
    return path_b_a, "B__A"

return None, None
```

**Lógica:**
1. Tenta ordem direta (`A__B`).
2. Tenta ordem inversa (`B__A`).
3. Retorna `(None, None)` se nenhum existe.

**Importância:** o `direction` é usado para determinar se inverter sinais da translação (conforme política de geração de matches).

---

## 4. Função `ensure_min_degree_two(G_geo: nx.DiGraph, G_topo: nx.Graph, rejected_candidates: list, min_deg: int = 2) -> nx.DiGraph`

### 4.1 Assinatura e responsabilidades

```python
def ensure_min_degree_two(
    G_geo: nx.DiGraph, 
    G_topo: nx.Graph, 
    rejected_candidates: list, 
    min_deg: int = 2
) -> nx.DiGraph:
```

**Parâmetros:**
- `G_geo`: grafo geométrico (possivelmente com nós isolados ou baixo grau).
- `G_topo`: grafo topológico (referência de vizinhança esperada).
- `rejected_candidates`: lista de arestas rejeitadas por gates, candidatas para recuperação.
- `min_deg`: grau mínimo esperado (padrão: 2, pois 4-vizinhança em tópografia).

**Responsabilidade:** recuperar conectividade reintroduzindo arestas rejeitadas, priorizando por peso.

### 4.2 Implementação detalhada

#### Passo 1: Converter para não-dirigido e organizar candidatos

```python
G_und = G_geo.to_undirected()

rejected_candidates = sorted(rejected_candidates, key=lambda x: x[0], reverse=True)
```

- Versão não-dirigida para contar graus corretamente.
- Ordena candidatos por peso descendente (prioriza matches mais confiáveis).

#### Passo 2: Loop de recuperação

```python
changed = True
while changed:
    changed = False
    G_und = G_geo.to_undirected()

    low_deg_nodes = [n for n in G_und.nodes() if G_und.degree(n) < min_deg]

    if not low_deg_nodes:
        break
```

**Lógica:**
- Identifica nós com grau < mínimo.
- Se nenhum, estabilizou → termina.
- Senão, tenta recuperar arestas que ajudem.

#### Passo 3: Tentar reintroduzir arestas

```python
for w, u, v, dx, dy, match_file, direction in rejected_candidates:
    # só tenta se ajuda alguém com grau baixo
    if (u not in low_set) and (v not in low_set):
        continue

    # se já existe conexão u-v (em qualquer direção), pula
    if G_und.has_edge(u, v):
        continue

    # adiciona as duas direções para manter consistência
    G_geo.add_edge(u, v, dx=dx, dy=dy, weight=w, match_file=match_file, match_direction=direction)
    G_geo.add_edge(v, u, dx=-dx, dy=-dy, weight=w, match_file=match_file, match_direction=f"reverse_of_{direction}")

    changed = True
```

**Critério de reintrodução:**
- Apenas se pelo menos um dos nós tem grau baixo (evita readicionar desnecessariamente).
- Não readiciona se já existe aresta.
- Mantém bidireção com pesos opostos.

#### Passo 4: Validação e término

```python
if all(G_und.degree(n) >= min_deg for n in G_und.nodes()):
    return G_geo
```

- Se todos os nós estabilizaram, retorna.
- Senão, continua iterando.

**Benefício:** garante que o grafo não fique desconexo ou com componentes isoladas que prejudicam a propagação de posições.

---

## 5. Função `build_geometric_graph_translation(G_topo: nx.Graph, matches_dir: Path) -> nx.DiGraph`

### 5.1 Assinatura e responsabilidades

```python
def build_geometric_graph_translation(
    G_topo: nx.Graph,
    matches_dir: Path,
) -> nx.DiGraph:
```

**Parâmetros:**
- `G_topo`: grafo topológico (vizinhanças esperadas).
- `matches_dir`: diretório com arquivos `.zarr` de matching.

**Retorna:** grafo geométrico dirigido com pesos (dx, dy).

**Responsabilidade:** orquestrar construção do grafo geométrico com validação robusta.

### 5.2 Passo a passo

#### Passo 1: Validação e inicialização

```python
if not matches_dir.exists():
    raise FileNotFoundError(f"Diretório de matches não encontrado: {matches_dir}")

G_geo = nx.DiGraph()

for node, attrs in G_topo.nodes(data=True):
    G_geo.add_node(node, **attrs)

rejected_candidates = []
```

- Valida existência de matches.
- Cria grafo dirigido vazio.
- Copia nós e atributos do topológico.
- Inicializa lista de rejeitados (para recuperação).

#### Passo 2: Iterar sobre vizinhanças do topológico

```python
for u, v in G_topo.edges():
    tile_u = G_topo.nodes[u].get("label")
    tile_v = G_topo.nodes[v].get("label")

    if not tile_u or not tile_v:
        logger.warning(f"Aresta ({u}, {v}) ignorada: nó sem atributo 'label'...")
        continue
```

- Para cada vizinhança no topológico, extrai labels dos tiles.
- Rejeita se labels faltam (nó sem correspondência em matches).

#### Passo 3: Localizar arquivo de match

```python
zarr_path, direction = _find_match_zarr_path(matches_dir, tile_u, tile_v)

if zarr_path is None:
    logger.warning(f"Match .zarr não encontrado para: {tile_u} <-> {tile_v}. Aresta geométrica não criada.")
    continue
```

- **Regra importante:** se não há `.zarr` do par, NÃO cria aresta (evita matches espúrios).
- Registra aviso para debug.

#### Passo 4: Ler translação do zarr

```python
root = zarr.open(str(zarr_path), mode="r")
matches_group = root["matches"]

if "translation_matrix" not in matches_group.attrs:
    logger.warning(f"Arquivo {zarr_path.name} não contém attrs['translation_matrix']...")
    continue

M = np.array(matches_group.attrs["translation_matrix"], dtype=np.float64)

dx = float(M[0, 2])
dy = float(M[1, 2])

inlier_count = int(matches_group.attrs.get("inlier_count", 0))
raw_match_count = int(matches_group.attrs.get("raw_match_count", 0))
ransac_rmse = float(matches_group.attrs.get("ransac_rmse", float("inf")))
```

**Extração:**
- Abre arquivo `.zarr` e acessa grupo "matches".
- Extrai matriz homogênea 3×3 (apenas translação esperada).
- Extrai `dx` (M[0,2]) e `dy` (M[1,2]).
- Lê métricas: inliers, matches brutos, RMSE de reprojeção.

#### Passo 5: Calcular peso normalizado

```python
if raw_match_count <= 0:
    w = 0.0
else:
    w = (inlier_count / raw_match_count) * (1.0 / (1.0 + ransac_rmse))

if not np.isfinite(w):
    w = 0.0
else:
    w = float(np.clip(w, 0.0, 1.0))
```

**Fórmula de peso:**
$$w = \frac{\text{inliers}}{\text{total\_matches}} \times \frac{1}{1 + \text{RMSE}}$$

**Interpretação:**
- Prioriza pares com alta proporção de inliers.
- Penaliza alto RMSE de reprojeção.
- Normalizado em [0, 1].

**Proteção:** verifica NaN/inf e clipa em range válido.

#### Passo 6: Corrigir sinal da translação (conforme sentido do match)

```python
if direction == "A__B":
    dx, dy = -dx, -dy
```

**Crítico:** conforme `direction`:
- `"A__B"`: arquivo encontrado é `tile_a__tile_b`, então inverte sinal (para compatibilidade).
- `"B__A"`: arquivo é `tile_b__tile_a`, então mantém como está.

**Nota:** essa lógica depende de como o `match.py` gera as matrizes. Ajuste conforme necessário.

#### Passo 7: Validar magnitude do deslocamento

```python
shift = math.hypot(dx, dy)
if shift > Config.MAX_SHIFT:
    logger.warning(f"Descartando aresta (shift>MAX_SHIFT): {u}->{v} dx={dx:.2f} dy={dy:.2f} shift={shift:.2f}...")
    continue
```

- Magnitude: `shift = sqrt(dx² + dy²)`.
- Rejeita se > limite (possível erro de matching).

#### Passo 8: Aplicar gates direcionais

```python
ux, uy = u
vx, vy = v

is_horizontal = (uy == vy) and (abs(ux - vx) == 1)
is_vertical   = (ux == vx) and (abs(uy - vy) == 1)

if is_horizontal:
    if (abs(dx) < Config.MIN_MAIN) or (abs(dx) > Config.MAX_MAIN) or (abs(dy) > Config.MAX_ORTHO):
        logger.warning(f"Descartando aresta (gate horizontal): {u}->{v}...")
        rejected_candidates.append((w, u, v, dx, dy, str(zarr_path), direction))
        continue
elif is_vertical:
    if (abs(dy) < Config.MIN_MAIN) or (abs(dy) > Config.MAX_MAIN) or (abs(dx) > Config.MAX_ORTHO):
        logger.warning(f"Descartando aresta (gate vertical): {u}->{v}...")
        rejected_candidates.append((w, u, v, dx, dy, str(zarr_path), direction))
        continue
```

**Gates direcionais (validação por tipo de vizinhança):**

| Tipo | Condição | Requerimento |
|------|----------|--------------|
| Horizontal (vizinho à esquerda/direita) | `uy == vy` e `\|ux - vx\| = 1` | `MIN_MAIN ≤ \|dx\| ≤ MAX_MAIN` e `\|dy\| ≤ MAX_ORTHO` |
| Vertical (vizinho acima/abaixo) | `ux == vx` e `\|uy - vy\| = 1` | `MIN_MAIN ≤ \|dy\| ≤ MAX_MAIN` e `\|dx\| ≤ MAX_ORTHO` |

**Interpretação:**
- **MIN_MAIN:** deslocamento mínimo esperado (tiles têm sobreposição).
- **MAX_MAIN:** deslocamento máximo na direção principal.
- **MAX_ORTHO:** movimento máximo perpendicular (erros de alinhamento).

**Exemplo:** para vizinhos horizontais, esperamos `dx` grande (deslocamento principal) e `dy` pequeno (erro).

#### Passo 9: Guardar rejeitados e criar arestas

```python
# Criar as duas direções no grafo geométrico
G_geo.add_edge(
    u, v,
    dx=dx,
    dy=dy,
    match_file=str(zarr_path),
    match_direction=direction,
    weight=w,
)

G_geo.add_edge(
    v, u,
    dx=-dx,
    dy=-dy,
    match_file=str(zarr_path),
    match_direction=f"reverse_of_{direction}",
    weight=w,
)
```

**Bidireção:** cada aresta topológica `u-v` gera:
- `u → v` com `(dx, dy)`.
- `v → u` com `(-dx, -dy)`.

Ambas com mesmo peso (confiança).

#### Passo 10: Recuperar conectividade

```python
G_geo = ensure_min_degree_two(
    G_geo=G_geo,
    G_topo=G_topo,
    rejected_candidates=rejected_candidates,
    min_deg=2,
)
```

- Tenta readicionar arestas rejeitadas para garantir grau mínimo.

---

## 6. Função `plot_geometric_graph_with_weights(G_geo: nx.DiGraph, output_path: Path, show_arrows: bool = False, decimals: int = 1) -> None`

### 6.1 Assinatura e responsabilidades

```python
def plot_geometric_graph_with_weights(
    G_geo: nx.DiGraph,
    output_path: Path,
    show_arrows: bool = False,
    decimals: int = 1,
) -> None:
```

**Parâmetros:**
- `G_geo`: grafo geométrico dirigido.
- `output_path`: caminho para salvar PNG.
- `show_arrows`: se `True`, desenha setas (dirigido); se `False`, não-dirigido simplificado.
- `decimals`: casas decimais para rótulos.

**Responsabilidade:** visualizar grafo com labels de transformação (dx, dy).

### 6.2 Implementação

#### Passo 1: Definir posições dos nós

```python
pos = {node: (node[0], -node[1]) for node in G_geo.nodes()}
```

- Nó em coordenada `(x, y)` → posição `(x, -y)` (inverte Y para top-down).

#### Passo 2: Visualização não-dirigida (padrão)

```python
if not show_arrows:
    G_und = G_geo.to_undirected()

    edge_labels = {}
    for u, v in G_und.edges():
        if G_geo.has_edge(u, v):
            dx = G_geo.edges[u, v]["dx"]
            dy = G_geo.edges[u, v]["dy"]
        else:
            dx = -G_geo.edges[v, u]["dx"]
            dy = -G_geo.edges[v, u]["dy"]

        edge_labels[(u, v)] = f"({dx:.{decimals}f},{dy:.{decimals}f})"

    nx.draw(G_und, pos, with_labels=True, node_size=900, font_size=8)
    nx.draw_networkx_edge_labels(G_und, pos, edge_labels=edge_labels, font_size=7)
```

- Converte para não-dirigido para evitar duplicação visual.
- Escolhe um sentido canônico (u→v) para cada par.
- Desenha nós e labels de arestas.

#### Passo 3: Visualização dirigida (opcional)

```python
else:
    edge_labels = {}
    for u, v in G_geo.edges():
        dx = G_geo.edges[u, v]["dx"]
        dy = G_geo.edges[u, v]["dy"]

        if (dx > 0) or (dx == 0 and dy > 0):
            edge_labels[(u, v)] = f"({dx:.{decimals}f},{dy:.{decimals}f})"

    nx.draw(G_geo, pos, with_labels=True, node_size=900, font_size=8, arrows=True, arrowsize=15)
    nx.draw_networkx_edge_labels(G_geo, pos, edge_labels=edge_labels, font_size=7)
```

- Desenha DiGraph com setas.
- Filtro anti-duplicação: só rotula se `dx > 0` ou `(dx == 0 e dy > 0)` (semi-arbitrário).

---

## 7. Função `generate_geometric_graph()`

### 7.1 Entry point principal

```python
def generate_geometric_graph():
    G_topo = load_graph(Config.TOPOLOGY_GRAPH_FILE)

    G_geo = build_geometric_graph_translation(G_topo, matches_dir=Config.MATCHING_ZARR_PATH)

    logger.info(f"Grafo geométrico criado com {G_geo.number_of_nodes()} nós e {G_geo.number_of_edges()} arestas.")

    geo_graph_path = Config.GEOMETRIC_GRAPH_FILE
    save_graph(G_geo, geo_graph_path)

    plot_geometric_graph_with_weights(
        G_geo,
        output_path=Config.GEOMETRIC_GRAPH_WEIGHTS_FILE,
        show_arrows=False,
    )
```

**Fluxo:**
1. Carrega grafo topológico.
2. Constrói grafo geométrico.
3. Loga informações.
4. Salva em pickle.
5. Visualiza.

---

## 8. Configurações esperadas (`Config`)

| Parâmetro | Tipo | Descrição | Padrão sugerido |
|-----------|------|-----------|-----------------|
| `TOPOLOGY_GRAPH_FILE` | Path | Grafo topológico (entrada). | `output/result/topology.gpickle` |
| `MATCHING_ZARR_PATH` | Path | Diretório de matches (entrada). | `output/matches/` |
| `GEOMETRIC_GRAPH_FILE` | Path | Grafo geométrico (saída). | `output/result/geometric.gpickle` |
| `GEOMETRIC_GRAPH_WEIGHTS_FILE` | Path | Visualização (saída). | `output/result/geometric_weights.png` |
| `MAX_SHIFT` | float | Deslocamento máximo permitido. | `1000.0` |
| `MIN_MAIN` | float | Deslocamento mínimo na direção principal. | `50.0` |
| `MAX_MAIN` | float | Deslocamento máximo na direção principal. | `800.0` |
| `MAX_ORTHO` | float | Deslocamento máximo na direção ortogonal. | `100.0` |

---

## 9. Fluxo de dados resumido

```
topology.gpickle (nós com labels, vizinhanças)
         ↓
build_geometric_graph_translation()
         ├─ para cada vizinhança (u, v)
         │  ├─ find_match_zarr_path() → path + direction
         │  ├─ ler translation_matrix de .zarr
         │  ├─ calcular peso: w = (inliers/total) / (1 + RMSE)
         │  ├─ corrigir sinal conforme direction
         │  ├─ validar magnitude (shift <= MAX_SHIFT)
         │  ├─ aplicar gate direcional (MIN_MAIN, MAX_MAIN, MAX_ORTHO)
         │  └─ add_edge(u->v, dx, dy, w) + add_edge(v->u, -dx, -dy, w)
         │
         ├─ ensure_min_degree_two() → recuperar nós com grau baixo
         │
         ↓
geometric.gpickle (nós, arestas dirigidas com (dx, dy, w))
+
geometric_weights.png (visualização com labels de transformação)
```

---

## 10. Estrutura de dados: o grafo geométrico

### 10.1 Nós

```python
G_geo.nodes[node] = {
    'valid': bool,        # tile é válido (herdado do topológico)
    'label': str          # nome do tile (herdado do topológico)
}
```

### 10.2 Arestas dirigidas

```python
G_geo.edges[u, v] = {
    'dx': float,                      # deslocamento em X
    'dy': float,                      # deslocamento em Y
    'weight': float,                  # confiança [0, 1]
    'match_file': str,                # caminho do .zarr
    'match_direction': str            # "A__B" ou "B__A" ou "reverse_of_..."
}
```

---

## 11. Casos de tratamento e robustez

### Arquivo de match não encontrado
- **Comportamento:** aresta não é criada (regra solicitada).
- **Impacto:** pode deixar o grafo desconexo.
- **Mitigation:** `ensure_min_degree_two` tenta recuperar.

### Translação fora de gates
- **Comportamento:** aresta é rejeitada e guardada como candidato.
- **Impacto:** pode criar componentes isoladas.
- **Mitigation:** `ensure_min_degree_two` reintroduz se necessário.

### RMSE muito alto
- **Comportamento:** peso fica baixo (penalizado na fórmula).
- **Impacto:** aresta fica em baixa prioridade de recuperação.

### Nó sem label
- **Comportamento:** aresta é ignorada.
- **Impacto:** falha silenciosa (deve ser rara).

---

## 12. Otimizações e pontos de atenção

### ✅ Boas práticas presentes

1. **Gates direcionais:** valida transformações conforme tipo de vizinhança.
2. **Recuperação de conectividade:** garante que grafo não fique fragmentado.
3. **Peso normalizado:** facilita análise e filtragem futura.
4. **Logging informativo:** debug fácil.
5. **Visualização incluída:** feedback visual da construção.

### ⚠️ Possíveis melhorias

1. **Validação de ciclos:** detectar ciclos inconsistentes (dx em loop ≠ 0).
2. **Agregação de múltiplas estimativas:** se há várias arestas entre u-v (via caminhos diferentes), agregá-las.
3. **Estatísticas por gate:** contar quantas arestas foram rejeitadas por cada gate.
4. **Threshold adaptativo:** MIN_MAIN/MAX_MAIN poderiam variar por tipo de tile.
5. **Análise de componentes conexas:** alertar se houver multiplas componentes.

---

## 13. Exemplo de uso

### Setup mínimo

```python
from src.config.config import Config

Config.TOPOLOGY_GRAPH_FILE = Path("output/result/topology.gpickle")
Config.MATCHING_ZARR_PATH = Path("output/matches/")
Config.GEOMETRIC_GRAPH_FILE = Path("output/result/geometric.gpickle")
Config.GEOMETRIC_GRAPH_WEIGHTS_FILE = Path("output/result/geometric_weights.png")
Config.MAX_SHIFT = 1000.0
Config.MIN_MAIN = 50.0
Config.MAX_MAIN = 800.0
Config.MAX_ORTHO = 100.0
```

### Execução

```bash
python -m src.modules.graph.geograph
```

### Consulta de resultados

```python
from src.modules.graph.graph import load_graph

G_geo = load_graph(Path("output/result/geometric.gpickle"))

print(f"Nós: {G_geo.number_of_nodes()}")
print(f"Arestas: {G_geo.number_of_edges()}")

# Inspecionar aresta
u, v = list(G_geo.edges())[0]
edge_data = G_geo.edges[u, v]
print(f"{u} -> {v}: dx={edge_data['dx']:.2f}, dy={edge_data['dy']:.2f}, w={edge_data['weight']:.4f}")
```

---

## 14. Conexão com pipeline

### Inputs (dependências anteriores)
- `src.modules.graph.graph` → gera `topology.gpickle` (grafo topológico).
- `src.modules.features.match.match` → gera `output/matches/*.zarr` (matches com transformações).

### Outputs (fornecidos para próximo passo)
- `geometric.gpickle` → consumido por `src.modules.graph.globalpos` (calcula posições globais).
- `geometric_weights.png` → validação visual pelo usuário.

### Próximo passo
- **Posições globais:** propagar posições usando o grafo geométrico.

---

## 15. Gates e validação

### Por que gates direcionais?

Após matching, temos estimativas de `(dx, dy)` que podem conter erros. Gates garantem que:

1. **Deslocamentos fazem sentido:** não esperamos pulo de 2000 pixels em vizinhos adjacentes.
2. **Alinhamento está bom:** erro perpendicular (MAX_ORTHO) é pequeno.
3. **Topologia é consistente:** vizinhos horizontais têm deslocamento principalmente em X.

### Exemplo de cenário

```
Vizinhos horizontais: (1,1) e (2,1)
Esperado: dx ≈ -512 (tiles se sobrepõem), dy ≈ 0
Gate rejeita se: |dx| < 50 OU |dx| > 800 OU |dy| > 100

Match.py retorna: dx = -514.2, dy = 3.1
Resultado: ACEITO (passa em todos os critérios)
```

---

## 16. Recuperação de conectividade

### Por que é necessária?

Se muitas arestas forem rejeitadas por gates, o grafo pode ficar fragmentado. Exemplo:

```
Grafo topológico: (1,1)-(1,2)-(1,3)-(1,4)

Após gates, rejeita (1,2)-(1,3) e (1,3)-(1,4) → componentes:
  - {(1,1), (1,2)}
  - {(1,3), (1,4)}

Consequência: propagação de posições falha (não alcança todos os nós).
```

**Solução:** `ensure_min_degree_two` reintroduz arestas rejeitadas com peso > 0, garantindo conectividade.

---

## 17. Resumo

O `geograph.py` é o **construtor robusto do grafo geométrico** do pipeline:

- **Lê matches** de múltiplos arquivos `.zarr`.
- **Calcula pesos** baseado em confiança (inliers, RMSE).
- **Valida transformações** com gates direcionais (MIN_MAIN, MAX_MAIN, MAX_ORTHO).
- **Constrói grafo dirigido** com pesos bidirecionais.
- **Recupera conectividade** para garantir propagação posterior.
- **Visualiza** com labels de transformação.

O grafo geométrico é essencial: define como tiles se deslocam um em relação ao outro. Sua qualidade determina se o mosaico final fica bem alinhado ou não.

Os gates direcionais são o "guardião" da qualidade: filtram matches ruins que quebraria todo o pipeline posterior.

