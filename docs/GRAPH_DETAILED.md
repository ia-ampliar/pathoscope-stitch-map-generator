# Documentação Detalhada: `graph.py`

**Arquivo:** `src/modules/graph/graph.py`

**Propósito:** Construir e persistir o **grafo topológico** que representa a vizinhança espacial entre tiles com base em suas coordenadas extraídas.

---

## 1. Visão geral

O módulo `graph.py` é responsável pela **etapa de construção do grafo topológico** no pipeline de stitching. Ele:

1. Carrega tiles válidos (resultado da classificação).
2. **Extrai coordenadas** de cada tile a partir do nome.
3. Constrói um **grafo de vizinhança** onde cada nó é um tile e arestas conectam tiles adjacentes.
4. **Persiste** o grafo em disco (formato pickle).
5. Gera uma **visualização (imagem PNG)** do grafo topológico.

**Contexto no pipeline:**
- Entrada: `valid_tiles.json` (resultado de `classifier`).
- Saída: `topology.gpickle` (grafo persistido) + `topology.png` (visualização).
- Próximas etapas: matching de features, construção do grafo geométrico.

---

## 2. Importações e dependências

```python
import json                    # Leitura de JSON
import logging                 # Sistema de logging
import re                      # Expressões regulares
from pathlib import Path       # Manipulação de caminhos
from typing import Dict        # Type hints

import matplotlib.pyplot as plt           # Visualização de grafos
import networkx as nx                    # Biblioteca de grafos
import pickle                            # Serialização
from matplotlib.patches import Patch     # Legendas em plots

from src.config.config import Config                      # Configurações centralizadas
from src.utils.coordinates import extract_coordinates    # Extrator de coordenadas
```

**Papel de cada import:**
- `json`, `logging`, `re`, `Path`: utilidades padrão.
- `matplotlib`, `networkx`: núcleo da construção e visualização de grafos.
- `pickle`: serialização eficiente de objetos Python complexos.
- `Config`, `extract_coordinates`: injeção de configuração e parsing de nomes.

---

## 3. Função `save_graph(G: nx.Graph, path: Path) -> None`

### 3.1 Assinatura e responsabilidades

```python
def save_graph(G: nx.Graph, path: Path) -> None:
```

**Parâmetros:**
- `G` (nx.Graph): grafo NetworkX a persistir.
- `path` (Path): caminho de saída (ex.: `Config.TOPOLOGY_GRAPH_FILE`).

**Responsabilidade:** serializar grafo em disco para recuperação futura.

### 3.2 Implementação

```python
path.parent.mkdir(parents=True, exist_ok=True)

with open(path, "wb") as f:
    pickle.dump(G, f)

logger.info(f"Grafo salvo em: {path}")
```

**Lógica:**

1. **Criar diretórios:** `mkdir(parents=True, exist_ok=True)` garante que estrutura de pastas existe.
2. **Serializar:** `pickle.dump()` salva grafo em formato binário.
   - Preserva estrutura completa: nós, arestas, atributos.
   - Mantém tipos Python (ex.: nós como tuplas `(x, y)`).
   - Rápido e eficiente para uso interno do pipeline.
3. **Log:** registra sucesso.

### 3.3 Por que pickle?

Alternativas:
- **GraphML, GML:** formatos textuais, legíveis, mas verbosos.
- **JSON:** requer conversão (NetworkX retorna tuplas; JSON não suporta).
- **Pickle:** binário, rápido, preserva tipos Python, ideal para pipeline.

**Trade-off:** pickle não é portável entre linguagens, mas para pipeline interno Python é adequado.

---

## 4. Função `load_graph(path: Path) -> nx.Graph`

### 4.1 Assinatura e responsabilidades

```python
def load_graph(path: Path) -> nx.Graph:
```

**Parâmetros:**
- `path` (Path): caminho do arquivo pickled.

**Retorna:** grafo desserializado (nx.Graph).

**Responsabilidade:** recuperar grafo persistido de disco.

### 4.2 Implementação

```python
if not path.exists():
    raise FileNotFoundError(f"Arquivo de grafo não encontrado: {path}")

with open(path, "rb") as f:
    G = pickle.load(f)

logger.info(f"Grafo carregado de: {path}")
return G
```

**Lógica:**

1. **Validação:** fail-fast se arquivo não existe (evita ambiguidade).
2. **Desserialização:** `pickle.load()` recupera grafo.
3. **Log:** registra sucesso.

**Tratamento de erros:** `FileNotFoundError` deixa claro o problema e facilita debugging.

---

## 5. Função `load_valid_tiles(path: Path) -> Dict[str, bool]`

### 5.1 Assinatura e responsabilidades

```python
def load_valid_tiles(path: Path) -> Dict[str, bool]:
```

**Parâmetros:**
- `path` (Path): caminho do arquivo JSON (`output/metadata/valid_tiles.json`).

**Retorna:** dicionário `{tile_name: is_valid}`.

**Responsabilidade:** carregar resultado da classificação de tiles.

### 5.2 Implementação

```python
with open(path) as f:
    return json.load(f)
```

**Simples e direto:** apenas desserializa JSON. Exemplo:

```python
{
    "00001_x1_y1_zp1": true,
    "00002_x2_y1_zp1": true,
    "00003_x3_y1_zp1": false,
    ...
}
```

---

## 6. Função `build_graph(valid_tiles: Dict[str, bool], pattern: re.Pattern) -> nx.Graph`

### 6.1 Assinatura e responsabilidades

```python
def build_graph(valid_tiles: Dict[str, bool], pattern: re.Pattern) -> nx.Graph:
```

**Parâmetros:**
- `valid_tiles`: dicionário com tiles e seus status.
- `pattern`: regex compilado para extrair coordenadas (ex.: `(\d+)_x(\d+)_y(\d+)_zp(\d+)`).

**Retorna:** grafo topológico (nx.Graph, não dirigido).

**Responsabilidade:** construir estrutura de nós e arestas baseado em vizinhança espacial.

### 6.2 Passo a passo

#### Passo 1: Criar grafo vazio

```python
G = nx.Graph()
```

- Grafo não-dirigido (arestas bidirecionais).
- Adequado para representar vizinhança (se A é vizinho de B, B é vizinho de A).

#### Passo 2: Adicionar nós

```python
for tile_name, is_valid in valid_tiles.items():
    coord = extract_coordinates(tile_name, pattern)
    if coord:
        G.add_node(coord, valid=is_valid, label=tile_name)
```

**Lógica por tile:**

1. **Extrair coordenadas:** `extract_coordinates()` usa regex para obter (x, y) de nome como "00001_x1_y1_zp1".
   - Resultado: tupla ou lista (ex.: `[1, 1]` ou `(1, 1)`).

2. **Validação:** se `coord` é truthy (não None/vazio), prossegue.

3. **Adicionar nó:** `add_node(coord, ...)` cria nó com atributos:
   - Chave do nó: tupla de coordenadas (ex.: `(1, 1)`).
   - Atributo `valid`: booleano indicando se tile é válido.
   - Atributo `label`: nome original do tile (usado em visualização).

**Exemplo de nó criado:**
```python
G.nodes[(1, 1)]
# Resultado:
# {'valid': True, 'label': '00001_x1_y1_zp1'}
```

#### Passo 3: Conectar vizinhos

```python
for x, y in G.nodes:
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neighbor = (x + dx, y + dy)
        if neighbor in G.nodes:
            G.add_edge((x, y), neighbor)
```

**Lógica:**

1. **Iterar nós:** para cada nó (x, y).
2. **Definir vizinhos:** 4 direções (cima, baixo, esquerda, direita).
   - `(-1, 0)`: esquerda.
   - `(1, 0)`: direita.
   - `(0, -1)`: cima.
   - `(0, 1)`: baixo.
3. **Calcular vizinho:** `(x + dx, y + dy)`.
4. **Conectar se existe:** adiciona aresta apenas se vizinho está no grafo.

**Exemplo:**
```
Se nós: {(0, 0), (1, 0), (0, 1)}
Para (0, 0):
  - Esquerda (-1, 0) → não existe
  - Direita (1, 0) → EXISTE → add edge (0,0)-(1,0)
  - Cima (0, -1) → não existe
  - Baixo (0, 1) → EXISTE → add edge (0,0)-(0,1)
```

**Estrutura resultante:**
```
(0,0) ------- (1,0)
  |
(0,1)
```

### 6.3 Resultado final

```python
return G
```

Grafo pronto com:
- N nós (um por tile válido).
- Arestas conectando vizinhos espaciais.
- Atributos de validade e label em cada nó.

---

## 7. Função `plot_graph(G: nx.Graph, output_path: Path, spacing: int = 100)`

### 7.1 Assinatura e responsabilidades

```python
def plot_graph(G: nx.Graph, output_path: Path, spacing: int = 100):
```

**Parâmetros:**
- `G`: grafo a visualizar.
- `output_path`: caminho da imagem PNG de saída.
- `spacing`: espaçamento visual entre nós (em pixels).

**Responsabilidade:** gerar visualização PNG do grafo para validação.

### 7.2 Implementação

#### Passo 1: Calcular posições dos nós

```python
pos = {
    (x, y): (x * spacing, -y * spacing) for (x, y) in G.nodes
}
```

- **Mapeamento coordenadas → posições de plot.**
- `(x, y)` → `(x * spacing, -y * spacing)`.
- **Negação de Y:** inverte eixo Y para exibição top-down (como esperado em imagens).
  - Sem negação: Y cresceria para baixo (padrão tela).
  - Com negação: Y cresce para cima (padrão gráfico).

**Exemplo:**
```python
Nó (1, 1) → posição (100, -100)
Nó (2, 1) → posição (200, -100)
Nó (1, 2) → posição (100, -200)
```

#### Passo 2: Definir cores dos nós

```python
node_colors = ["blue" if G.nodes[n]["valid"] else "red" for n in G.nodes]
```

- **Lista de cores** correspondendo aos nós.
- Azul: tile válido.
- Vermelho: tile inválido.

#### Passo 3: Extrair labels

```python
labels = {n: G.nodes[n]["label"] for n in G.nodes}
```

- Labels dos nós (nomes de tiles, ex.: "00001_x1_y1_zp1").

#### Passo 4: Criar figura e desenhar

```python
plt.figure(figsize=(12, 10))
nx.draw_networkx_edges(G, pos, edge_color="#aaaaaa")
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
nx.draw_networkx_labels(G, pos, labels=labels, font_size=4)
```

- **Figura:** 12×10 polegadas.
- **Arestas:** cor cinza claro.
- **Nós:** cores conforme validade, tamanho 100.
- **Labels:** font size 4 (pequeno para não poluir).

#### Passo 5: Adicionar legenda

```python
legend_elements = [
    Patch(facecolor="blue", edgecolor="black", label="Registrado"),
    Patch(facecolor="red", edgecolor="black", label="Não registrado"),
]
plt.legend(handles=legend_elements, loc="upper right")
```

- Patch customizado para legenda.
- Localizado no canto superior direito.

#### Passo 6: Finalizar e salvar

```python
plt.title("Grafo de Vizinhança dos Tiles (Azul = Registrado, Vermelho = Não Registrado)")
plt.axis("off")
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()
```

- Título descritivo.
- Desativa eixos (sem números/grid).
- Layout ajustado.
- Salva em alta resolução (300 DPI).
- Fecha figura para liberar memória.

### 7.3 Exemplo de saída

Imagem PNG com:
- Nós azuis representando tiles válidos.
- Nós vermelhos representando tiles inválidos.
- Arestas conectando vizinhos.
- Disposição espacial refletindo coordenadas reais (gradeado).

---

## 8. Função `get_or_build_topological_graph(valid_tiles: Dict[str, bool], pattern: re.Pattern, force_rebuild: bool = False) -> nx.Graph`

### 8.1 Assinatura e responsabilidades

```python
def get_or_build_topological_graph(
    valid_tiles: Dict[str, bool],
    pattern: re.Pattern,
    force_rebuild: bool = False,
) -> nx.Graph:
```

**Parâmetros:**
- `valid_tiles`: dicionário de tiles.
- `pattern`: regex para coordenadas.
- `force_rebuild`: se `True`, ignora cache em disco e reconstrói.

**Retorna:** grafo topológico (do disco ou recém-construído).

**Responsabilidade:** implementar caching inteligente (construct-once strategy).

### 8.2 Implementação

#### Caso 1: Usar grafo em cache

```python
graph_path = Config.TOPOLOGY_GRAPH_FILE

if graph_path.exists() and not force_rebuild:
    logger.info("Grafo topológico encontrado em disco. Carregando...")
    return load_graph(graph_path)
```

- Se arquivo existe E `force_rebuild` é `False`, carrega e retorna.
- Economiza tempo: não reconstrói a cada execução.

#### Caso 2: Construir novo grafo

```python
logger.info("Construindo novo grafo topológico...")
G = build_graph(valid_tiles, pattern)

save_graph(G, graph_path)

return G
```

- Se arquivo não existe OU `force_rebuild` é `True`, constrói novo.
- Persiste em disco para próximas execuções.
- Retorna grafo.

### 8.3 Benefícios

- **Performance:** grafo é construído uma vez, reutilizado.
- **Flexibilidade:** parâmetro `force_rebuild` permite forçar reconstrução se necessário.
- **Robustez:** grafo sempre retornado (do cache ou construído).

---

## 9. Função `generate_graph()`

### 9.1 Entry point principal

```python
def generate_graph():
    pattern = re.compile(Config.COORDINATES_PATTERN)

    valid_tiles = load_valid_tiles(Config.VALID_TILES_FILE)

    graph = get_or_build_topological_graph(valid_tiles, pattern)

    plot_graph(graph, Config.GRAPH_FILE)
```

**Fluxo:**

1. **Compilar regex:** padrão de coordenadas do Config.
2. **Carregar tiles válidos:** JSON de classificação.
3. **Obter/construir grafo:** logic de cache.
4. **Visualizar:** gera PNG.

**Orquestração clara:** cada etapa tem responsabilidade definida.

---

## 10. Entry point do módulo

```python
if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.DEBUG)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    generate_graph()
```

- Logging em nível DEBUG.
- Suprime avisos do matplotlib (que são verbosos).
- Execução: `python -m src.modules.graph.graph`

---

## 11. Configurações esperadas (`Config`)

O arquivo depende dos seguintes valores em `src.config.config.Config`:

| Parâmetro | Tipo | Descrição | Exemplo |
|-----------|------|-----------|---------|
| `COORDINATES_PATTERN` | str (regex) | Padrão para extrair coordenadas. | `r"(\d+)_x(\d+)_y(\d+)_zp(\d+)"` |
| `VALID_TILES_FILE` | Path | Caminho do JSON com tiles válidos. | `output/metadata/valid_tiles.json` |
| `TOPOLOGY_GRAPH_FILE` | Path | Caminho para salvar grafo (pickle). | `output/result/topology.gpickle` |
| `GRAPH_FILE` | Path | Caminho para salvar visualização (PNG). | `output/result/topology.png` |

---

## 12. Fluxo de dados resumido

```
valid_tiles.json (output/metadata/)
         ↓
   load_valid_tiles()
   {tile_name: is_valid, ...}
         ↓
   get_or_build_topological_graph()
   ├─ check se arquivo em disco existe
   ├─ se sim: load_graph()
   └─ se não: build_graph()
         ↓
      [build_graph]
      - para cada tile, extract_coordinates()
      - add_node(coord, valid=is_valid, label=tile_name)
      - para cada nó, conectar vizinhos (4 direções)
         ↓
   save_graph() → topology.gpickle
         ↓
   plot_graph()
   - calcular posições dos nós
   - colorir por validade
   - adicionar labels
   - salvar PNG em alta res
         ↓
   topology.png
   +
   topology.gpickle
```

---

## 13. Estrutura de dados: o grafo

### 13.1 Nós

Cada nó representa um tile. Formato:
```python
G.nodes[(x, y)] = {
    'valid': bool,      # tile é válido (passou em classification)
    'label': str        # nome do tile, ex.: '00001_x1_y1_zp1'
}
```

### 13.2 Arestas

Cada aresta conecta vizinhos espaciais. Formato:
```python
G.edges[((x1, y1), (x2, y2))] = {}  # sem atributos no grafo topológico
```

### 13.3 Exemplo

```python
G = build_graph(
    {'00001_x1_y1_zp1': True, '00002_x2_y1_zp1': True, '00001_x1_y2_zp1': True},
    pattern
)

# Resultado:
# Nós: (1, 1), (2, 1), (1, 2)
# G.nodes[(1, 1)] = {'valid': True, 'label': '00001_x1_y1_zp1'}
# G.nodes[(2, 1)] = {'valid': True, 'label': '00002_x2_y1_zp1'}
# G.nodes[(1, 2)] = {'valid': True, 'label': '00001_x1_y2_zp1'}

# Arestas:
# G.edges: [((1,1), (2,1)), ((1,1), (1,2))]
```

---

## 14. Casos de tratamento e robustez

### Arquivo de tiles válidos não existe
- **Comportamento:** `load_valid_tiles()` levanta `FileNotFoundError`.
- **Recomendação:** adicionar try-except em `generate_graph()`.

### Padrão regex não funciona
- **Comportamento:** `extract_coordinates()` levanta `ValueError` ou retorna None.
- **Consequência:** nó não é adicionado ao grafo.
- **Recomendação:** validar padrão antes; logar warnings para tiles ignorados.

### Grafo vazio
- **Comportamento:** se nenhum tile gera coordenadas válidas, grafo fica vazio.
- **Impacto:** `plot_graph()` gera imagem vazia; próximas etapas falham.
- **Recomendação:** validar que grafo tem nós antes de continuar.

### Falha ao salvar pickle/PNG
- **Comportamento:** exceção levantada.
- **Recomendação:** adicionar try-except com feedback claro.

---

## 15. Otimizações e pontos de atenção

### ✅ Boas práticas presentes

1. **Caching inteligente:** grafo construído uma vez, reutilizado.
2. **Visualização incluída:** validação visual do grafo é crítica.
3. **Pickle para serialização:** simples e eficiente para pipeline interno.
4. **Atributos em nós:** suporta extensões futuras (ex.: arestas com pesos).
5. **Logging informativo:** fácil debug.

### ⚠️ Possíveis melhorias

1. **Validação de grafos vazios:** checar antes de salvar/visualizar.
2. **Suporte a diferentes topologias:** atualmente assume 4-vizinhos; poderia ser extensível.
3. **Estatísticas do grafo:** logar número de nós, arestas, componentes conexos.
4. **Melhor tratamento de erros:** try-except em `generate_graph()`.
5. **Personalização de visualização:** cores, spacing, via Config.
6. **Métricas de conectividade:** detectar tiles isolados antes de matching.

---

## 16. Exemplo de uso

### Setup mínimo

```python
from pathlib import Path
from src.config.config import Config

Config.COORDINATES_PATTERN = r"(\d+)_x(\d+)_y(\d+)_zp(\d+)"
Config.VALID_TILES_FILE = Path("output/metadata/valid_tiles.json")
Config.TOPOLOGY_GRAPH_FILE = Path("output/result/topology.gpickle")
Config.GRAPH_FILE = Path("output/result/topology.png")
```

### Execução

```bash
python -m src.modules.graph.graph
```

### Consulta de resultados

```python
from src.modules.graph.graph import load_graph

G = load_graph(Path("output/result/topology.gpickle"))

print(f"Número de nós: {G.number_of_nodes()}")
print(f"Número de arestas: {G.number_of_edges()}")

# Inspecionar um nó
node = list(G.nodes())[0]
print(f"Nó: {node}")
print(f"Atributos: {G.nodes[node]}")

# Vizinhos de um nó
neighbors = list(G.neighbors(node))
print(f"Vizinhos: {neighbors}")
```

**Exemplo de saída:**
```
Número de nós: 150
Número de arestas: 348
Nó: (1, 1)
Atributos: {'valid': True, 'label': '00001_x1_y1_zp1'}
Vizinhos: [(2, 1), (1, 2)]
```

---

## 17. Conexão com pipeline

### Inputs (dependências anteriores)
- `src.modules.tile.classify.classifier` → gera `valid_tiles.json`.

### Outputs (fornecidos para próximo passo)
- `topology.gpickle` → consumido por `src.modules.features.match.match` (define pares a fazer match).
- `topology.png` → validação visual pelo usuário.

### Próximo passo
- **Matching de features:** extrai correspondências entre tiles vizinhos (conforme grafo topológico).

---

## 18. NetworkX basics

O código usa conceitos de grafo do NetworkX:

| Conceito | Exemplo |
|----------|---------|
| **Nó** | `G.add_node((1, 1), valid=True, label='00001')` |
| **Atributos de nó** | `G.nodes[(1, 1)]['valid']` |
| **Aresta** | `G.add_edge((1, 1), (2, 1))` |
| **Vizinhos de nó** | `G.neighbors((1, 1))` |
| **Número de nós** | `G.number_of_nodes()` |
| **Número de arestas** | `G.number_of_edges()` |
| **Grau de nó** | `G.degree((1, 1))` |
| **Iterar nós** | `for node in G.nodes(): ...` |
| **Iterar arestas** | `for edge in G.edges(): ...` |

---

## 19. Resumo

O `graph.py` é o **construtor do grafo topológico** do pipeline:

- **Mapeia** tiles para coordenadas e as conecta por vizinhança.
- **Persiste** em pickle para rápido acesso posterior.
- **Visualiza** para validação manual.
- **Implementa caching:** construído uma vez, reutilizado.
- **Simples** e focado: responsabilidade bem definida.

O grafo topológico é essencial: define quais pares de tiles devem ter features matchadas. Sem ele, não há estrutura para o restante do pipeline (matching, grafo geométrico, propagação de posições).

A visualização PNG oferece feedback crítico: permite ao usuário validar se a topologia foi corretamente interpretada a partir dos nomes de tiles.

