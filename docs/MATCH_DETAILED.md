# Documentação Detalhada: `match.py`

**Arquivo:** `src/modules/features/match/match.py`

**Propósito:** Calcular correspondências (matches) entre pares de tiles vizinhos com base em descritores de features, aplicar RANSAC para robustez, e persister matrizes de transformação estimadas.

---

## 1. Visão geral

O módulo `match.py` é responsável pela **etapa de matching de features** no pipeline de stitching. Ele:

1. Carrega keypoints e descritores extraídos (resultado de `detect.py`).
2. Para cada par de tiles vizinhos, realiza **matching de descritores**.
3. Filtra matches robustos aplicando **RANSAC** para estimar transformação afim (translação).
4. Persiste **matriz de transformação** e métricas em `.zarr` por par de tiles.

**Contexto no pipeline:**
- Entrada: features em `output/features/features.zarr` (resultado de `detect`).
- Saída: matches em `output/matches/` (um `.zarr` por par de tiles).
- Próxima etapa: construção do grafo geométrico a partir dos deslocamentos estimados.

---

## 2. Importações e dependências

```python
import re                      # Expressões regulares
import time                    # Medição de tempo
from typing import ...         # Type hints

import cv2                     # OpenCV (RANSAC, transformações)
import numpy as np             # Operações com arrays
import zarr                    # Armazenamento de dados
from joblib import Parallel, delayed   # Paralelização

from src.config.config import Config                      # Configurações centralizadas
from src.utils.coordinates import extract_coordinates    # Extrator de coordenadas
from .registry import get_matcher                         # Factory para matcher
```

**Papel de cada import:**
- `cv2`: núcleo do matching (KNN, RANSAC, transformações).
- `numpy`: álgebra linear (multiplicação de matrizes, máscara de inliers).
- `zarr`: armazenamento eficiente de matches e metadados.
- `joblib`: paralelização de pares de tiles.
- `Config`, `extract_coordinates`: injeção de configuração.
- `get_matcher`: factory para instanciar matcher (BruteForceMatcher, FlannMatcher, etc.).

---

## 3. Função `load_keypoints_and_descriptors(zarr_store, tile_name: str)`

### 3.1 Assinatura e responsabilidades

```python
def load_keypoints_and_descriptors(zarr_store, tile_name: str):
```

**Parâmetros:**
- `zarr_store`: store zarr aberto (modo "r") contendo features de todos os tiles.
- `tile_name` (str): nome do tile (ex.: "00001_x1_y1_zp1").

**Retorna:** tupla `(keypoints, descriptors)`
- `keypoints`: lista de objetos `cv2.KeyPoint`.
- `descriptors`: array numpy uint8 com descritores (shape: `(N, descriptor_size)`).

**Responsabilidade:** reconstruir estruturas OpenCV a partir de arrays Zarr para uso em matching.

### 3.2 Passo a passo

#### Passo 1: Acessar grupo do tile

```python
group = zarr_store[tile_name]
kp_array = group["keypoints"][:]
descriptors = group["descriptors"][:]
```

- Acessa grupo específico do tile no zarr (criado por `detect.py`).
- Carrega arrays em memória (slice `[:]`).

#### Passo 2: Reconstruir KeyPoints OpenCV

```python
keypoints = [
    cv2.KeyPoint(
        x=float(row[0]),
        y=float(row[1]),
        size=float(row[2]),
        angle=float(row[3]),
        response=float(row[4]),
        octave=int(row[5]),
        class_id=int(row[6]),
    )
    for row in kp_array
]
```

**Reconstrução:** array armazenado em 7 colunas (ex-output de `detect.py`) → objetos `cv2.KeyPoint`.

**Campos:**
- `x, y`: coordenadas do keypoint.
- `size`: escala/tamanho.
- `angle`: orientação (em graus).
- `response`: intensidade de resposta (confiança).
- `octave`: nível de pirâmide Gaussiana.
- `class_id`: identificador de classe (raramente usado).

#### Passo 3: Logging e retorno

```python
print(f"Tile {tile_name}: {len(keypoints)} keypoints, descritores shape = {descriptors.shape}")
return keypoints, descriptors
```

---

## 4. Função `match_pair(tile_a, tile_b) -> int`

### 4.1 Assinatura e responsabilidades

```python
def match_pair(tile_a, tile_b) -> int:
```

**Parâmetros:**
- `tile_a`, `tile_b` (str): nomes dos tiles (par vizinho).

**Retorna:** int (`0` ou `1`)
- `1`: matches salvos com sucesso.
- `0`: falha (sem matches, erro, etc.).

**Responsabilidade:** processar um par de tiles isoladamente (função paralelizável).

### 4.2 Passo a passo

#### Passo 1: Instanciar matcher

```python
matcher = get_matcher(
    Config.MATCHER,
    algorithm=Config.DETECTION_ALGORITHM,
    ratio_thresh=Config.MATCHING_RATIO_THRESH,
)
store = zarr.open(Config.KEYPOINTS_ZARR_STORE, mode="r")
```

- Factory `get_matcher()` retorna matcher configurado (ex.: FLANN ou BruteForceMatcher).
- Abre store zarr em modo leitura.

#### Passo 2: Carregar features dos dois tiles

```python
try:
    kp1, desc1 = load_keypoints_and_descriptors(store, tile_a)
    kp2, desc2 = load_keypoints_and_descriptors(store, tile_b)
except Exception as e:
    print(f"Falha ao carregar dados: {tile_a} <-> {tile_b}. Erro: {e}")
    return 0
```

- Carrega keypoints e descritores.
- Se falha, retorna 0 (graceful degradation).

#### Passo 3: Validação básica

```python
if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
    print(f"Descritores vazios: {tile_a} ou {tile_b}")
    return 0
```

- Rejeita tiles sem features (evita erro em matcher).

#### Passo 4: Matching inicial

```python
raw_matches = matcher.match(kp1, desc1, kp2, desc2)
raw_match_count = len(raw_matches)

if len(raw_matches) < 4:
    print(f"Matches insuficientes para RANSAC: {len(raw_matches)}")
    return 0
```

**Lógica:**
- Matcher calcula correspondências (ex.: KNN + ratio test via Lowe's criterion).
- Retorna lista de `cv2.DMatch` objects.
- **Threshold:** mínimo 4 matches para estimar transformação afim (RANSAC requer 4 pontos).

#### Passo 5: Preparar pontos para RANSAC

```python
src_pts = np.float32([kp1[m.queryIdx].pt for m in raw_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in raw_matches]).reshape(-1, 1, 2)

src_xy = src_pts.reshape(-1, 2).astype(np.float32)
dst_xy = dst_pts.reshape(-1, 2).astype(np.float32)
```

**Extração:**
- `queryIdx`: índice em `kp1` (tile A).
- `trainIdx`: índice em `kp2` (tile B).
- Reshape: `(-1, 1, 2)` para compatibilidade com OpenCV; depois reshape para `(-1, 2)`.

**Arrays:**
- `src_xy`: coordenadas em tile A (shape: `(N, 2)`).
- `dst_xy`: coordenadas correspondentes em tile B (shape: `(N, 2)`).

#### Passo 6: Estimar transformação afim com RANSAC

```python
M, mask = cv2.estimateAffinePartial2D(
    src_xy,
    dst_xy,
    method=cv2.RANSAC,
    ransacReprojThreshold=5.0,
    maxIters=2000,
    confidence=0.99,
    refineIters=10,
)

if M is None or mask is None:
    print(f"Falha ao estimar translação robusta para {tile_a} e {tile_b}")
    return 0
```

**Função `estimateAffinePartial2D`:**
- Estima transformação afim parcial (apenas translação + rotação, sem escala/shear).
- **Parâmetros:**
  - `ransacReprojThreshold`: tolância de erro (5 pixels).
  - `maxIters`: iterações máximas do RANSAC (2000).
  - `confidence`: confiança (0.99 = 99%).
  - `refineIters`: refinamento final (10 iterações).
- **Retorna:**
  - `M`: matriz 2×3 de transformação afim (apenas translação relevante).
  - `mask`: array binário indicando inliers (1) e outliers (0).

**Nota:** comentário menciona `findHomography` como alternativa (homografia completa), mas `estimateAffinePartial2D` é mais conservador (apenas translação).

#### Passo 7: Extrair translação

```python
dx = float(M[0, 2])
dy = float(M[1, 2])
```

- `M[0, 2]`: deslocamento em X.
- `M[1, 2]`: deslocamento em Y.

#### Passo 8: Construir matriz homogênea 3×3

```python
matrix = np.array(
    [
        [1.0, 0.0, dx],
        [0.0, 1.0, dy],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
```

**Formato padrão:** matriz homogênea 3×3 para coordenadas 2D.
- Apenas translação: `(1, 0, dx; 0, 1, dy; 0, 0, 1)`.
- Usada por etapas posteriores (canvas.populate).

#### Passo 9: Filtrar inliers

```python
matches_mask = mask.ravel().tolist()
good_matches = [raw_matches[i] for i in range(len(raw_matches)) if matches_mask[i]]
```

- Seleciona apenas matches cujo ponto foi classificado como inlier pelo RANSAC.

#### Passo 10: Calcular RMSE de reprojeção nos inliers

```python
inlier_src = src_xy[mask.ravel() == 1]
inlier_dst = dst_xy[mask.ravel() == 1]

pred = (inlier_src @ M[:, :2].T) + M[:, 2]
err = inlier_dst - pred
ransac_rmse = float((err[:, 0] ** 2 + err[:, 1] ** 2).mean() ** 0.5)
```

**Cálculo:**
1. Seleciona pontos inliers.
2. Aplica transformação M: `pred = src @ M[:2, :2].T + M[:, 2]`.
3. Calcula erro: `err = dst - pred`.
4. RMSE: raiz quadrada da média de `(dx² + dy²)`.

**Interpretação:** quanto menor RMSE, melhor a estimativa de transformação.

#### Passo 11: Validação final

```python
print(f"[RANSAC] Filtrados {len(good_matches)} inliers de {len(raw_matches)} matches totais.")

if len(good_matches) == 0:
    return 0
```

- Rejeita pares sem inliers (transformação não confiável).

#### Passo 12: Salvar em Zarr

```python
match_filename = f"{tile_a}__{tile_b}.zarr"
output_path = Config.MATCHING_ZARR_PATH / match_filename
match_zarr_store = zarr.open(output_path, mode="w")
group = match_zarr_store.create_group("matches", overwrite=True)
```

- Cria um `.zarr` por par de tiles (ex.: `00001_x1_y1_zp1__00002_x2_y1_zp1.zarr`).
- Dentro, grupo "matches" armazena dados.

#### Passo 13: Armazenar matches validados

```python
zarr.array(
    np.array([(m.queryIdx, m.trainIdx) for m in good_matches]),
    store=group.store,
    path=f"{group.path}/matches",
    chunks=(100, 2),
    dtype=int,
)
```

- Array 2D: cada linha é `(índice_em_A, índice_em_B)`.
- Chunks de 100 linhas para acesso eficiente.

#### Passo 14: Armazenar metadados

```python
group.attrs["tile_a"] = tile_a
group.attrs["tile_b"] = tile_b
group.attrs["translation_matrix"] = matrix.tolist()  # Converter para lista JSON
group.attrs["inlier_count"] = len(good_matches)
group.attrs["raw_match_count"] = int(raw_match_count)
group.attrs["ransac_rmse"] = float(ransac_rmse)
```

**Metadados armazenados:**
- Nomes dos tiles.
- Matriz de transformação (serializada como lista para compatibilidade JSON).
- Contagem de inliers e matches totais.
- RMSE de reprojeção.

**Importância:** próximas etapas (geograph.py, populate_geom.py) usam `translation_matrix` e `inlier_count`.

#### Passo 15: Log e retorno

```python
print(f"[SALVO] {output_path} com matriz de transformação.")
return 1
```

- Sucesso: retorna 1 (para contagem em paralelização).

---

## 5. Função `match()`

### 5.1 Entry point principal

```python
def match():
```

**Responsabilidade:** orquestrar matching de todos os pares de tiles.

### 5.2 Passo a passo

#### Passo 1: Validar diretórios

```python
Config.MATCHING_ZARR_PATH.mkdir(parents=True, exist_ok=True)
```

- Cria diretório de saída se não existir.

#### Passo 2: Carregar features e filtrar tiles válidos

```python
with open(Config.VALID_TILES_FILE) as f:
    valid_tiles = json.load(f)

store = zarr.open(Config.KEYPOINTS_ZARR_STORE, mode="r")
all_tile_names = list(store.group_keys())

# Filtrar apenas tiles válidos
tile_names = [tile for tile in all_tile_names if valid_tiles.get(tile, False)]
print(f"Tiles encontrados no Zarr: {len(all_tile_names)}")
print(f"Tiles válidos para matching: {len(tile_names)}")
```

- Abre store zarr de features (resultado de `detect.py`).
- Carrega `valid_tiles.json` (resultado de `classifier`) para filtrar tiles inválidos.
- Apenas tiles marcados como `True` em `valid_tiles` participam do matching.
- Isso evita processamento desnecessário de tiles sem conteúdo significativo.

#### Passo 3: Extrair coordenadas dos tiles

```python
pattern = re.compile(Config.COORDINATES_PATTERN)
tile_coords = {tile: extract_coordinates(tile, pattern) for tile in tile_names}
coord_to_tile = {v: k for k, v in tile_coords.items()}
```

- `tile_coords`: mapeamento `{tile_name: (x, y)}`.
- `coord_to_tile`: mapeamento reverso `{(x, y): tile_name}` para lookup rápido.

#### Passo 4: Determinar pares vizinhos

```python
tile_pairs = set()
for tile_name, (x, y) in tile_coords.items():
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        neighbor_coords = (x + dx, y + dy)
        neighbor_tile = coord_to_tile.get(neighbor_coords)
        if neighbor_tile:
            pair = tuple(sorted((tile_name, neighbor_tile)))
            tile_pairs.add(pair)

print(f"Total de pares únicos de vizinhos: {len(tile_pairs)}")
```

**Lógica:**
1. Para cada tile, calcula coordenadas de 4 vizinhos.
2. Verifica se vizinho existe.
3. Adiciona par à set (ordenado para evitar duplicatas).

**Exemplo:**
```
Tile (1, 1) com vizinhos (2, 1), (0, 1), (1, 2), (1, 0):
  Pares: {(00001, 00002), (00001, 00004), (00001, 00005), ...}
```

#### Passo 5: Processamento paralelo

```python
total_salvos = Parallel(n_jobs=Config.MATCHING_N_JOBS)(
    delayed(match_pair)(tile_a, tile_b) for tile_a, tile_b in sorted(tile_pairs)
)

print(f"\n[FIM] Total de matches salvos: {sum(total_salvos)}")
```

- `joblib.Parallel`: distribui matching entre N workers.
- `sorted()`: ordem determinística (reprodutibilidade).
- `sum()`: conta quantos pares foram salvos com sucesso (número de 1s retornados).

---

## 6. Entry point do módulo

```python
if __name__ == "__main__":
    start_time = time.perf_counter()
    match()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")
```

- Medição de tempo total.
- Execução: `python -m src.modules.features.match.match`

---

## 7. Configurações esperadas (`Config`)

O arquivo depende dos seguintes valores em `src.config.config.Config`:

| Parâmetro | Tipo | Descrição | Exemplo |
|-----------|------|-----------|---------|
| `KEYPOINTS_ZARR_STORE` | Path | Caminho do zarr com features (entrada). | `output/features/features.zarr` |
| `MATCHING_ZARR_PATH` | Path | Diretório para salvar matches (saída). | `output/matches/` |
| `COORDINATES_PATTERN` | str (regex) | Padrão para extrair coordenadas. | `r"(\d+)_x(\d+)_y(\d+)_zp(\d+)"` |
| `MATCHER` | str | Tipo de matcher. | `"flann"` ou `"bfmatcher"` |
| `DETECTION_ALGORITHM` | str | Algoritmo de detecção. | `"sift"` ou `"orb"` |
| `MATCHING_RATIO_THRESH` | float | Threshold de ratio Lowe. | `0.7` |
| `MATCHING_N_JOBS` | int | Número de workers paralelos. | `4` ou `-1` (todos cores) |

---

## 8. Fluxo de dados resumido

```
output/features/features.zarr (features de todos tiles)
         ↓
   load zarr store
   tile_names = [00001_x1_y1_zp1, 00002_x2_y1_zp1, ...]
         ↓
   extract coordenadas de cada tile
   tile_coords = {00001: (1,1), 00002: (2,1), ...}
         ↓
   determinar pares vizinhos (4-vizinhos)
   tile_pairs = {(00001, 00002), (00001, 00004), ...}
         ↓
   for each pair in parallel via joblib
         ↓
      [match_pair(A, B)]
      - load keypoints/descriptors de A e B
      - matcher.match() → raw_matches
      - RANSAC → M, mask, inliers
      - validar inliers > 0
      - salvar .zarr com matches, matrix, métricas
         ↓
   output/matches/A__B.zarr (um por par bem-sucedido)
```

---

## 9. Estrutura Zarr de matches

### 9.1 Por par de tiles

Exemplo: `output/matches/00001_x1_y1_zp1__00002_x2_y1_zp1.zarr/`

```
matches.zarr/
├── matches
│   └── array (shape: (N_inliers, 2), dtype: int)
│       - Cada linha: (queryIdx, trainIdx)
└── attrs
    ├── tile_a: "00001_x1_y1_zp1"
    ├── tile_b: "00002_x2_y1_zp1"
    ├── translation_matrix: [[1, 0, dx], [0, 1, dy], [0, 0, 1]]  (lista 3×3)
    ├── inlier_count: int (número de inliers do RANSAC)
    ├── raw_match_count: int (matches antes de RANSAC)
    └── ransac_rmse: float (RMSE de reprojeção)
```

---

## 10. RANSAC e transformações

### 10.1 O que é RANSAC

**RANSAC** (Random Sample Consensus):
- Algoritmo robusto para estimar modelo (ex.: transformação geométrica).
- Iterativamente seleciona subconjuntos aleatórios de pontos.
- Ajusta modelo a cada subset.
- Conta inliers (pontos bem ajustados).
- Retorna melhor modelo.

**Vantagem:** rejeita outliers automaticamente.

### 10.2 Transformação Afim Parcial

`cv2.estimateAffinePartial2D` estima:
- **Translação:** Δx, Δy
- **Rotação:** (opcional, depende dos dados)
- **Sem escala/shear:** preserva formas

**Matriz 2×3:**
```
[1    0   tx]
[0    1   ty]
```

**Convertida para 3×3 para compatibilidade:**
```
[1    0   tx]
[0    1   ty]
[0    0   1 ]
```

### 10.3 Reprojeção e RMSE

**Reprojeção:** aplicar transformação aos pontos fonte e comparar com destino.

```python
pred = (src @ M[:2, :2].T) + M[:, 2]    # Aplicar transformação
err = dst - pred                         # Erro
rmse = sqrt(mean(dx² + dy²))             # Root mean square error
```

**Interpretação:** quanto menor RMSE, melhor a transformação.

---

## 11. Factory de matchers

O código usa `get_matcher()` para instanciar matcher. Exemplo de implementação esperada em `registry.py`:

```python
def get_matcher(matcher_type, algorithm, ratio_thresh):
    if matcher_type == "flann":
        return FlannMatcher(algorithm, ratio_thresh)
    elif matcher_type == "bfmatcher":
        return BruteForceMatcher(algorithm, ratio_thresh)
    else:
        raise ValueError(f"Matcher desconhecido: {matcher_type}")
```

**Matchers comuns:**
- **BruteForceMatcher:** compara todos os pares (lento mas robusto).
- **FLANN:** árvores KD-tree (rápido, bom para features de alta dimensão).

---

## 12. Casos de tratamento e robustez

### Tile sem features

**Comportamento:** `desc1` ou `desc2` é None/vazio → retorna 0.

**Impacto:** par não contribui ao grafo geométrico.

### Matches insuficientes (< 4)

**Comportamento:** RANSAC precisa mínimo 4 pontos → retorna 0.

**Impacto:** par sem conexão no grafo geométrico.

### RANSAC falha (M ou mask é None)

**Comportamento:** `cv2.estimateAffinePartial2D` retorna None → retorna 0.

**Causa:** configuração de RANSAC muito restritiva ou dados patológicos.

### Nenhum inlier após RANSAC

**Comportamento:** todos os matches são outliers → retorna 0.

**Impacto:** transformação não confiável, par descartado.

### Falha ao salvar Zarr

**Comportamento:** exceção não capturada pode interromper worker.

**Recomendação:** adicionar try-except em `match_pair`.

---

## 13. Otimizações e pontos de atenção

### ✅ Boas práticas presentes

1. **RANSAC para robustez:** remove outliers automaticamente.
2. **Paralelização:** N workers para N pares.
3. **Cálculo de RMSE:** métrica de confiança da transformação.
4. **Armazenamento modular:** um zarr por par, fácil acesso posterior.
5. **Metadados ricos:** matrix, inlier_count, RMSE salvos para análise.

### ⚠️ Possíveis melhorias

1. **Tratamento robusto de exceções:** try-except em `match_pair` para não falhar por outlier.
2. **Threshold adaptativo:** `ransacReprojThreshold` poderia variar conforme tamanho de tile.
3. **Pré-filtro de matches:** antes de RANSAC, remover matches muito ruins (ex.: Lowe's ratio test).
4. **Validação de transformação:** rejeitar se `dx` e `dy` estão fora de bounds esperados.
5. **Logging de falhas:** contar e registrar quantos pares falharam e por quê.
6. **Checkpoint/incremental:** se houver falha, retomar do último par processado.
7. **Batch processing:** em lugar de zarr por par, um zarr global com partições.

---

## 14. Exemplo de uso

### Setup mínimo

```python
from src.config.config import Config

Config.KEYPOINTS_ZARR_STORE = Path("output/features/features.zarr")
Config.MATCHING_ZARR_PATH = Path("output/matches/")
Config.MATCHER = "flann"
Config.DETECTION_ALGORITHM = "sift"
Config.MATCHING_RATIO_THRESH = 0.7
Config.MATCHING_N_JOBS = 4
```

### Execução

```bash
python -m src.modules.features.match.match
```

### Consulta de resultados

```python
import zarr
from pathlib import Path

matches_dir = Path("output/matches/")

for match_file in sorted(matches_dir.glob("*.zarr")):
    store = zarr.open(match_file, mode="r")
    group = store["matches"]
    
    tile_a = group.attrs["tile_a"]
    tile_b = group.attrs["tile_b"]
    inliers = group.attrs["inlier_count"]
    matrix = group.attrs["translation_matrix"]
    rmse = group.attrs["ransac_rmse"]
    
    print(f"{tile_a} <-> {tile_b}")
    print(f"  Inliers: {inliers}")
    print(f"  Translation: dx={matrix[0][2]:.2f}, dy={matrix[1][2]:.2f}")
    print(f"  RMSE: {rmse:.4f}")
```

**Exemplo de saída:**
```
00001_x1_y1_zp1 <-> 00002_x2_y1_zp1
  Inliers: 145
  Translation: dx=-512.34, dy=12.45
  RMSE: 1.2345
00001_x1_y1_zp1 <-> 00003_x1_y2_zp1
  Inliers: 92
  Translation: dx=8.23, dy=-512.67
  RMSE: 2.1234
```

---

## 15. Conexão com pipeline

### Inputs (dependências anteriores)
- `src.modules.features.detect.detect` → gera `output/features/features.zarr` com keypoints/descritores.
- `src.modules.graph.graph` → define implicitamente vizinhos via coordenadas.

### Outputs (fornecidos para próximo passo)
- `output/matches/*.zarr` → consumido por:
  - `src.modules.graph.geograph` (extrai `translation_matrix` para construir grafo geométrico).
  - `src.modules.canvas.populate_geom` (usa `translation_matrix` para colagem).

### Próximo passo
- **Grafo geométrico:** construir grafo com arestas ponderadas pelos deslocamentos.

---

## 16. Algoritmos de matching

### BruteForceMatcher

- Compara cada descritor de A com todos de B.
- Retorna 2 melhores matches para ratio test (Lowe's criterion).
- Lento para descritores grandes ou muitos tiles.

### FLANN (Fast Approximate Nearest Neighbors)

- Usa estruturas de índice (KD-trees para SIFT, hash tables para ORB).
- Muito mais rápido que brute force.
- Recomendado para SIFT e descritores de alta dimensão.

### Lowe's Ratio Test

- Compara distância do 1º melhor com 2º melhor match.
- Se `dist1 / dist2 < threshold` (ex.: 0.7), considera match bom.
- Filtra matches ambíguos.

---

## 17. Resumo

O `match.py` é o **calculador de correspondências** do pipeline:

- **Extrai matches** entre tiles vizinhos usando descritores.
- **Aplica RANSAC** para robustez contra outliers.
- **Estima transformação afim** (translação + possível rotação).
- **Persiste em Zarr** modular para acesso eficiente.
- **Paraleliza** para performance em datasets grandes.

Sua saída (matrizes de transformação + métricas de confiança) é essencial para:
1. **Grafo geométrico:** definir deslocamentos entre tiles.
2. **Propagação de posições:** calcular layout final do mosaico.
3. **Colagem (canvas):** posicionar tiles corretamente.

A inclusão de RMSE e inlier_count permite análise posterior: pares com low RMSE são confiáveis; pares com poucos inliers podem ser filtrados se houver múltiplas estimativas disponíveis.

