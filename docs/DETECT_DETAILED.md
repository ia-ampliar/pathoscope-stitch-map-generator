# Documentação Detalhada: `detect.py`

**Arquivo:** `src/modules/features/detect/detect.py`

**Propósito:** Extrair e armazenar keypoints e descritores de features de tiles de imagem em paralelo, utilizando algoritmos configuráveis de detecção.

---

## 1. Visão geral

O módulo `detect.py` é responsável pela **etapa de detecção de features** no pipeline de stitching. Ele:

1. Carrega os metadados dos tiles e a lista de tiles válidos (não descartados).
2. Processa cada tile em **paralelo** (via `joblib.Parallel`).
3. Detecta keypoints e calcula descritores usando um algoritmo configurável (SIFT ou ORB).
4. Armazena os resultados em uma **estrutura Zarr** para acesso eficiente posterior.

**Contexto no pipeline:**
- Entrada: tiles normalizados em `output/tiles/normalized/` + metadados JSON (`output/metadata/dataset.json`).
- Saída: `.zarr` centralizado em `output/features/` com keypoints e descritores por tile.
- Próxima etapa: matching entre tiles vizinhos.

---

## 2. Importações e dependências

```python
import gc                          # Limpeza de memória (garbage collection)
import json                        # Leitura/escrita de JSON
import logging                     # Sistema de logging
from pathlib import Path           # Manipulação de caminhos (type-safe)
from time import perf_counter      # Medição de tempo de execução

import cv2                         # OpenCV (leitura de imagens, detecção)
import numpy as np                 # Operações com arrays
import zarr                        # Armazenamento HDF5-like para dados n-dimensionais
from joblib import Parallel, delayed  # Paralelização de tarefas

from src.config.config import Config       # Configurações centralizadas
from .registry import get_detector         # Factory para instanciar detectores
```

**Papel de cada import:**
- `gc` e `perf_counter`: monitoramento de performance e liberação de memória.
- `cv2`, `numpy`, `zarr`: núcleo do processamento de imagens e armazenamento.
- `joblib`: paralelização eficiente com suporte a pickle (serialização).
- `Config` e `get_detector`: injeção de configuração e padrão factory.

---

## 3. Função `process_tile(algorithm, tile, valid_tiles)`

### 3.1 Assinatura e responsabilidades

```python
def process_tile(algorithm, tile, valid_tiles):
```

**Parâmetros:**
- `algorithm` (str): nome do algoritmo de detecção (ex.: "SIFT", "ORB").
- `tile` (dict): dicionário com chaves `name`, `path`, `coordinates`.
- `valid_tiles` (dict): mapeamento `{tile_name: bool}` indicando se o tile é válido.

**Retorna:** tupla com 5 elementos
```python
(tile_name, kp_array, descriptors, registered, coordinates)
```

**Responsabilidade:** processar um único tile de forma isolada (sem estado compartilhado).

---

### 3.2 Passo a passo

#### Passo 1: Inicializar detector

```python
detector = get_detector(algorithm)
```

- Instancia um novo detector dentro da função.
- **Razão:** evitar problemas de serialização (pickle) em ambientes paralelos.
- A factory `get_detector()` encapsula a criação de detectores (SIFT, ORB, etc.).

#### Passo 2: Verificar se o tile é válido

```python
tile_name = Path(tile["name"]).stem
if not valid_tiles.get(tile_name, False):
    return tile_name, None, None, False, tile["coordinates"]
```

- Extrai o nome base do arquivo (sem extensão) usando `Path.stem`.
- Se não está em `valid_tiles` ou `valid_tiles[tile_name] == False`, retorna `None` para keypoints/descriptors.
- Isso permite que tiles inválidos (muito pequenos, sem conteúdo, etc.) sejam ignorados silenciosamente.

#### Passo 3: Carregar imagem

```python
img_path = Path(tile["path"]) / tile["name"]
img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
if img is None:
    logger.error(f"[!] Erro ao abrir imagem: {img_path}")
    return tile_name, None, None, False, tile["coordinates"]
```

- Constrói caminho seguro usando `Path`.
- Lê a imagem em **escala de cinza** (requisito para a maioria dos detectores).
- Se falhar, registra erro e retorna tuple de falha.

#### Passo 4: Detectar keypoints e descritores

```python
keypoints, descriptors = detector.detect_and_compute(img)
```

- Usa o detector configurado para extrair features.
- `keypoints`: lista de objetos `cv2.KeyPoint`.
- `descriptors`: array numpy com descritores. O tamanho depende do algoritmo:
  - **ORB**: 32 bytes por keypoint → shape `(N, 32)`, dtype nativo `uint8`.
  - **SIFT**: 128 floats por keypoint → shape `(N, 128)`, dtype nativo `float32` (convertido para `uint8` no passo seguinte).

#### Passo 5: Validar e converter para formato Zarr-compatível

```python
if keypoints and len(keypoints) > 4:
    kp_array = np.array(
        [
            [
                kp.pt[0],          # x
                kp.pt[1],          # y
                kp.size,           # tamanho do keypoint
                kp.angle,          # ângulo de orientação
                kp.response,       # intensidade da resposta (confiança)
                kp.octave,         # oitava (nível de pirâmide)
                kp.class_id,       # ID de classe (se aplicável)
            ]
            for kp in keypoints
        ],
        dtype=np.float32,
    )
    descriptors = descriptors.astype(np.uint8)
    registered = True
else:
    kp_array = np.zeros((0, 7), dtype=np.float32)
    descriptors = np.zeros((0, 32), dtype=np.uint8)
    registered = False
```

**Conversão de keypoints:**
- Cada `cv2.KeyPoint` é decomponível em 7 atributos numéricos.
- Array resultante: shape `(n_keypoints, 7)`, type `float32`.
- Threshold: rejeita tiles com < 5 keypoints como "sem features".

**Conversão de descritores:**
- Cast para `uint8` garante compatibilidade com armazenamento zarr.
- Se 0 keypoints: array vazio `(0, 32)` como placeholder (valor padrão; na prática, o tamanho da segunda dimensão depende do algoritmo usado: 32 para ORB, 128 para SIFT).

**Flag `registered`:**
- `True` se keypoints > 4 (tile tem features válidas).
- `False` se tile vazio ou mal-sucedido.

#### Passo 6: Limpeza de memória

```python
del detector, img, keypoints
gc.collect()
```

- Libera explicitamente objetos grandes em ambientes paralelos.
- Evita acúmulo de memória em workers.

#### Passo 7: Retornar resultado

```python
return tile_name, kp_array, descriptors, registered, tile["coordinates"]
```

- Sempre retorna 5 valores (garantia de desempacotamento correto em save_to_zarr).

---

## 4. Função `save_to_zarr(zarr_store, results)`

### 4.1 Propósito

Percorre resultados de todos os tiles processados e armazena em estrutura Zarr.

### 4.2 Lógica

```python
def save_to_zarr(zarr_store, results):
    for tile_name, kp_array, descriptors, registered, coords in results:
        if kp_array is None or descriptors is None:
            logger.info(f"[!] Tile '{tile_name}' ignorado (keypoints/descriptors None)")
            continue
```

- **Iteração segura:** impede salvamento de tiles sem features.
- **Log informativo:** registra o motivo do skip.

```python
        group = zarr_store.create_group(tile_name, overwrite=True)
        zarr.array(
            kp_array,
            store=group.store,
            path=f"{group.path}/keypoints",
            chunks=kp_array.shape,
            dtype=np.float32,
        )
        zarr.array(
            descriptors,
            store=group.store,
            path=f"{group.path}/descriptors",
            chunks=descriptors.shape,
            dtype=np.uint8,
        )
```

- Cria um **grupo Zarr** por tile (namespace isolado).
- Armazena:
  - `/tile_name/keypoints` → array float32 `(n, 7)`
  - `/tile_name/descriptors` → array uint8 `(n, D)` onde D depende do algoritmo: 32 para ORB, 128 para SIFT.
- **Chunking:** tamanho do chunk = tamanho do array inteiro (sem fragmentação).

```python
        group.attrs["registered"] = registered
        group.attrs["coordinates"] = coords
```

- Metadados por tile:
  - `registered`: booleano (tile teve features válidas?).
  - `coordinates`: tupla de coordenadas `(x, y)`.

### 4.3 Estrutura Zarr resultante

```
output/features/features.zarr/
├── 00001_x1_y1_zp1/
│   ├── keypoints    (array float32, shape: (N, 7))
│   ├── descriptors  (array uint8, shape: (N, D))  # D=32 para ORB, D=128 para SIFT
│   └── attrs: {registered, coordinates}
├── 00002_x2_y1_zp1/
│   └── ...
└── ...
```

---

## 5. Função `detect()`

### 5.1 Fluxo principal

#### Inicialização e logging

```python
start = perf_counter()

with open(Config.METADATA_FILE) as f:
    dataset_metadata = json.load(f)

with open(Config.VALID_TILES_FILE) as f:
    valid_tiles = json.load(f)

tiles = dataset_metadata
```

- Carrega dois JSONs:
  - `METADATA_FILE`: lista de tiles com nome, caminho, coordenadas (gerado por `fetch.py`).
  - `VALID_TILES_FILE`: mapeamento de tiles válidos (gerado por `classify.py`).
- `tiles` é a lista de dicts preparada para iteração.

#### Processamento paralelo ou serial

```python
if Config.DETECTION_N_JOBS == 1:
    results = [
        process_tile(Config.DETECTION_ALGORITHM, tile, valid_tiles)
        for tile in tiles
    ]
else:
    results = Parallel(n_jobs=Config.DETECTION_N_JOBS)(
        delayed(process_tile)(Config.DETECTION_ALGORITHM, tile, valid_tiles)
        for tile in tiles
    )
```

**Lógica:**
- Se `DETECTION_N_JOBS == 1`: execução **serial** (sem paralelização, mais simples para debug).
- Caso contrário: execução **paralela** com N workers.
- `delayed()` encapsula a função para execução diferida.
- `Parallel(n_jobs=N)` distribui tarefas entre processadores.

**Vantagens:**
- Serial: fácil debugging, sem overhead de pickle/IPC.
- Paralelo: aproveitamento de múltiplos cores; essencial para datasets grandes.

#### Salvamento em Zarr

```python
zarr_store = zarr.open(Config.KEYPOINTS_ZARR_STORE, mode="w")
save_to_zarr(zarr_store, results)
```

- Abre (ou cria) o armazém Zarr em modo escrita.
- Chama a função de persistência.

#### Logging final

```python
end = perf_counter()
logger.info(f"Features extraídas e salvas em {Config.KEYPOINTS_ZARR_STORE}")
logger.info(f"Tempo total: {end - start:.2f} segundos")
```

- Registra local de salvamento e tempo total de execução.

---

## 6. Entry point

```python
if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.INFO)
    detect()
```

- Configuração de logging padrão.
- Execução: `python -m src.modules.features.detect.detect`

---

## 7. Configurações esperadas (`Config`)

O arquivo depende de valores em `src.config.config.Config`:

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `METADATA_FILE` | str (path) | Caminho para JSON com metadados dos tiles. |
| `VALID_TILES_FILE` | str (path) | Caminho para JSON com tiles válidos. |
| `KEYPOINTS_ZARR_STORE` | str (path) | Caminho de saída do armazém Zarr (ex.: `output/features/features.zarr`). |
| `DETECTION_ALGORITHM` | str | Nome do algoritmo ("SIFT", "ORB", "AKAZE", etc.). |
| `DETECTION_N_JOBS` | int | Número de workers paralelos (1 = serial). |

---

## 8. Fluxo de dados resumido

```
metadata.json + valid_tiles.json
         ↓
    load files
         ↓
   for each tile in parallel
         ↓
    [process_tile]
    - get detector
    - load image
    - detect keypoints & descriptors
    - convert to numpy arrays
         ↓
   [save_to_zarr]
    - create zarr group per tile
    - store keypoints, descriptors, metadata
         ↓
   features.zarr (output/features/)
```

---

## 9. Casos de tratamento e robustez

### Tile inválido (não passou em `classify.py`)
- **Retorno:** `(tile_name, None, None, False, coords)`
- **Salvamento:** skipped em `save_to_zarr`
- **Benefício:** evita processamento desnecessário.

### Falha ao abrir imagem
- **Log:** erro registrado em stderr.
- **Retorno:** falha graceful com metadados.
- **Consequência:** tile não tem features no zarr final.

### Tile sem features significativas (< 5 keypoints)
- **Registro:** `registered = False`, mas arrays vazios são salvos.
- **Razão:** permitir que matching posterior saiba que o tile foi processado mas não tem features.
- **Trade-off:** economiza espaço vs. mantém rastreabilidade.

### Erro de memória em parallelização
- **Mitigation:** `gc.collect()` explícito após cada tile.
- **Melhorias possíveis:** limitar tamanho de batch de tiles.

---

## 10. Otimizações e pontos de atenção

### ✅ Boas práticas presentes

1. **Inicialização de detector dentro de `process_tile`**: evita problemas de pickle.
2. **Limpeza explícita de memória**: `del + gc.collect()`.
3. **Logging informativo**: facilita debugging.
4. **Parametrização via `Config`**: facilita reuso e testes.
5. **Estrutura Zarr modular**: acesso eficiente a subconjuntos de features.

### ⚠️ Possíveis melhorias

1. **Validação de formato de imagem**: atualmente só tenta abrir; poderia filtrar por extensão.
2. **Caching de detector**: se `algorithm` é sempre o mesmo, instanciar uma vez seria mais rápido.
3. **Controle de tamanho de batch em parallelização**: evita picos de memória.
4. **Validação de `valid_tiles` before processing**: falhar rápido se JSON está vazio.
5. **Checkpoint/restart**: se houver falha, retomar do último tile processado.

---

## 11. Exemplo de uso

### Setup mínimo

```python
# setup: config com valores
Config.DETECTION_ALGORITHM = "SIFT"
Config.DETECTION_N_JOBS = 4  # usar 4 processadores
Config.METADATA_FILE = "output/metadata/tiles_metadata.json"
Config.VALID_TILES_FILE = "output/metadata/valid_tiles.json"
Config.KEYPOINTS_ZARR_STORE = "output/features/features.zarr"
```

### Execução

```bash
python -m src.modules.features.detect.detect
```

### Consulta de resultados

```python
import zarr

store = zarr.open("output/features/features.zarr", mode="r")
for tile_name in store.keys():
    group = store[tile_name]
    keypoints = group["keypoints"][:]  # shape: (N, 7)
    descriptors = group["descriptors"][:]  # shape: (N, 32)
    print(f"{tile_name}: {len(keypoints)} keypoints, registered={group.attrs['registered']}")
```

---

## 12. Conexão com pipeline

### Inputs (dependências anteriores)
- `src.modules.tile.fetch.fetch` → gera `tiles_metadata.json`
- `src.modules.tile.classify.classifier` → gera `valid_tiles.json`

### Outputs (fornecidos para próximo passo)
- `features.zarr` → consumido por `src.modules.features.match.match`

### Próximo passo
- **Match de features:** calcula correspondências entre tiles vizinhos usando os descritores extraídos.

---

## Resumo

O `detect.py` é o núcleo da **extração de features** do pipeline:

- **Paraleliza** o processamento de tiles para eficiência.
- **Converte** keypoints OpenCV para arrays numéricos.
- **Persiste** em Zarr para acesso rápido e modular.
- **Rastreia** estado (registered) e metadados de cada tile.
- **Robusto** contra falhas isoladas (tiles inválidos, sem features).

Sua saída alimenta o matching, que por sua vez gera o grafo geométrico usado para calcular posições globais e montar o canvas final.

