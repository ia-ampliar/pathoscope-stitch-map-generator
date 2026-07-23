# Documentação Detalhada: `classifier.py`

**Arquivo:** `src/modules/tile/classify/classifier.py`

**Propósito:** Classificar tiles como válidos ou inválidos baseado em critérios de qualidade (ex.: quantidade de conteúdo vs. branco), gerando um dicionário de tiles válidos para uso posterior.

---

## 1. Visão geral

O módulo `classifier.py` é responsável pela **etapa de classificação e filtragem** no pipeline de stitching. Ele:

1. Processa cada tile em **paralelo** (via `joblib.Parallel`).
2. Aplica uma estratégia de classificação (via `ThresholdClassifier`) para determinar validade.
3. Gera **imagens de debug** mostrando resultado da classificação.
4. Persiste um dicionário booleano em JSON com tiles válidos/inválidos.

**Contexto no pipeline:**
- Entrada: tiles normalizados em `output/tiles/normalized/` (resultado de `preprocesser`).
- Saída: `valid_tiles.json` em `output/metadata/` + imagens de debug em `output/tmp/classified/`.
- Próxima etapa: detecção de features (apenas em tiles válidos).

---

## 2. Importações e dependências

```python
import gc                    # Limpeza de memória
import json                  # Serialização JSON
import logging               # Sistema de logging
import time                  # Medição de tempo
from pathlib import Path     # Manipulação segura de caminhos
from typing import Tuple     # Type hints

import cv2                   # OpenCV (leitura/escrita de imagens, processamento)
from joblib import Parallel, delayed  # Paralelização eficiente

from src.config.config import Config       # Configurações centralizadas
from .threshold import ThresholdClassifier # Estratégia de classificação
```

**Papel de cada import:**
- `gc`, `time`, `logging`: monitoramento e limpeza.
- `cv2`: leitura de imagens e escrita de debug.
- `joblib`: paralelização.
- `Config`: injeção de configuração (caminhos, parâmetros).
- `ThresholdClassifier`: encapsula lógica de classificação (strategy pattern).

---

## 3. Função `_process_tile(img_path: Path, classifier_cls=ThresholdClassifier) -> Tuple[str, bool]`

### 3.1 Assinatura e responsabilidades

```python
def _process_tile(
    img_path: Path, classifier_cls=ThresholdClassifier
) -> Tuple[str, bool]:
```

**Parâmetros:**
- `img_path` (Path): caminho absoluto do arquivo de imagem.
- `classifier_cls` (type): classe do classificador a instanciar (padrão: `ThresholdClassifier`). Permite injeção de dependência para testes.

**Retorna:** tupla `(tile_name, is_valid)`
- `tile_name` (str): nome base do arquivo sem extensão.
- `is_valid` (bool): `True` se tile passou na classificação, `False` caso contrário.

**Responsabilidade:** processar um único tile de forma isolada, determinar validade, salvar debug.

### 3.2 Passo a passo

#### Passo 1: Instanciar classificador

```python
classifier = classifier_cls()
```

- Cria nova instância do classificador.
- **Razão:** evitar problemas de estado compartilhado em ambientes paralelos (similar ao padrão em `detect.py`).
- Permite testes mockando a classe com uma implementação diferente.

#### Passo 2: Extrair nome e carregar imagem

```python
tile_name = img_path.stem
image = cv2.imread(str(img_path))
```

- `stem`: nome base sem extensão (ex.: "00001_x1_y1_zp1").
- `cv2.imread`: carrega imagem em BGR (padrão OpenCV).

#### Passo 3: Classificar

```python
is_valid, binary_debug = classifier.classify(image)
```

- Invoca método `classify()` do classificador.
- **Retorna:**
  - `is_valid`: booleano indicando se tile é válido.
  - `binary_debug`: array imagem mostrando o resultado da classificação (ex.: máscara binária).
- **Lógica interna** (em `ThresholdClassifier`): analisa níveis de branco, contraste, presença de conteúdo, etc.

#### Passo 4: Salvar imagem de debug

```python
debug_output_path = Config.CLASSIFIED_DIR / f"{tile_name}.jpg"
cv2.imwrite(str(debug_output_path), binary_debug)
```

- Escreve imagem de debug em `output/classified/`.
- Permite validação visual: usuário pode inspecionar quais tiles foram considerados válidos/inválidos e por quê.
- **Exemplo:** se tile tem muito branco, a máscara binária mostrará grande área branca.

#### Passo 5: Limpeza e retorno

```python
del image, binary_debug, classifier
gc.collect()

return tile_name, bool(is_valid)
```

- Libera memória explicitamente.
- `bool(is_valid)`: garante que o retorno é Python `bool` (não numpy bool ou outro tipo).

---

## 4. Função `run_classification(parallel: bool = True, n_jobs: int = -1)`

### 4.1 Assinatura e responsabilidades

```python
def run_classification(parallel: bool = True, n_jobs: int = -1):
```

**Parâmetros:**
- `parallel` (bool): se `True`, processa tiles em paralelo; se `False`, serial.
- `n_jobs` (int): número de workers para joblib. `-1` = usar todos os cores disponíveis.

**Responsabilidade:** orquestrar fluxo completo (criar pastas, processar tiles, salvar resultados).

### 4.2 Passo a passo

#### Passo 1: Inicialização

```python
start_time = time.perf_counter()

Config.CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
Config.VALID_TILES_FILE.parent.mkdir(parents=True, exist_ok=True)
```

- Inicia cronômetro.
- Cria diretórios de saída (com todos os intermediários).

#### Passo 2: Listar imagens

```python
image_paths = sorted(Config.NORMALIZED_DIR.glob("*.jpg"))
```

- Busca todos os `.jpg` em `NORMALIZED_DIR`.
- **Ordenação:** garante processamento determinístico (importante para reprodutibilidade e debug).

#### Passo 3: Processamento paralelo ou serial

```python
if parallel:
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_tile)(img_path) for img_path in image_paths
    )
else:
    results = [_process_tile(img_path) for img_path in image_paths]
```

**Lógica:**
- **Paralelo:** `joblib.Parallel` distribui `_process_tile` entre N workers.
  - `delayed()` encapsula chamada para execução diferida.
  - Eficiente para datasets grandes (aproveita múltiplos cores).
  - `-1` jobs: automático = número de cores disponíveis.
- **Serial:** loop simples, sem overhead de IPC/pickle.
  - Útil para debug ou ambientes com restrições.

**Resultado:** lista de tuplas `[(tile_name1, is_valid1), (tile_name2, is_valid2), ...]`.

#### Passo 4: Construir dicionário de tiles válidos

```python
valid_tiles = {tile_name: is_valid for tile_name, is_valid in results}
```

- Converte lista de tuplas em dicionário.
- **Formato:** `{nome_tile: bool}`.

**Exemplo:**
```python
{
    "00001_x1_y1_zp1": True,
    "00002_x2_y1_zp1": True,
    "00003_x3_y1_zp1": False,  # Tile com muito branco
    "00004_x4_y1_zp1": True,
    ...
}
```

#### Passo 5: Salvar JSON

```python
with open(Config.VALID_TILES_FILE, "w") as f:
    json.dump(valid_tiles, f, indent=2)
```

- Persiste dicionário em JSON (`output/metadata/valid_tiles.json`).
- `indent=2`: formatação legível para inspeção manual.

#### Passo 6: Logging final

```python
elapsed = time.perf_counter() - start_time
logger.info(
    f"Classificação de {len(image_paths)} imagens concluída em {elapsed:.2f} segundos."
)
logger.info(f"Resultados salvos em {Config.VALID_TILES_FILE}")
```

- Registra tempo total e localização do resultado.

**Exemplo de log:**
```
[INFO] - Classificação de 150 imagens concluída em 2.45 segundos.
[INFO] - Resultados salvos em output/metadata/valid_tiles.json
```

---

## 5. Função auxiliar: `ThresholdClassifier` (em `threshold.py`)

Embora não definida neste arquivo, é crítica para `classifier.py`.

### 5.1 Interface esperada

```python
class ThresholdClassifier:
    def classify(self, image) -> Tuple[bool, np.ndarray]:
        """
        Classifica uma imagem.
        
        Args:
            image: array BGR (OpenCV format)
        
        Returns:
            (is_valid, binary_debug)
            - is_valid: bool indicando se tile tem conteúdo significativo
            - binary_debug: array uint8 mostrando máscara de classificação
        """
```

### 5.2 Lógica típica

A implementação provavelmente:

1. **Converter para escala de cinza ou analisar canais:**
   - Detecta região de branco (alto brilho em todos os canais).

2. **Calcular percentual de conteúdo:**
   - Ex.: se > 80% da imagem é branco, `is_valid = False`.

3. **Gerar máscara de debug:**
   - Retorna imagem binária mostrando regiões de interesse.

4. **Aplicar thresholds configuráveis:**
   - Parâmetros em `Config` controlam sensibilidade.

### 5.3 Exemplo de retorno

```python
image = cv2.imread("00001_x1_y1_zp1.jpg")  # shape: (H, W, 3), dtype: uint8
is_valid, binary_debug = classifier.classify(image)
# is_valid = True (tile tem conteúdo)
# binary_debug.shape = (H, W), valores: 0 (branco) ou 255 (conteúdo)
```

---

## 6. Entry point do módulo

```python
if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.DEBUG)
    run_classification(parallel=True)
```

- Configuração de logging em nível DEBUG.
- Executa classificação em paralelo por padrão.
- Execução: `python -m src.modules.tile.classify.classifier`

---

## 7. Configurações esperadas (`Config`)

O arquivo depende dos seguintes valores em `src.config.config.Config`:

| Parâmetro | Tipo | Descrição | Exemplo |
|-----------|------|-----------|---------|
| `NORMALIZED_DIR` | Path | Diretório com tiles normalizados (`.jpg`). | `output/tiles/normalized` |
| `CLASSIFIED_DIR` | Path | Diretório para salvar imagens de debug. | `output/tmp/classified` |
| `VALID_TILES_FILE` | Path | Caminho do JSON com resultado da classificação. | `output/metadata/valid_tiles.json` |

**Nota:** `ThresholdClassifier` pode usar valores adicionais do Config para tuning (ex.: `WHITE_THRESHOLD`, `MIN_CONTENT_PERCENTAGE`).

---

## 8. Fluxo de dados resumido

```
output/tiles/normalized/ (N arquivos .jpg normalizados)
         ↓
   sorted().glob("*.jpg")
         ↓
   for each image_path in parallel (or serial)
         ↓
      [_process_tile]
      - instantiate ThresholdClassifier
      - load image (BGR)
      - classify(image) -> (is_valid, binary_debug)
      - save binary_debug to output/tmp/classified/
      - cleanup memory
      - return (tile_name, is_valid)
         ↓
   Collect all (tile_name, is_valid) tuples
         ↓
   Convert to dict: {tile_name: is_valid}
         ↓
   Save JSON
         ↓
   output/metadata/valid_tiles.json
   +
   output/classified/ (imagens de debug)
```

---

## 9. Casos de tratamento e robustez

### Arquivo de imagem não pode ser aberto
- **Comportamento atual:** `cv2.imread()` retorna `None` → passado para `classifier.classify()`.
- **Consequência:** `classify()` provavelmente levanta exceção ou retorna falha.
- **Recomendação:** adicionar verificação `if image is None` antes de chamar `classify()`.

### Classificador retorna tipo inesperado
- **Proteção:** `bool(is_valid)` força conversão para Python bool.
- **Benefício:** robustos contra numpy bool ou outros tipos.

### Diretório de tiles vazio
- **Comportamento:** `glob()` retorna lista vazia → nenhum resultado → `valid_tiles` vazio → JSON vazio salvo.
- **Impacto:** não é erro (fail-soft), mas próximas etapas podem falhar se esperam pelo menos 1 tile válido.

### Falha de escrita em CLASSIFIED_DIR
- **Comportamento atual:** `cv2.imwrite()` pode falhar silenciosamente ou levantar exceção.
- **Recomendação:** adicionar try-except para tratamento explícito.

### Paralelização com poucos tiles
- **Overhead:** para poucas imagens, paralelização pode ser mais lenta que serial (overhead de IPC).
- **Solução:** usar `parallel=False` para datasets pequenos.

---

## 10. Otimizações e pontos de atenção

### ✅ Boas práticas presentes

1. **Paralelização com joblib:** eficiente para datasets grandes.
2. **Limpeza de memória explícita:** `del + gc.collect()` em cada worker.
3. **Logging informativo:** tempo total, local de salvamento.
4. **Injeção de dependência:** `classifier_cls` parametrizável para testes.
5. **Ordenação determinística:** `sorted()` garante reprodutibilidade.
6. **Imagens de debug:** visual feedback para validação.

### ⚠️ Possíveis melhorias

1. **Validação de imagem antes de classificar:** verificar se `cv2.imread()` sucedeu.
2. **Tratamento robusto de exceções:** capturar erros em `_process_tile` sem interromper parallelização.
3. **Estatísticas resumidas:** contar quantos tiles foram válidos vs. inválidos.
4. **Configurabilidade de classificador:** permitir trocar `ThresholdClassifier` por outro via Config.
5. **Filtro de extensão extensível:** atualmente hardcoded `*.jpg`; poderia usar `Config.SUPPORTED_EXTENSIONS`.
6. **Cache/incremental:** se dataset crescer, reclassificar tudo é ineficiente.
7. **Timeout para workers:** prevenir travamento em imagens problemáticas.

---

## 11. Exemplo de uso

### Setup mínimo

```python
from pathlib import Path
from src.config.config import Config

# Configurar
Config.NORMALIZED_DIR = Path("output/tiles")
Config.CLASSIFIED_DIR = Path("output/classified")
Config.VALID_TILES_FILE = Path("output/metadata/valid_tiles.json")
```

### Execução

```bash
# Paralelo (padrão)
python -m src.modules.tile.classify.classifier

# Serial (debug)
# Modificar o arquivo ou chamar função diretamente:
# run_classification(parallel=False)
```

### Consulta de resultados

```python
import json
from pathlib import Path

valid_tiles_file = Path("output/metadata/valid_tiles.json")
with open(valid_tiles_file) as f:
    valid_tiles = json.load(f)

total = len(valid_tiles)
valid_count = sum(1 for v in valid_tiles.values() if v)
invalid_count = total - valid_count

print(f"Total: {total}")
print(f"Válidos: {valid_count} ({100*valid_count/total:.1f}%)")
print(f"Inválidos: {invalid_count} ({100*invalid_count/total:.1f}%)")

# Primeiros inválidos
print("\nPrimeiros inválidos:")
for name, is_valid in valid_tiles.items():
    if not is_valid:
        print(f"  {name}")
```

**Exemplo de saída:**
```
Total: 150
Válidos: 145 (96.7%)
Inválidos: 5 (3.3%)

Primeiros inválidos:
  00025_x2_y2_zp1
  00087_x7_y4_zp1
  00142_x11_y5_zp1
```

### Inspecionar imagem de debug

```bash
# Abrir em visualizador de imagens
output/classified/00001_x1_y1_zp1.jpg
```

---

## 12. Formato do JSON gerado

### Estrutura completa

```json
{
  "00001_x1_y1_zp1": true,
  "00002_x2_y1_zp1": true,
  "00003_x3_y1_zp1": false,
  "00004_x4_y1_zp1": true,
  ...
}
```

### Notas

- Dicionário simples: chave = nome tile (sem extensão), valor = booleano.
- JSON preserva booleans como `true`/`false` (lowercase JSON).
- Ordenação: mesma ordem de processamento (determinística se `sorted()` for respeitado).

---

## 13. Conexão com pipeline

### Inputs (dependências anteriores)
- `src.modules.tile.preprocessing.preprocesser` → normaliza e armazena tiles em `.jpg` em `output/tiles/`.

### Outputs (fornecidos para próximo passo)
- `output/metadata/valid_tiles.json` → consumido por:
  - `src.modules.features.detect.detect` (filtra tiles para detecção).
  - `src.modules.tile.fetch.fetch` (complementa metadados com informação de validade).
- `output/classified/` → imagens de debug para validação visual do usuário.

### Próximo passo
- **Detecção de features:** extrair keypoints apenas dos tiles válidos.

---

## 14. Padrão de design

O código utiliza dois padrões importantes:

### Strategy Pattern

```python
def _process_tile(img_path: Path, classifier_cls=ThresholdClassifier) -> Tuple[str, bool]:
    classifier = classifier_cls()
    # ...
```

- `classifier_cls` parametrizável permite trocar estratégia de classificação.
- Facilita testes (mock classifier) e extensões futuras (outros classificadores).

### Factory Pattern (implícito)

```python
classifier = classifier_cls()
```

- Instancia classificador sem saber detalhes internos.
- Encapsula criação.

---

## 15. Performance e escalabilidade

### Tempo esperado

Para N tiles:
- **Serial:** O(N) (tempo linear, 1 core).
- **Paralelo:** O(N/C) onde C = número de cores (ideal).
- **Overhead:** + setup/teardown de workers (~1-2s para paralelização).

**Exemplo:** 150 tiles em máquina com 8 cores
- Serial: ~15-20s (100ms/tile).
- Paralelo: ~3-5s (aproveita 8 cores).

### Memória

- Por worker: O(H × W × 3) para imagem + overhead do classificador.
- Total: O(C × H × W × 3) onde C = cores em uso.
- **Otimização:** `gc.collect()` mitiga acúmulo.

---

## 16. Resumo

O `classifier.py` é o **filtro de qualidade** do pipeline:

- **Valida** tiles antes de investir em detecção de features (etapa cara).
- **Paraleliza** eficientemente para datasets grandes.
- **Oferece visual feedback** (imagens de debug).
- **Simples** e focado: uma responsabilidade bem definida.
- **Extensível:** permite trocar estratégia de classificação via injeção.

Sua saída é essencial: sem conhecer quais tiles são válidos, etapas posteriores processariam tiles inúteis (branco puro, defeituosos, etc.), desperdiçando tempo e recursos. O JSON `valid_tiles.json` funciona como "filtro de entrada" para todo o restante do pipeline.

