# Documentação Detalhada: `fetch.py`

**Arquivo:** `src/modules/tile/fetch/fetch.py`

**Propósito:** Extrair metadados (nome, caminho, coordenadas) dos tiles normalizados e organizá-los em um arquivo JSON centralizado.

---

## 1. Visão geral

O módulo `fetch.py` é responsável pela **etapa de extração de metadados** no pipeline de stitching. Ele:

1. Lista todos os arquivos de imagem em um diretório (tipicamente pós-normalização).
2. Extrai **coordenadas** do nome de arquivo usando regex configurável.
3. Organiza informações em dicts estruturados.
4. Persiste tudo em um JSON para consumo posterior.

**Contexto no pipeline:**
- Entrada: tiles normalizados em `output/tiles/normalized/` (resultado de `preprocesser`).
- Saída: `dataset.json` em `output/metadata/`.
- Próxima etapa: classificação de tiles válidos (`classifier`).

---

## 2. Importações e dependências

```python
import json                    # Serialização/desserialização JSON
import logging                 # Sistema de logging
import time                    # Medição de tempo
from pathlib import Path       # Manipulação segura de caminhos
from typing import List        # Type hints

from src.config.config import Config                           # Configurações centralizadas
from src.utils.coordinates import extract_coordinates          # Função helper para parsing
```

**Papel de cada import:**
- `json`, `logging`, `time`, `Path`: utilidades padrão.
- `List` (typing): anotações para melhor IDE support e linting.
- `Config`: injeção de configurações (extensões, padrões, caminhos).
- `extract_coordinates`: encapsula lógica de regex para extrair coordenadas do nome do arquivo.

---

## 3. Função `list_image_files(directory: Path, extensions: List[str]) -> List[str]`

### 3.1 Assinatura e responsabilidades

```python
def list_image_files(directory: Path, extensions: List[str]) -> List[str]:
```

**Parâmetros:**
- `directory` (Path): caminho do diretório a escanear.
- `extensions` (List[str]): lista de extensões aceitas (ex.: `[".png", ".jpg", ".tiff"]`).

**Retorna:** lista de nomes de arquivo (strings simples, sem caminho).

**Responsabilidade:** enumerar arquivos válidos com extensão suportada.

### 3.2 Implementação

```python
if not directory.exists():
    raise FileNotFoundError(f"Diretório não encontrado: {directory}")
```

- Valida existência do diretório antes de processar.
- Lança erro explícito se não encontra (fail-fast).

```python
return [f.name for f in directory.iterdir() if f.suffix.lower() in extensions]
```

- `directory.iterdir()`: itera sobre todos os arquivos/pastas.
- `f.suffix.lower()`: extrai extensão e normaliza para minúsculas (`.PNG` → `.png`).
- List comprehension: filtra apenas arquivos com extensão válida.

**Exemplo:**
```
Input: directory = Path("output/tiles"), extensions = [".png", ".jpg"]
Output: ["00001.png", "00002.jpg", "00003.png"]
```

**Observação:** retorna apenas nomes, não caminhos completos (economia de memória para datasets grandes).

---

## 4. Função `generate_metadata(directory: Path) -> List[dict]`

### 4.1 Assinatura e responsabilidades

```python
def generate_metadata(directory: Path) -> List[dict]:
```

**Parâmetros:**
- `directory` (Path): diretório contendo tiles.

**Retorna:** lista de dicts, cada um com campos: `name`, `path`, `coordinates`.

**Responsabilidade:** orquestrar extração de metadados para todos os tiles.

### 4.2 Passo a passo

#### Passo 1: Listar imagens

```python
images = list_image_files(directory, Config.SUPPORTED_EXTENSIONS)
metadata = []
```

- Invoca `list_image_files` com extensões do Config.
- Inicializa lista vazia para acumular resultados.

#### Passo 2: Iterar e extrair coordenadas

```python
for image in images:
    try:
        coordinates = extract_coordinates(image, Config.COORDINATES_PATTERN)
        metadata.append(
            {"name": image, "path": str(directory), "coordinates": coordinates}
        )
    except ValueError as e:
        logging.error(f"Ignorando arquivo inválido: {image} - {e}")
```

**Lógica por arquivo:**

1. **Extração de coordenadas:**
   - `extract_coordinates(image, pattern)` analisa o nome do arquivo com regex.
   - Espera-se que o padrão capture 2 grupos (ex.: "00001_x1_y1_zp1.png" → (1, 1)).
   - Levanta `ValueError` se o nome não corresponde ao padrão.

2. **Acúmulo de metadados:**
   - `name`: nome do arquivo (ex.: "00001_x1_y1_zp1.png").
   - `path`: caminho absoluto como string (ex.: "/output/tiles/normalized").
   - `coordinates`: tupla extraída (ex.: (1, 1)).

3. **Tratamento de erros:**
   - Se parsing falha (nome inválido), loga erro e continua.
   - Tiles mal-nomeados são silenciosamente ignorados (não interrompe execução).

**Exemplo de output:**
```python
[
    {
        "name": "00001_x1_y1_zp1.png",
        "path": "/home/user/output/tiles/normalized",
        "coordinates": [1, 1]
    },
    {
        "name": "00002_x2_y1_zp1.png",
        "path": "/home/user/output/tiles/normalized",
        "coordinates": [2, 1]
    },
    ...
]
```

#### Passo 3: Retornar lista

```python
return metadata
```

- Retorna lista completa (pode estar vazia se nenhum arquivo for válido).

---

## 5. Função `save_metadata(metadata: List[dict], output_path: Path) -> None`

### 5.1 Assinatura e responsabilidades

```python
def save_metadata(metadata: List[dict], output_path: Path) -> None:
```

**Parâmetros:**
- `metadata`: lista de dicts com metadados.
- `output_path`: caminho do arquivo JSON de saída.

**Responsabilidade:** persistir metadados em JSON com formatação legível.

### 5.2 Implementação

```python
output_path.parent.mkdir(parents=True, exist_ok=True)
```

- Cria diretório pai (e todos os intermediários) se não existem.
- `exist_ok=True`: não falha se já existe.

```python
with open(output_path, "w") as f:
    json.dump(metadata, f, indent=2)
```

- Abre arquivo em modo escrita.
- `json.dump` serializa lista para JSON com indentação (legibilidade).
- File é fechado automaticamente ao sair do `with`.

**Exemplo de saída (output/metadata/dataset.json):**
```json
[
  {
    "name": "00001_x1_y1_zp1.png",
    "path": "/home/user/output/tiles/normalized",
    "coordinates": [1, 1]
  },
  {
    "name": "00002_x2_y1_zp1.png",
    "path": "/home/user/output/tiles/normalized",
    "coordinates": [2, 1]
  }
]
```

---

## 6. Função `extract_and_save_metadata(tiles_dir: Path, output_path: Path) -> int`

### 6.1 Assinatura e responsabilidades

```python
def extract_and_save_metadata(tiles_dir: Path, output_path: Path) -> int:
```

**Parâmetros:**
- `tiles_dir`: diretório de tiles.
- `output_path`: caminho para salvar JSON.

**Retorna:** número de tiles processados com sucesso.

**Responsabilidade:** orquestrar fluxo completo (extração + validação + salvamento).

### 6.2 Implementação

```python
metadata = generate_metadata(tiles_dir)
if not metadata:
    raise ValueError(f"Nenhuma imagem válida encontrada em {tiles_dir}")
save_metadata(metadata, output_path)
return len(metadata)
```

**Lógica:**

1. Invoca `generate_metadata` para extrair dados.
2. Valida se há pelo menos um tile válido (fail-fast se diretório vazio ou padrão inválido).
3. Persiste dados.
4. Retorna contagem para logging.

**Benefício:** centraliza toda a lógica em um único ponto, facilitando testes e reuso.

---

## 7. Função `main() -> None`

### 7.1 Entry point principal

```python
def main() -> None:
    start_time = time.perf_counter()
    logger.info("Iniciando extração dos metadados...")

    try:
        count = extract_and_save_metadata(Config.NORMALIZED_DIR, Config.METADATA_FILE)
        elapsed = time.perf_counter() - start_time
        logger.info(f"{count} imagens processadas em {elapsed:.4f} segundos.")
        logger.info(f"Metadados salvos em {Config.METADATA_FILE}")
    except Exception as e:
        logger.error(f"Falha na execução: {e}")
```

**Responsabilidades:**

1. **Medição de tempo:**
   - `perf_counter()` para precisão (não afetado por ajustes de relógio do SO).

2. **Logging informativo:**
   - Início, sucesso, tempo elapsed, localização do output.

3. **Tratamento robusto de exceções:**
   - Qualquer erro em `extract_and_save_metadata` é capturado e logado.
   - Impede crash e oferece mensagem clara.

**Exemplo de log (sucesso):**
```
[INFO] - Iniciando extração dos metadados...
[INFO] - 150 imagens processadas em 0.2341 segundos.
[INFO] - Metadados salvos em output/metadata/tiles_metadata.json
```

**Exemplo de log (falha):**
```
[ERROR] - Falha na execução: Nenhuma imagem válida encontrada em output/tiles/normalized
```

---

## 8. Entry point do módulo

```python
if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.DEBUG)
    main()
```

- Configuração de logging (nível DEBUG para verbosidade).
- Execução: `python -m src.modules.tile.fetch.fetch`

---

## 9. Configurações esperadas (`Config`)

O arquivo depende dos seguintes valores em `src.config.config.Config`:

| Parâmetro | Tipo | Descrição | Exemplo |
|-----------|------|-----------|---------|
| `NORMALIZED_DIR` | Path | Diretório contendo tiles normalizados. | `output/tiles/normalized` |
| `METADATA_FILE` | Path | Caminho do arquivo JSON de saída. | `output/metadata/dataset.json` |
| `SUPPORTED_EXTENSIONS` | List[str] | Extensões de imagem suportadas. | `[".png", ".jpg", ".jpeg", ".tif", ".tiff"]` |
| `COORDINATES_PATTERN` | str (regex) | Padrão para extrair coordenadas (x, y) do nome. | `r".*_x(\d+)_y(\d+)_.*"` |

---

## 10. Função auxiliar: `extract_coordinates` (no módulo `utils/coordinates.py`)

Embora não definida em `fetch.py`, é crítica para sua operação.

### 10.1 Comportamento esperado

```python
# Exemplo
extract_coordinates("00001_x1_y1_zp1.png", r".*_x(\d+)_y(\d+)_.*")
# Retorna: (1, 1)
```

- Recebe nome de arquivo e padrão regex.
- Usa `re.search` para buscar matches.
- Extrai os dois grupos capturados (x e y).
- Retorna como `Tuple[int, int]` ou levanta `ValueError` se nenhum match.

### 10.2 Tratamento de erros

Se o padrão não corresponder:
```python
raise ValueError(f"Formato inválido: '{filename}' não contém coordenadas com o padrão '{pattern}'.")
```

Exemplo: arquivo chamado "image.png" (sem coordenadas) → ValueError → ignorado em `generate_metadata`.

---

## 11. Fluxo de dados resumido

```
output/tiles/normalized/ (diretório com N arquivos de imagem)
         ↓
   list_image_files()
   → filtra por extensão suportada
         ↓
   generate_metadata()
   → for cada arquivo:
      - extract_coordinates() via regex
      - assembla dict {name, path, coordinates}
   → filtra arquivos inválidos (erro no parsing)
         ↓
   extract_and_save_metadata()
   → valida se há pelo menos 1 tile
   → chama save_metadata()
         ↓
   save_metadata()
   → cria diretórios necessários
   → serializa JSON com indent=2
         ↓
   output/metadata/dataset.json
```

---

## 12. Casos de tratamento e robustez

### Arquivo com extensão não suportada
- **Ação:** ignorado por `list_image_files`.
- **Benefício:** permite ter múltiplos tipos em diretório; apenas os suportados são processados.

### Arquivo com nome que não corresponde ao padrão
- **Ação:** exception em `extract_coordinates` → capturada em `generate_metadata` → loggado erro → arquivo ignorado.
- **Benefício:** extração robusta; não interrompe pipeline por um arquivo mal-nomeado.

### Diretório de tiles vazio
- **Ação:** `generate_metadata` retorna lista vazia → `extract_and_save_metadata` levanta `ValueError` → capturada em `main()` → loggado erro.
- **Benefício:** fail-fast; feedback claro ao usuário.

### Diretório de tiles não existe
- **Ação:** `list_image_files` levanta `FileNotFoundError` → propagado para `main()` → capturado e loggado.
- **Benefício:** erro claro e acionável.

### Falha ao criar diretório de output
- **Ação:** `mkdir` falha → exceção propagada para `main()` → capturada e loggado.
- **Benefício:** avisa se não tem permissão ou caminho é inválido.

---

## 13. Otimizações e pontos de atenção

### ✅ Boas práticas presentes

1. **Type hints:** facilita detecção de erros e IDE support.
2. **Logging estruturado:** debug fácil com níveis apropriados.
3. **Fail-fast:** valida pré-condições (diretório existe, há tiles válidos).
4. **Separação de responsabilidades:** cada função faz uma coisa bem.
5. **Tratamento robusto de erros:** não interrompe por tiles individuais inválidos.
6. **Configuração centralizada:** fácil adaptar padrões, extensões, caminhos.

### ⚠️ Possíveis melhorias

1. **Logging de warnings para arquivos ignorados:** atualmente apenas DEBUG/ERROR; seria útil saber quantos foram ignorados.
2. **Deduplicação de coordenadas:** se houver dois tiles com mesma coordenada, não há validação (potencial problema para grafo topológico).
3. **Esquema de versão para JSON:** futuras mudanças de formato quebram compatibilidade.
4. **Cache/incremental processing:** se diretório crescer, reler tudo é ineficiente.
5. **Validação de intervalos de coordenadas:** não verifica se coordenadas fazem sentido (ex.: negativas, muito grandes).

---

## 14. Exemplo de uso

### Setup mínimo

```python
from pathlib import Path
from src.config.config import Config

# Configurar (ou verificar se já está em Config)
Config.NORMALIZED_DIR = Path("output/tiles/normalized")
Config.METADATA_FILE = Path("output/metadata/dataset.json")
Config.SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
Config.COORDINATES_PATTERN = r".*_x(\d+)_y(\d+)_.*"
```

### Execução

```bash
python -m src.modules.tile.fetch.fetch
```

### Consulta de resultados

```python
import json
from pathlib import Path

metadata_file = Path("output/metadata/tiles_metadata.json")
with open(metadata_file) as f:
    metadata = json.load(f)

print(f"Total de tiles: {len(metadata)}")
for tile in metadata[:5]:  # Primeiros 5
    print(f"  {tile['name']} @ {tile['coordinates']}")
```

**Exemplo de saída:**
```
Total de tiles: 150
  00001_x1_y1_zp1.png @ [1, 1]
  00002_x2_y1_zp1.png @ [2, 1]
  00003_x3_y1_zp1.png @ [3, 1]
  00004_x4_y1_zp1.png @ [4, 1]
  00005_x5_y1_zp1.png @ [5, 1]
```

---

## 15. Conexão com pipeline

### Inputs (dependências anteriores)
- `src.modules.initializer.initialize_structure` → cria `output/` e pastas.
- `src.modules.tile.preprocessing.preprocesser` → normaliza e armazena tiles em `output/tiles/normalized/`.

### Outputs (fornecidos para próximo passo)
- `output/metadata/dataset.json` → consumido por `src.modules.tile.classify.classifier` e indiretamente por `src.modules.features.detect.detect`.

### Próximo passo
- **Classificação:** valida tiles e determina quais serão usados no matching.

---

## 16. Formato do JSON gerado

### Estrutura completa

```json
[
  {
    "name": "00001_x1_y1_zp1.png",
    "path": "/absolute/path/to/output/tiles/normalized",
    "coordinates": [1, 1]
  },
  {
    "name": "00002_x2_y1_zp1.png",
    "path": "/absolute/path/to/output/tiles/normalized",
    "coordinates": [2, 1]
  },
  ...
]
```

### Notas

- Array ordenado (mesma ordem de `iterdir()`; geralmente alfabética no POSIX, pode variar no Windows).
- `coordinates` é lista de 2 elementos `[x, y]` (JSON não tem tuplas).
- `path` é string do caminho absoluto ou relativo (conforme `Config.NORMALIZED_DIR`).

---

## Resumo

O `fetch.py` é o **extrator de metadados** do pipeline:

- **Enumera** tiles por extensão suportada.
- **Extrai** coordenadas de nome de arquivo via regex configurável.
- **Organiza** em estrutura JSON padronizada.
- **Robusto** contra files inválidos (ignora, não interrompe).
- **Simples** e focado: uma responsabilidade bem definida.

Sua saída é essencial para todas as etapas subsequentes: classificação, matching, grafo topológico, etc. Sem metadados corretos e completos, o restante do pipeline não pode funcionar adequadamente.

