**create_geom.py — Documentação Detalhada

**Visão Geral**
- **Arquivo:** src/modules/canvas/create_geom.py
- **Propósito:** Criar e preparar um canvas em branco (memmap) que acomodará o mosaico geométrico baseado nas posições globais dos tiles. Fornece utilitários para calcular o shape do canvas a partir das posições, criar um memmap em disco preenchido com valor de fundo (branco) em chunks, e salvar o shape para uso posterior.
- **Saídas principais:** um arquivo memmap (Config.BLANK_CANVAS_GEOM_PATH) contendo o canvas branco e um arquivo com o shape do canvas salvo em Config.CANVAS_GEOM_SHAPE_PATH.

**Contexto**
- `create_geom.py` é usado depois que o pipeline já estimou posições globais (arquivo Config.GLOBAL_POS_FILE). Essas posições representam as coordenadas (X,Y) do canto superior esquerdo de cada tile no canvas final.
- O canvas é tipicamente grande; por isso é criado como um `numpy.memmap` e preenchido em blocos (chunks) para evitar consumo excessivo de RAM.

**Funções principais e comportamento**

- `compute_canvas_shape_from_positions(positions: Positions, tile_w: int, tile_h: int, channels: int = 3) -> Tuple[Tuple[int,int,int], Tuple[float,float,float,float]]`
  - Calcula a bounding box do conjunto de posições: minX, minY, maxX, maxY.
  - Computa o tamanho do canvas como: canvas_w = ceil((maxX - minX) + tile_w), canvas_h = ceil((maxY - minY) + tile_h).
  - Retorna `canvas_shape = (H, W, C)` e o `bbox` (min_x, min_y, max_x, max_y).
  - Observação: soma `tile_w` e `tile_h` para garantir cobertura completa dos tiles posicionados nos cantos.

- `create_blank_canvas_geom(force_recompute: bool = False) -> Tuple[Path, Tuple[int,int,int]]`
  - Fluxo resumido:
    1. Verifica cache: se `Config.BLANK_CANVAS_GEOM_PATH` e `Config.CANVAS_GEOM_SHAPE_PATH` existem e `force_recompute=False`, carrega shape e retorna caminho do memmap reutilizando-o.
    2. Carrega `positions` com `load_positions_pickle(Config.GLOBAL_POS_FILE)` (posições globais previamente geradas).
    3. Seleciona um tile de amostra via `list_tile_images(Config.TILES_DIR)` e infere `tile_h, tile_w, channels` com `infer_tile_shape()`.
    4. Calcula `canvas_shape` chamando `compute_canvas_shape_from_positions()`.
    5. Cria diretório `Config.CANVAS_OUTPUT_PATH` e, se `force_recompute`, remove memmap antigo.
    6. Cria `np.memmap` com `dtype=np.uint8`, `mode='w+'`, shape=`canvas_shape` no caminho `Config.BLANK_CANVAS_GEOM_PATH`.
    7. Preenche o memmap em blocos (iterando em `chunk_h`, `chunk_w`) com `fill_value` (por padrão branco: 255), usando `Config.CANVAS_CHUNK_SIZE` para chunking.
    8. Faz `flush()` no memmap e salva `canvas_shape` em `Config.CANVAS_GEOM_SHAPE_PATH` via `np.save`.
  - Retorna `(memmap_path, canvas_shape)`.
  - Observações de segurança:
    - Valida que `channels == 3` (o pipeline espera imagens RGB).
    - Valida `Config.CANVAS_CHUNK_SIZE` para ter 3 canais no tuplo.

- Funções auxiliares e bloco `main()`:
  - Há uma função `create()` no arquivo que parece uma versão alternativa (não utilizada no fluxo principal) que faz uso de `dask` para criar um `darr` e depois escrever para um memmap. Essa função depende de imports (`da`) que não aparecem no topo do arquivo — sugere que é código legado/exemplificativo e não é chamado pelo fluxo padrão.
  - `main()` chama `create_blank_canvas_geom(force_recompute=True)` e loga o resultado.

**I/O e arquivos esperados**
- Entradas:
  - `Config.GLOBAL_POS_FILE` — pickle com `Positions` (obtido de `globalpos.py`).
  - Arquivos de tile em `Config.TILES_DIR` — usados apenas para inferir tamanho do tile (shape).
- Saídas:
  - `Config.BLANK_CANVAS_GEOM_PATH` — arquivo memmap contendo o canvas em branco (uint8 RGB).
  - `Config.CANVAS_GEOM_SHAPE_PATH` — arquivo `.npy` com shape do canvas salvo.

**Parâmetros de configuração relevantes (em `Config`)**
- `Config.GLOBAL_POS_FILE` — caminho para arquivo de posições globais.
- `Config.TILES_DIR` — diretório com tiles de amostra.
- `Config.BLANK_CANVAS_GEOM_PATH` — caminho destino do memmap.
- `Config.CANVAS_GEOM_SHAPE_PATH` — caminho onde o shape é salvo.
- `Config.CANVAS_OUTPUT_PATH` — pasta de saída do canvas.
- `Config.CANVAS_CHUNK_SIZE` — tupla (chunk_h, chunk_w, chunk_c) usada para preencher em blocos.
- `Config.CANVAS_FILL_VALUE` (opcional) — valor de preenchimento do canvas (por ex. 255 para branco).

**Comportamento em casos de erro e validações**
- Lança `FileNotFoundError` se não houver tiles em `Config.TILES_DIR`.
- Lança `ValueError` se o tile de amostra não for RGB (channels != 3).
- Lança `ValueError` se `Config.CANVAS_CHUNK_SIZE` não tiver 3 elementos com canal igual a 3.
- Se `Config.GLOBAL_POS_FILE` não existir, `load_positions_pickle` (em src/utils/io) provavelmente levanta erro; o código assume que as posições foram geradas antes.

**Desempenho e escalabilidade**
- O uso de `np.memmap` permite criar canvases maiores que a memória RAM disponível.
- O preenchimento em chunks evita alocar o canvas inteiro em memória, escrevendo pedaços iterativamente.
- Recomendações:
  - Ajustar `Config.CANVAS_CHUNK_SIZE` para equilibrar I/O e memória.
  - Em sistemas com SSD rápido, chunks maiores reduzem overhead; em HDD, chunks menores reduzem latência de escrita por bloco.
  - Evitar recomputar o memmap desnecessariamente usando `force_recompute=False` quando possível.

**Limpezas e notas de manutenção**
- ⚠️ **Código legado:** A função `create()` usa `dask` e `da.full` mas não importa `dask.array as da` no topo do arquivo. Isso significa que chamar `create()` geraria um `NameError`. Esta função é **código morto/legado** e **não é utilizada pelo pipeline atual** (o fluxo principal usa `main()` → `create_blank_canvas_geom()`). Mantida no arquivo apenas como referência histórica.
- O arquivo atual faz o trabalho principal diretamente em NumPy (`np.memmap`) — abordagem simples e portátil.
- Verificar se `load_positions_pickle` valida o tipo de dados carregado para evitar problemas de casting.

**Exemplos de uso**
- Criar (recriar) o canvas geométrico a partir do zero (linha de comando):

  python -m src.modules.canvas.create_geom

- Em código Python:

  from src.modules.canvas.create_geom import create_blank_canvas_geom
  memmap_path, canvas_shape = create_blank_canvas_geom(force_recompute=True)

- Após a criação, o memmap pode ser aberto para escrita/colocação dos tiles pelos módulos de `populate_geom`/`canvas`.

**Conclusão**
- `create_geom.py` é o componente responsável por materializar no disco o canvas branco que servirá como base para compor o mosaico geométrico, dimensionado a partir das posições globais estimadas.
- O design prioriza escalabilidade por uso de `np.memmap` e escrita em chunks.
- A função `create()` presente no arquivo é **código legado não funcional** (import de `dask.array` ausente) e não deve ser utilizada. O ponto de entrada correto é `main()` → `create_blank_canvas_geom()`.

---