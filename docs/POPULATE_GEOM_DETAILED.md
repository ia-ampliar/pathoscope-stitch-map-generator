**populate_geom.py — Documentação Detalhada

**Visão Geral**
- **Arquivo:** src/modules/canvas/populate_geom.py
- **Propósito:** Preencher o canvas geométrico (memmap) com os tiles normalizados, gerar um preview JPEG e um BigTIFF final. Fornece funções para abrir o memmap do canvas, mapear nós do grafo para caminhos de imagem, colar tiles (overwrite) e exportar saídas.
- **Entrada principal:** `Config.GLOBAL_POS_FILE` (posições globais) e tiles normalizados em `Config.NORMALIZED_DIR`.
- **Saídas principais:** preview JPEG (`Config.CANVAS_PREVIEW_PATH`) e BigTIFF (`Config.CANVAS_GEOM_PATH`).

**Funções e comportamento detalhado**

- `open_geom_canvas() -> np.memmap`
  - Abre o memmap criado por `create_geom.create_blank_canvas_geom` em modo `r+` para leitura e escrita.
  - Valida existência de `Config.CANVAS_GEOM_SHAPE_PATH` e `Config.BLANK_CANVAS_GEOM_PATH` e carrega `canvas_shape` via `np.load`.
  - Retorna um objeto `np.memmap` com `dtype=np.uint8` e `shape=canvas_shape`.
  - Erros possíveis: `FileNotFoundError` se shape ou memmap não existirem.

- `resolve_node_to_image_path(G_geo, positions: Positions, tiles_dir: Path) -> NodeToPath`
  - Para cada `node` presente em `positions`, recupera `label = G_geo.nodes[node]["label"]` e procura por `tiles_dir.glob(f"{label}.*")`.
  - Exige exatamente um arquivo por `label`. Lança `FileNotFoundError` se nenhum arquivo for encontrado e `ValueError` se houver ambiguidade (mais de um arquivo).
  - Também valida que o nó existe no grafo e que possui atributo `label`; caso contrário lança `KeyError`.
  - Retorna um dicionário `node_to_path` mapeando `node -> Path(caminho_do_tile)`.
  - Observação: assume que o stem (label) é único entre os arquivos normalizados.

- `paste_tiles_overwrite(canvas: np.memmap, positions: Positions, node_to_path: NodeToPath) -> None`
  - Para cada par `(node, (X, Y))` em `positions`:
    - Valida que o nó existe em `node_to_path`.
    - Arredonda `X, Y` para inteiros (pixel mais próximo) e lê o tile com `cv2.imread(..., cv2.IMREAD_COLOR)`.
    - Converte BGR->RGB e obtém shape do tile (`th, tw, tc`).
    - Calcula `x1 = min(x0 + tw, W)`, `y1 = min(y0 + th, H)` para respeitar bordas do canvas.
    - Se tile estiver totalmente fora do canvas (`x0 >= W or y0 >= H`), emite `logger.warning` e ignora.
    - Corta tile para área visível `tile_rgb[0:(y1-y0), 0:(x1-x0), :]` e escreve diretamente no memmap `canvas[y0:y1, x0:x1, :] = tile_crop` (overwrite).
  - Ao final, chama `canvas.flush()` para garantir gravação em disco.
  - Observações:
    - Operação é destrutiva (overwrite) — não faz blending entre tiles.
    - Coordenadas float são arredondadas; isso pode levar a pequenas inconsistências quando posições contêm frações.

- `export_preview_jpg(canvas: np.memmap, out_path: Path, quality: int = 95) -> None`
  - Converte o `memmap` (RGB) para um array em memória (`np.asarray(canvas)`), converte para BGR e salva com `cv2.imwrite` usando parâmetro de qualidade JPEG.
  - Valida retorno de `cv2.imwrite`; lança `RuntimeError` se houver falha.
  - Útil para checagem rápida do mosaico.

- `export_bigtiff_rgb(canvas: np.memmap, out_path: Path) -> None`
  - Usa `tifffile.imwrite` para exportar um BigTIFF RGB com `bigtiff=True` e `compression='jpeg'`.
  - Ideal para mosaicos muito grandes que precisam ser abertos por softwares GIS/imagem.

- `main()`
  - Fluxo típico da CLI:
    1. `open_geom_canvas()` — abre memmap.
    2. `load_positions_pickle(Config.GLOBAL_POS_FILE)` — carrega posições.
    3. `load_graph(Config.GEOMETRIC_GRAPH_FILE)` — carrega `G_geo`.
    4. `resolve_node_to_image_path(G_geo, positions, Config.NORMALIZED_DIR)` — mapeia nós para arquivos.
    5. `paste_tiles_overwrite(canvas, positions, node_to_path)` — cola tiles.
    6. `export_preview_jpg(canvas, Config.CANVAS_PREVIEW_PATH)`.
    7. `export_bigtiff_rgb(canvas, Config.CANVAS_GEOM_PATH)` (opcional, é executado no fluxo atual).
  - Logs informativos e mensagens de progresso são emitidas ao longo do processo.

**Entradas e Saídas (I/O)**
- Entradas:
  - `Config.BLANK_CANVAS_GEOM_PATH` e `Config.CANVAS_GEOM_SHAPE_PATH` (memmap e shape) criados por `create_geom.py`.
  - `Config.GLOBAL_POS_FILE` (posições globais saved by `globalpos.py`).
  - `Config.GEOMETRIC_GRAPH_FILE` (grafo geométrico, para atributo `label`).
  - Tiles normalizados em `Config.NORMALIZED_DIR` (usados para procurar imagens por label).
- Saídas:
  - `Config.CANVAS_PREVIEW_PATH` (JPEG preview).
  - `Config.CANVAS_GEOM_PATH` (BigTIFF final).

**Erros e validações**
- Lança `FileNotFoundError` se memmap/shape do canvas não existirem.
- `resolve_node_to_image_path` lança `FileNotFoundError`, `ValueError`, ou `KeyError` dependendo do problema de mapeamento.
- `paste_tiles_overwrite` lança `FileNotFoundError` se `cv2.imread` falhar para um tile.

**Limitações e considerações**
- Overwrite simples: não faz blending/feathering entre tiles; sobreposições são substituídas pela última escrita na ordem de iteração de `positions` (que é a ordem das chaves do dicionário). Se a ordem não for determinística, o resultado em zonas de sobreposição pode variar.
- Arredondamento de posições: `round(X), round(Y)` pode causar deslocamentos de ±1px. Se for importante, considerar usar interpolação ou armazenar offsets sub-pixel e aplicar blending.
- Uso de `np.asarray(canvas)` na exportação cria uma cópia/visão na memória que, para mosaicos muito grandes, pode exceder RAM ao gerar o preview; atualmente, o código assume que o preview é pequeno o suficiente.
- Exportar BigTIFF grava todo o mosaico em disco via `tifffile`; isso pode demorar e consumir espaço. Compressão JPEG é usada para reduzir tamanho.

**Melhorias sugeridas**
- Tornar a colagem compatível com blends (ex.: média ponderada) para zonas de sobreposição, com opção `mode='overwrite'|'blend'`.
- Ordenar `positions` por alguma heurística (por ex. raster scan) para garantir previsibilidade em sobreposições.
- Implementar streaming para export_preview_jpg (escrever em tiles e compor um preview reduzido em memória) quando o canvas exceder certa dimensão.
- Adicionar contador/progresso por tile (logging ou `tqdm`) para feedback em execuções longas.

**Exemplos de uso**
- Executar como script:

  python -m src.modules.canvas.populate_geom

- Em código:

  from src.modules.canvas.populate_geom import main
  main()

**Integração**
- Este módulo depende de `create_geom.py` (memmap) e `globalpos.py` (posições globais) e usa `graph.load_graph` para obter labels dos nós.
- O resultado (BigTIFF) é o produto final usado para inspeção ou entrada para etapas de processamento externo.

---
