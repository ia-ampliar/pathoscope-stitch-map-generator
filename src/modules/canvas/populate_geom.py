import time
import logging
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime

import cv2
import numpy as np
import tifffile 

from src.config.config import Config
from src.modules.graph.graph import load_graph
from src.utils.io import load_positions_pickle

logger = logging.getLogger(__name__)

Node = Tuple[int, int]
PosXY = Tuple[float, float]
Positions = Dict[Node, PosXY]
NodeToPath = Dict[Node, Path]


def open_geom_canvas() -> np.memmap:
    """
    Abre o canvas geométrico já criado pelo create_geom.py como np.memmap.
    """
    shape_path = Config.CANVAS_GEOM_SHAPE_PATH
    memmap_path = Config.BLANK_CANVAS_GEOM_PATH

    if not shape_path.exists():
        raise FileNotFoundError(f"Shape do canvas geom não encontrado: {shape_path}")
    if not memmap_path.exists():
        raise FileNotFoundError(f"Memmap do canvas geom não encontrado: {memmap_path}")

    canvas_shape_arr = np.load(shape_path)
    canvas_shape = tuple(int(x) for x in canvas_shape_arr.tolist())

    logger.info(f"[GEOM] Shape carregado de: {shape_path} => {canvas_shape}")
    logger.info(f"[GEOM] Abrindo memmap: {memmap_path}")

    canvas = np.memmap(
        memmap_path,
        dtype=np.uint8,
        mode="r+",
        shape=canvas_shape
    )
    return canvas


def resolve_node_to_image_path(
    G_geo,
    positions: Positions,
    tiles_dir: Path
) -> NodeToPath:
    """
    Para cada nó presente em `positions`, pega:
      - label = G_geo.nodes[node]["label"]
      - procura em tiles_dir por: f"{label}.*"
    Exige 1 match por nó (você disse que não há stems duplicados).
    """
    if not tiles_dir.exists():
        raise FileNotFoundError(f"TILES_DIR não encontrado: {tiles_dir}")

    node_to_path: NodeToPath = {}

    for node in positions.keys():
        if node not in G_geo.nodes:
            raise KeyError(f"Nó {node} existe em positions mas não existe no grafo geométrico.")

        label = G_geo.nodes[node].get("label")
        if not label:
            raise KeyError(f"Nó {node} não possui atributo 'label' no grafo geométrico.")

        matches = list(tiles_dir.glob(f"{label}.*"))

        if len(matches) == 0:
            raise FileNotFoundError(
                f"Nenhuma imagem encontrada para label='{label}' em {tiles_dir} (padrão '{label}.*')."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguidade: {len(matches)} arquivos encontrados para label='{label}': {matches}"
            )

        node_to_path[node] = matches[0]

    logger.info(f"[GEOM] Mapeamento node->imagem concluído (total={len(node_to_path)}).")
    return node_to_path


def paste_tiles_overwrite(
    canvas: np.memmap,
    positions: Positions,
    node_to_path: NodeToPath,
) -> None:
    """
    Cola os tiles no canvas (memmap) usando overwrite.
    - positions[node] = (X, Y) em pixels (canto superior esquerdo)
    - node_to_path[node] = caminho da imagem do tile

    Estratégia:
      - lê tile com cv2 (BGR)
      - converte para RGB
      - escreve direto no memmap: canvas[y:y+h, x:x+w, :] = tile_rgb
    """
    H, W, C = canvas.shape

    for node, (X, Y) in positions.items():
        if node not in node_to_path:
            raise KeyError(f"node_to_path não contém o nó {node}")

        tile_path = node_to_path[node]

        # Coordenadas globais podem ser float -> arredondamos para o pixel mais próximo
        x0 = int(round(X))
        y0 = int(round(Y))

        tile_bgr = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
        if tile_bgr is None:
            raise FileNotFoundError(f"Não foi possível ler tile: {tile_path}")

        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        th, tw, tc = tile_rgb.shape

        # Checagem simples de limite (clip para evitar index error)
        x1 = min(x0 + tw, W)
        y1 = min(y0 + th, H)

        # Caso o tile fique totalmente fora (não deveria), ignora com warning
        if x0 >= W or y0 >= H:
            logger.warning(f"[GEOM] Tile fora do canvas: node={node}, pos=({x0},{y0}), tile={tile_path.name}")
            continue

        # Se precisar recortar tile (borda), recorta tile também
        tile_crop = tile_rgb[0 : (y1 - y0), 0 : (x1 - x0), :]

        canvas[y0:y1, x0:x1, :] = tile_crop

    canvas.flush()
    logger.info("[GEOM] Colagem concluída (overwrite) e memmap flush() executado.")


def export_preview_jpg(canvas: np.memmap, out_path: Path, quality: int = 95) -> None:
    """
    Exporta um preview JPG do canvas por SUBAMOSTRAGEM.

    Em vez de materializar o canvas inteiro em RAM (np.asarray), lê apenas os
    pixels de um passo (step) direto do memmap, produzindo uma cópia reduzida
    cujo maior lado fica em torno de Config.CANVAS_PREVIEW_MAX_DIM. Assim o
    pico de RAM é O(preview), não O(mosaico inteiro).
    O canvas está em RGB; o OpenCV salva em BGR, então convertemos.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    H, W = canvas.shape[:2]
    max_dim = int(getattr(Config, "CANVAS_PREVIEW_MAX_DIM", 4000))
    step = max(1, int(np.ceil(max(H, W) / max_dim)))

    # Slicing com passo no memmap materializa somente os pixels amostrados.
    small_rgb = np.array(canvas[::step, ::step, :])
    small_bgr = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR)

    ok = cv2.imwrite(str(out_path), small_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"Falha ao salvar preview JPG em: {out_path}")

    logger.info(
        f"[GEOM] Preview JPG salvo em: {out_path} (step={step}, shape={small_rgb.shape})"
    )


def export_bigtiff_rgb(canvas: np.memmap, out_path: Path) -> None:
    """
    Exporta um BigTIFF RGB gravando em tiles.

    O memmap é passado diretamente ao tifffile (sem np.asarray), e a escrita
    tile a tile lê fatias do memmap sob demanda — evitando materializar o
    mosaico inteiro em RAM. Útil para mosaicos grandes.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tile_size = int(getattr(Config, "CANVAS_TIFF_TILE", 256))
    tifffile.imwrite(
        str(out_path),
        canvas,
        photometric="rgb",
        bigtiff=True,
        compression="jpeg",
        tile=(tile_size, tile_size),
    )
    logger.info(f"[GEOM] BigTIFF salvo em: {out_path}")


def main(normalize: bool = True) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] - %(message)s")

    # 1) Abre o canvas geométrico (memmap)
    canvas = open_geom_canvas()
    logger.info(f"[GEOM] Canvas aberto com shape: {canvas.shape}")

    # 2) Carrega posições globais
    positions = load_positions_pickle(Config.GLOBAL_POS_FILE)
    logger.info(f"[GEOM] Positions carregadas: {len(positions)}")

    # 3) Carrega grafo geométrico
    G_geo = load_graph(Config.GEOMETRIC_GRAPH_FILE)
    logger.info(f"[GEOM] Grafo geométrico carregado: nós={G_geo.number_of_nodes()}, arestas={G_geo.number_of_edges()}")

    # 4) Resolve node -> imagem via label.*
    if normalize:
        node_to_path = resolve_node_to_image_path(G_geo, positions, Config.NORMALIZED_DIR)
    else:
        node_to_path = resolve_node_to_image_path(G_geo, positions, Config.TILES_DIR)

    sample = list(node_to_path.items())[:3]
    logger.info(f"[GEOM] Amostra node->path: {sample}")

    # 5) Cola tiles no canvas (overwrite)
    paste_tiles_overwrite(canvas, positions, node_to_path)

    
    # # adiciona data atual ao nome do arquivo
    # now = datetime.now()
    # timestamp = now.strftime("%Y%m%d_%H%M%S") 

    # 6) Export (preview)
    preview_path = Config.CANVAS_PREVIEW_PATH
    export_preview_jpg(canvas, preview_path)

    # 7) (Opcional) Export TIFF final
    tiff_path = Config.CANVAS_GEOM_PATH
    export_bigtiff_rgb(canvas, tiff_path)

    logger.info("[GEOM] Passo 2 OK (colagem + export).")



if __name__ == "__main__":
    start_time = time.perf_counter()
    # populate()
    main()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")
