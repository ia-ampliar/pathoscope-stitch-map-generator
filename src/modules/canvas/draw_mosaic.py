import cv2
import zarr
import numpy as np
import pathlib
from src.config.config import Config
from datetime import datetime
import tifffile
from pathlib import Path

def load_mosaic_data():
    """
    Varre os stores Zarr de matches para recuperar as transformações calculadas.
    """
    transformations = []
    matches_path = Path(Config.MATCHING_ZARR_PATH)
    
    for zarr_path in matches_path.glob("*.zarr"):
        store = zarr.open(str(zarr_path), mode='r')
        # Acessa o grupo 'matches' e seus atributos salvos no passo de RANSAC
        group = store['matches']
        
        # Prioriza a matriz afim conforme discutido na conversa history
        matrix = group.attrs.get("affine_matrix") or group.attrs.get("translation_matrix")
        
        if matrix:
            transformations.append({
                "tile_a": group.attrs["tile_a"],
                "tile_b": group.attrs["tile_b"],
                "matrix": np.array(matrix)
            })
    return transformations

def draw_mosaic():
    """
    Aplica as transformações e salva o mosaico final em formato .tif [3, 4].
    """
    print("[MOSAICO] Iniciando a costura final...")
    transformations = load_mosaic_data()
    
    if not transformations:
        print("Nenhuma transformação encontrada nos arquivos Zarr.")
        return

    # 1. Definir o tamanho do Canvas (ajustar conforme os metadados dos tiles [8])
    # Para mosaicos grandes, o Zarr/Tifffile é superior ao JPG [3, 9]
    canvas_size = (5000, 5000, 3) 
    canvas = np.zeros(canvas_size, dtype=np.uint8)

    tiles_dir = Path("output/tiles")

    for trans in transformations:
        # Carregar imagem do tile_a (origem)
        img_path = tiles_dir / f"{trans['tile_a']}.jpg"
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"Aviso: Não foi possível carregar o tile {trans['tile_a']}")
            continue

        # 1. Se for uma matriz afim (2x3), transforme em 3x3
        matrix = trans['matrix']
        if matrix.shape == (2, 3):
            h_matrix = np.eye(3)
            h_matrix[:2, :] = matrix
            matrix = h_matrix


        # 2. Aplicar a transformação de perspectiva/afim [10, 11]
        # WarpPerspective funciona para ambas se a matriz for 3x3 [12]
        h, w = canvas_size[:2]
        warped_tile = cv2.warpPerspective(
            img, 
            matrix.astype(np.float32), 
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT
        )

        # 3. Blending básico: Adiciona ao canvas onde há pixels da nova imagem [13]
        mask = (warped_tile > 0)
        canvas[mask] = warped_tile[mask]
        print(f"Tile {trans['tile_a']} posicionado com sucesso.")

    # 4. Salvar utilizando tifffile para manter compatibilidade e performance [3, 4]
    agora = datetime.now()
    output_file = pathlib.Path(f"output/tmp/canvas/mosaic_final_{agora.strftime('%d%m%Y_%H%M%S')}.tif")

    tifffile.imwrite(
        str(output_file), 
        canvas, 
        compression='zstd', # Compressão eficiente suportada pelo ecossistema Zarr [14, 15]
        photometric='rgb'
    )
    
    print(f"[SUCESSO] Mosaico salvo em: {output_file}")

if __name__ == "__main__":
    draw_mosaic()