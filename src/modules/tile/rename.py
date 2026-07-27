"""
Módulo para padronizar os nomes dos tiles no diretório de entrada.

Renomeia arquivos que seguem o padrão 'tile_xN_yN.<ext>' para
'tile_xN_yN_image.<ext>', garantindo consistência no pipeline.
"""

import re
import time
from pathlib import Path

from src.config.config import Config


def rename_tiles(tiles_dir: Path = None):
    """
    Renomeia tiles no diretório para o padrão esperado pelo pipeline.

    Padrão de entrada:  tile_x{N}_y{N}.ext  (sem sufixo '_image')
    Padrão de saída:    tile_x{N}_y{N}_image.ext

    Tiles que já possuem o sufixo '_image' são ignorados.
    """
    if tiles_dir is None:
        tiles_dir = Config.TILES_DIR

    # Captura tiles que NÃO possuem '_image' antes da extensão
    pattern = re.compile(r"^tile_x(\d+)_y(\d+)(\.[^.]+)$")

    renamed_count = 0
    skipped_count = 0

    for file in sorted(tiles_dir.iterdir()):
        if not file.is_file():
            continue

        match = pattern.match(file.name)
        if not match:
            skipped_count += 1
            continue

        x = int(match.group(1))
        y = int(match.group(2))
        ext = match.group(3)

        new_name = f"tile_x{x}_y{y}_image{ext}"
        new_path = file.with_name(new_name)

        if new_path.exists():
            print(f"[AVISO] Destino já existe, pulando: {new_name}")
            skipped_count += 1
            continue

        file.rename(new_path)
        renamed_count += 1

    print(f"Renomeação concluída: {renamed_count} arquivo(s) renomeado(s), "
          f"{skipped_count} ignorado(s).")


if __name__ == "__main__":
    start_time = time.perf_counter()
    rename_tiles()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")
