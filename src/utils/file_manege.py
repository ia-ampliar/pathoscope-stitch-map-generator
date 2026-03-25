import os
from pathlib import Path
import shutil

# Função que ler as imagens de uma pasta no formato 00001_x1_y1_zp1.jpg 
# e retorna uma lista com as imagens cujo o x e y são ímpares
def list_tile_images(dir_path: Path) -> list[Path]:
    """
    Lista os arquivos de imagem de tile em dir_path, filtrando apenas aqueles
    cujo nome segue o formato esperado e tem x e y ímpares.
    Exemplo de nome esperado: 00001_x1_y1_zp1.jpg
    """
    tile_paths = []
    counter = 0

    for img_path in dir_path.glob("*.*"):
        if img_path.is_file():
            name_parts = img_path.stem.split("_")
            if len(name_parts) >= 4:
                try:
                    x_part = next(part for part in name_parts if part.startswith("x"))
                    y_part = next(part for part in name_parts if part.startswith("y"))
                    x_val = int(x_part[1:])
                    y_val = int(y_part[1:])
                    if x_val % 2 == 1 and y_val % 2 == 1:
                        tile_paths.append(img_path)
                except (StopIteration, ValueError):
                    continue
    return sorted(tile_paths)



# Copia uma lista de imagens da orignem para o destino
def copy_tiles(tile_paths: list[Path], dest_dir: Path) -> None:
    """
    Copia os arquivos de tile listados em tile_paths para dest_dir.
    """
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)

    for tile_path in tile_paths:
        if tile_path.is_file():
            dest_path = dest_dir / tile_path.name
            if not dest_path.exists():
                shutil.copy(str(tile_path), str(dest_path))


#Função que ler as imagens de uma pasta no formato 00001_x1_y1_zp1.jpg 
# e retorna uma lista com as imagens cujo o primeiro número antes do _ é impar


def normalize_tile_coordinates(
    dir_path: Path, *, dest_dir: Path | None = None
) -> dict[str, str]:
    """Ajusta os índices coordenados para uma malha contínua.

    Leitura dos nomes de arquivo na pasta indicada e mapeamento dos valores
    x/y ímpares para uma sequência 1..N mantendo a relação de ordem.

    Se ``dest_dir`` for informado os arquivos são copiados para lá com os
    novos nomes; caso contrário os próprios arquivos em ``dir_path`` são
    renomeados. Retorna um dicionário de correspondência entre nome antigo e
    novo.
    """
    entries: list[tuple[Path, int, int, str, str | None, str]] = []

    for p in dir_path.glob("*.*"):
        if not p.is_file():
            continue
        parts = p.stem.split("_")
        if len(parts) < 3:
            continue
        try:
            prefix = parts[0]
            x_part = next(part for part in parts if part.startswith("x"))
            y_part = next(part for part in parts if part.startswith("y"))
            zp = next((part for part in parts if part.startswith("zp")), None)
            x_val = int(x_part[1:])
            y_val = int(y_part[1:])
        except (StopIteration, ValueError):
            continue
        entries.append((p, x_val, y_val, prefix, zp, p.suffix))

    if not entries:
        return {}

    xs = sorted({x for _, x, _, *_ in entries})
    ys = sorted({y for _, _, y, *_ in entries})
    map_x = {orig: i + 1 for i, orig in enumerate(xs)}
    map_y = {orig: i + 1 for i, orig in enumerate(ys)}

    if dest_dir:
        dest_dir.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, str] = {}
    for p, x, y, prefix, zp, ext in entries:
        new_x = map_x[x]
        new_y = map_y[y]
        pieces = [prefix, f"x{new_x}", f"y{new_y}"]
        if zp:
            pieces.append(zp)
        new_name = "_".join(pieces) + ext
        mapping[p.name] = new_name
        if dest_dir:
            shutil.copy(str(p), str(dest_dir / new_name))
        else:
            p.rename(p.with_name(new_name))
    return mapping


if __name__ == "__main__":    
    src_path = Path(r"C:\Users\IA\Documents\Pathoscope\Costura\escaneamento-27-01-2025-14-44-29-40x-LED\tiles")
    dst_path = Path(r"C:\Users\IA\Documents\Pathoscope\Costura\pathoscope-stitch-map-generator-dev\pathoscope-stitch-map-generator\output\tiles\src")

    # tile_paths = list_tile_images(src_path)
    # print(f"Arquivos encontrados: {len(tile_paths)}")
    # copy_tiles(tile_paths, dst_path)

    mapping = normalize_tile_coordinates(dst_path)