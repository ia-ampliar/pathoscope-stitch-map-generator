from pathlib import Path


class Config:
    # Expressão regular para coordenadas no nome do arquivo
    COORDINATES_PATTERN: str = r".*_x(\d+)_y(\d+)_.*"

    # Extensões de imagem suportadas
    SUPPORTED_EXTENSIONS: list[str] = [
        ext.lower() for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    ]

    # Diretórios
    BASE_DIR: Path = Path("output")
    TILES_DIR: Path = BASE_DIR / "tiles"

    # Onde as imagens classificadas serão salvas
    CLASSIFIED_DIR: Path = BASE_DIR / "tmp" / "classified"

    # Arquivos
    METADATA_FILE: Path = BASE_DIR / "metadata" / "dataset.json"
    VALID_TILES_FILE: Path = BASE_DIR / "metadata" / "valid_tiles.json"

    # Grafo
    GRAPH_FILE: Path = BASE_DIR / "result" / "graph.jpg"
