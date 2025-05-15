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

    # Arquivos
    METADATA_FILE: Path = BASE_DIR / "metadata" / "dataset.json"
