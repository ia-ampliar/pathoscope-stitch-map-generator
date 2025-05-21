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

    # Etapa de detecção de keypoints
    DETECTION_ALGORITHM: str = "orb"
    DETECTION_N_JOBS: int = -1
    FEATURES_DIR: Path = BASE_DIR / "features"
    KEYPOINTS_ZARR_STORE: Path = FEATURES_DIR / "features.zarr"

    # Etapa de matching
    MATCHING_ZARR_PATH: Path = BASE_DIR / "matches"
    MATCHER: str = "bf"
    MATCHING_RATIO_THRESH: float = 0.5
    MATCHING_N_JOBS = -1

    # Etapa de criação do canvas
    CANVAS_OUTPUT_PATH: Path = BASE_DIR / "tmp" / "canvas"
    BLANK_CANVAS_PATH: Path = CANVAS_OUTPUT_PATH / "canvas.dat"
    CANVAS_CHUNK_SIZE: tuple = (1000, 1000, 3)  # Ajuste conforme seu hardware
    CANVAS_SHAPE_PATH: Path = CANVAS_OUTPUT_PATH / "canvas_shape.npy"
    CANVAS_POPULATED_PATH: Path = CANVAS_OUTPUT_PATH / "canvas_populated.tif"
    CANVAS_GAP: int = 100
