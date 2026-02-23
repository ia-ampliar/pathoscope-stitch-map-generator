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
    BASE_DIRTILES_DIR: Path = BASE_DIR / "tiles"
    TILES_DIR: Path = BASE_DIR / "tiles" / "src"
    RAW_DIR: Path = BASE_DIR / "tiles" / "raw"
    AVERAGE_DIR: Path = BASE_DIR / "tiles" / "average"
    NORMALIZED_DIR: Path = BASE_DIR / "tiles" / "normalized"
    # Onde as imagens classificadas serão salvas
    CLASSIFIED_DIR: Path = BASE_DIR / "tiles" / "classified"

    # Diretórios para etapas posteriores
    FEATURES_DIR: Path = BASE_DIR / "features"
    MATCHES_DIR: Path = BASE_DIR / "matches"
    RESULT_DIR: Path = BASE_DIR / "result"

    # Constants
    DEFAULT_BRIGHTNESS_FACTOR: float = 0.8
    DEFAULT_EPSILON: float = 1e-5
    SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    # Etapa de classificação
    CLASSIFIER_MODEL_PATH: Path = BASE_DIR / "models" / "classifier_model.pkl"

    # Onde as imagens classificadas serão salvas
    CLASSIFIED_DIR: Path = BASE_DIR / "tmp" / "classified"

    # Arquivos
    METADATA_FILE: Path = BASE_DIR / "metadata" / "dataset.json"
    VALID_TILES_FILE: Path = BASE_DIR / "metadata" / "valid_tiles.json"

    # Grafo
    GRAPH_FILE: Path = BASE_DIR / "result" / "graph.jpg"
    TOPOLOGY_GRAPH_FILE: Path = BASE_DIR / "result" / "graph_topology.gpickle"
    GEOMETRIC_GRAPH_FILE: Path = BASE_DIR / "result" / "graph_geometric.gpickle"
    GEOMETRIC_GRAPH_WEIGHTS_FILE: Path = BASE_DIR / "result" / "graph_geometric_weights.jpg"
    GLOBAL_POS_FILE: Path = BASE_DIR / "result" / "global_positions.pkl"

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
    BLANK_CANVAS_GEOM_PATH: Path = CANVAS_OUTPUT_PATH / "canvas_geom.dat"
    CANVAS_CHUNK_SIZE: tuple = (1024, 1024, 3)  # Ajuste conforme seu hardware
    CANVAS_SHAPE_PATH: Path = CANVAS_OUTPUT_PATH / "canvas_shape.npy"
    CANVAS_GEOM_SHAPE_PATH: Path = CANVAS_OUTPUT_PATH / "canvas_geom_shape.npy"
    CANVAS_POPULATED_PATH: Path = CANVAS_OUTPUT_PATH / "canvas_populated.tif"
    CANVAS_WITH_DRAW_MATCHES_PATH: Path = CANVAS_OUTPUT_PATH / "canvas_with_matches.tif"
    CANVAS_PREVIEW_PATH: Path = CANVAS_OUTPUT_PATH / "mosaic_preview.jpg"
    CANVAS_GEOM_PATH: Path = CANVAS_OUTPUT_PATH / "mosaic_geom.tif"
    CANVAS_GAP: int = 100
    CANVAS_MAX_MATCHES_TO_DRAW = 20
    CANVAS_FILL_VALUE: int = 255

    # limites conservadores (ajustáveis)
    MAX_SHIFT: float = 2000.0          # limite duro do vetor
    MAX_ORTHO: float = 600.0           # quanto aceitamos de "escorregão" no eixo ortogonal
    MIN_MAIN: float = 100.0            # evita dx/dy ~0 em vizinho
    MAX_MAIN: float = 1400.0           # evita saltos > ~1 tile