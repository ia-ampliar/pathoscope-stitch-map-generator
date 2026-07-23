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
    TILES_DIR: Path = BASE_DIR / "tiles" / "src"
    RAW_DIR: Path = BASE_DIR / "tiles" / "raw"
    AVERAGE_DIR: Path = BASE_DIR / "tiles" / "average"
    NORMALIZED_DIR: Path = BASE_DIR / "tiles" / "normalized"

    # Diretórios para etapas posteriores
    FEATURES_DIR: Path = BASE_DIR / "features"
    MATCHES_DIR: Path = BASE_DIR / "matches"
    RESULT_DIR: Path = BASE_DIR / "result"

    # Constants
    DEFAULT_BRIGHTNESS_FACTOR: float = 0.9
    DEFAULT_EPSILON: float = 1e-6
    SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    # Etapa de classificação
    CLASSIFIER_MODEL_PATH: Path = BASE_DIR / "models" / "classifier_model.pkl"
    # Critérios do ThresholdClassifier (tecido vs. fundo)
    TISSUE_THRESHOLD_VALUE: int = 220    # pixels <= valor são considerados tecido
    TISSUE_MIN_BLACK_RATIO: float = 0.1  # proporção mínima de tecido p/ tile válido

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
    DETECTION_ALGORITHM: str = "sift"
    DETECTION_N_JOBS: int = -1
    DETECTION_MIN_KEYPOINTS: int = 4     # mínimo de keypoints p/ registrar o tile
    KEYPOINTS_ZARR_STORE: Path = FEATURES_DIR / "features.zarr"

    # Etapa de matching
    MATCHING_ZARR_PATH: Path = BASE_DIR / "matches"
    MATCHER: str = "bf"
    MATCHING_RATIO_THRESH: float = 0.5
    MATCHING_N_JOBS = -1
    MATCHING_MIN_MATCHES: int = 4          # mínimo de matches p/ tentar RANSAC
    # Parâmetros do RANSAC (cv2.estimateAffinePartial2D)
    RANSAC_REPROJ_THRESHOLD: float = 5.0   # tolerância de reprojeção (px)
    RANSAC_MAX_ITERS: int = 2000
    RANSAC_CONFIDENCE: float = 0.99
    RANSAC_REFINE_ITERS: int = 10

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
    CANVAS_PREVIEW_MAX_DIM: int = 4000  # dimensão máx. (px) do preview JPG (subamostragem)
    CANVAS_TIFF_TILE: int = 256         # tamanho do tile (px) na escrita do BigTIFF

    # Reprodutibilidade
    RANDOM_SEED: int = 42  # Seed para RANSAC (cv2) e numpy; garante resultados determinísticos

    # limites conservadores (ajustáveis)
    MAX_SHIFT: float = 2000.0          # limite duro do vetor
    MAX_ORTHO: float = 600.0           # quanto aceitamos de "escorregão" no eixo ortogonal
    MIN_MAIN: float = 100.0            # evita dx/dy ~0 em vizinho
    MAX_MAIN: float = 1400.0           # evita saltos > ~1 tile