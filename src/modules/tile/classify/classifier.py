import gc
import json
import logging
import time
from pathlib import Path
from typing import Tuple

import cv2
from joblib import Parallel, delayed

from src.config.config import Config

from .threshold import ThresholdClassifier

logger = logging.getLogger(__name__)


def _process_tile(
    img_path: Path, classifier_cls=ThresholdClassifier
) -> Tuple[str, bool]:
    classifier = classifier_cls()

    tile_name = img_path.stem
    image = cv2.imread(str(img_path))
    is_valid, binary_debug = classifier.classify(image)

    debug_output_path = Config.CLASSIFIED_DIR / f"{tile_name}.jpg"
    cv2.imwrite(str(debug_output_path), binary_debug)

    # Libera memória
    del image, binary_debug, classifier
    gc.collect()

    return tile_name, bool(is_valid)


def run_classification(parallel: bool = True, n_jobs: int = -1):
    start_time = time.perf_counter()

    Config.CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
    Config.VALID_TILES_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Lista com os caminhos das imagens
    image_paths = sorted(Config.NORMALIZED_DIR.glob("*.jpg"))

    if parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_tile)(img_path) for img_path in image_paths
        )
    else:
        results = [_process_tile(img_path) for img_path in image_paths]

    # Constrói o dicionário de tiles válidos
    valid_tiles = {tile_name: is_valid for tile_name, is_valid in results}

    # Salva JSON com tiles válidos
    with open(Config.VALID_TILES_FILE, "w") as f:
        json.dump(valid_tiles, f, indent=2)

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Classificação de {len(image_paths)} imagens concluída em {elapsed:.2f} segundos."
    )
    logger.info(f"Resultados salvos em {Config.VALID_TILES_FILE}")


if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.DEBUG)
    run_classification(parallel=True)
