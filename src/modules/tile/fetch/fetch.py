import json
import logging
import time
from pathlib import Path
from typing import List

from src.config.config import Config
from src.utils.coordinates import extract_coordinates

logger = logging.getLogger(__name__)


def list_image_files(directory: Path, extensions: List[str]) -> List[str]:
    if not directory.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {directory}")
    return [f.name for f in directory.iterdir() if f.suffix.lower() in extensions]


def generate_metadata(directory: Path) -> List[dict]:
    images = list_image_files(directory, Config.SUPPORTED_EXTENSIONS)
    metadata = []

    for image in images:
        try:
            coordinates = extract_coordinates(image, Config.COORDINATES_PATTERN)
            metadata.append(
                {"name": image, "path": str(directory), "coordinates": coordinates}
            )
        except ValueError as e:
            logging.error(f"Ignorando arquivo inválido: {image} - {e}")

    return metadata


def save_metadata(metadata: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)


def extract_and_save_metadata(tiles_dir: Path, output_path: Path) -> int:
    metadata = generate_metadata(tiles_dir)
    if not metadata:
        raise ValueError(f"Nenhuma imagem válida encontrada em {tiles_dir}")
    save_metadata(metadata, output_path)
    return len(metadata)


def main() -> None:
    start_time = time.perf_counter()
    logger.info("Iniciando extração dos metadados...")

    try:
        count = extract_and_save_metadata(Config.NORMALIZED_DIR, Config.METADATA_FILE)
        elapsed = time.perf_counter() - start_time
        logger.info(f"{count} imagens processadas em {elapsed:.4f} segundos.")
        logger.info(f"Metadados salvos em {Config.METADATA_FILE}")
    except Exception as e:
        # Propaga a falha (código de saída != 0) para que o orquestrador
        # (pipeline.py) aborte, em vez de seguir sobre um dataset.json ausente.
        logger.error(f"Falha na execução: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.DEBUG)
    main()
