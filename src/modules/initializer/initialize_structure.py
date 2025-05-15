import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.DEBUG)


def init_structure() -> None:
    base_dir = os.path.abspath("output")
    subdirs: list[str] = ["features", "matches", "metadata", "result", "tiles", "tmp"]

    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)

    logging.info(
        f"Estrutura de diretórios criada. Copie os tiles para: {base_dir}\\tiles"
    )


if __name__ == "__main__":
    init_structure()
