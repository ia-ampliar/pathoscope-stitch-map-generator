import re
from typing import Tuple


def extract_coordinates(filename: str, pattern: str) -> Tuple[int, int]:
    """
    Extrai as coordenadas (x, y) do nome de um arquivo de imagem, com base em uma regex.

    Args:
        filename (str): Nome do arquivo.
        pattern (str): Expressão regular com dois grupos numéricos.

    Returns:
        Tuple[int, int]: Coordenadas extraídas (x, y).

    Raises:
        ValueError: Se o padrão não for encontrado ou for incompleto.
    """
    match = re.search(pattern, filename)

    if not match or len(match.groups()) < 2:
        raise ValueError(
            f"Formato inválido: '{filename}' não contém coordenadas com o padrão '{pattern}'."
        )

    try:
        x, y = int(match.group(1)), int(match.group(2))
        return x, y
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Erro ao converter coordenadas extraídas de '{filename}' para inteiros: {e}"
        )
