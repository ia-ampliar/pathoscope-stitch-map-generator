import pickle
from pathlib import Path
from typing import Dict, Tuple


# Nó do grafo/topologia: (x, y) no grid
Node = Tuple[int, int]
# Posição global em pixels: (X, Y)
PosXY = Tuple[float, float]
Positions = Dict[Node, PosXY]

def load_positions_pickle(path: Path) -> Positions:
    """
    Carrega global_positions.pkl (pickle) no formato:
      {(x,y): (X,Y), ...}
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo global_positions.pkl não encontrado: {path}")

    with path.open("rb") as f:
        positions = pickle.load(f)

    if not isinstance(positions, dict):
        raise ValueError(f"Conteúdo inválido em {path}: esperado dict, obtido {type(positions)}")

    return positions
