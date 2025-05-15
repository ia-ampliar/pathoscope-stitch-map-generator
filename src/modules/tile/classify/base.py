from abc import ABC, abstractmethod

import numpy as np


class TileClassifier(ABC):
    @abstractmethod
    def classify(self, image: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Avalia se a imagem é candidata e a retorna com o filtro utilizado aplicado.
        """
        pass
