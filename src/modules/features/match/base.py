from abc import ABC, abstractmethod
from typing import List

import cv2


class FeatureMatcher(ABC):
    """
    Interface base para algoritmos de matching de features.
    """

    @abstractmethod
    def match(
        self,
        keypoints1: list[cv2.KeyPoint],
        descriptors1,
        keypoints2: list[cv2.KeyPoint],
        descriptors2,
    ) -> List[cv2.DMatch]:
        """
        Realiza matching entre dois conjuntos de keypoints e descritores.

        Args:
            keypoints1: Lista de cv2.KeyPoint da imagem A.
            descriptors1: Array de descritores da imagem A.
            keypoints2: Lista de cv2.KeyPoint da imagem B.
            descriptors2: Array de descritores da imagem B.

        Returns:
            Lista de objetos cv2.DMatch com os matches válidos.
        """
        pass
