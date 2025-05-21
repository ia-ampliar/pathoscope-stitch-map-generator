from typing import List, Literal

import cv2

from .base import FeatureMatcher

# Tipos válidos de algoritmo
FeatureType = Literal["orb", "sift"]


class BFMatcher(FeatureMatcher):
    def __init__(self, ratio_thresh: float = 0.5, algorithm: FeatureType = "orb"):
        self.ratio_thresh = ratio_thresh

        # Define o normType com base no tipo de algoritmo
        if algorithm == "orb":
            norm_type = cv2.NORM_HAMMING
        elif algorithm == "sift":
            norm_type = cv2.NORM_L2
        else:
            raise ValueError(
                f"[ERRO] Algoritmo de feature '{algorithm}' não suportado."
            )

        self.matcher = cv2.BFMatcher(normType=norm_type, crossCheck=False)

    def match(
        self,
        keypoints1: list[cv2.KeyPoint],
        descriptors1,
        keypoints2: list[cv2.KeyPoint],
        descriptors2,
    ) -> List[cv2.DMatch]:

        if descriptors1 is None or descriptors2 is None:
            return []

        raw_matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Aplica o filtro de Lowe
        good_matches = [
            m for m, n in raw_matches if m.distance < self.ratio_thresh * n.distance
        ]
        return good_matches
