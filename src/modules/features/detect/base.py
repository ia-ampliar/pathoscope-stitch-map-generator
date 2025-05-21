from abc import ABC, abstractmethod

import cv2
import numpy as np


class FeatureDetector(ABC):
    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def detect_and_compute(self, image: np.ndarray):
        """Retorna keypoints e descritores"""
        pass

    def draw_keypoints(self, image: np.ndarray, keypoints: list[cv2.KeyPoint]):
        """Desenha os keypoints sobre a imagem original."""
        return cv2.drawKeypoints(image, keypoints, None, flags=0)
