import cv2
import numpy as np

from src.config.config import Config

from .base import TileClassifier


class ThresholdClassifier(TileClassifier):
    def classify(self, image: np.ndarray) -> tuple[bool, np.ndarray]:
        # Converte a imagem para tons de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aplica threshold binário:
        #   - Pixels <= TISSUE_THRESHOLD_VALUE viram preto (0) → considerados como tecido
        #   - Pixels > TISSUE_THRESHOLD_VALUE viram branco (255) → considerados como fundo
        _, binary = cv2.threshold(
            gray, Config.TISSUE_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY
        )

        # Calcula a proporção de pixels pretos (0) na imagem binária
        black_pixels = np.sum(binary == 0)
        total_pixels = binary.size
        tissue_ratio = black_pixels / total_pixels

        # Considera que há tecido se a proporção de preto for suficiente
        return tissue_ratio > Config.TISSUE_MIN_BLACK_RATIO, binary
