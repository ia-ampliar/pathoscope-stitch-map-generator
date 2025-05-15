import cv2
import numpy as np

from .base import TileClassifier

# Proporção mínima de pixels pretos (tecido) para considerar a imagem válida
MIN_BLACK_RATIO = 0.01

# Limite: pixels abaixo desse valor são considerados como tecido
THRESHOLD_VALUE = 220


class ThresholdClassifier(TileClassifier):
    def classify(self, image: np.ndarray) -> tuple[bool, np.ndarray]:
        # Converte a imagem para tons de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aplica threshold binário:
        #   - Pixels <= THRESHOLD_VALUE viram preto (0) → considerados como tecido
        #   - Pixels > THRESHOLD_VALUE viram branco (255) → considerados como fundo
        _, binary = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        # Calcula a proporção de pixels pretos (0) na imagem binária
        black_pixels = np.sum(binary == 0)
        total_pixels = binary.size
        tissue_ratio = black_pixels / total_pixels

        # Considera que há tecido se a proporção de preto for suficiente
        return tissue_ratio > MIN_BLACK_RATIO, binary
