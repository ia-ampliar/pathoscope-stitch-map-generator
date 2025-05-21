import cv2

from .base import FeatureDetector


class SIFTDetector(FeatureDetector):
    def __init__(self, **params):
        super().__init__(**params)
        self.sift = cv2.SIFT_create(**params)

    def detect_and_compute(self, image):
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        return keypoints, descriptors
