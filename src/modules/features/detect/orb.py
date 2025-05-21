import cv2

from .base import FeatureDetector


class ORBDetector(FeatureDetector):
    def __init__(self, **params):
        super().__init__(**params)
        self.orb = cv2.ORB_create(**params)

    def detect_and_compute(self, image):
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors
