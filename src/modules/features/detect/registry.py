from .orb import ORBDetector
from .sift import SIFTDetector

DETECTOR_REGISTRY = {
    "orb": ORBDetector,
    "sift": SIFTDetector,
}


def get_detector(name, **params):
    detector_cls = DETECTOR_REGISTRY.get(name.lower())
    if not detector_cls:
        raise ValueError(f"Detector '{name}' não registrado.")
    return detector_cls(**params)
