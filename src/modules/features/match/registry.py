from typing import Any, Callable, Dict

from .base import FeatureMatcher
from .bf import BFMatcher

# Dicionário de matchers registrados
MATCHER_REGISTRY: Dict[str, Callable[..., FeatureMatcher]] = {
    "bf": lambda **kwargs: BFMatcher(**kwargs),
}


def get_matcher(name: str, **kwargs: Any) -> FeatureMatcher:
    """
    Recupera uma instância do matcher registrado, passando parâmetros opcionais.

    Args:
        name: Nome do matcher (ex: "bf")
        **kwargs: Parâmetros opcionais para o matcher (ex: algorithm="sift")

    Returns:
        Instância do FeatureMatcher
    """
    name = name.lower()
    if name not in MATCHER_REGISTRY:
        raise ValueError(
            f"Matcher '{name}' não registrado. Opções: {list(MATCHER_REGISTRY.keys())}"
        )
    return MATCHER_REGISTRY[name](**kwargs)
