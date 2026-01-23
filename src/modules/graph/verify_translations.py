from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import zarr

from src.config.config import Config
from src.modules.graph.graph import load_graph

logger = logging.getLogger(__name__)

Node = Tuple[int, int]
Pair = Tuple[Node, Node]


def _find_match_zarr_for_pair(matches_dir: Path, label_u: str, label_v: str) -> Tuple[Path, str]:
    """
    Procura o arquivo .zarr correspondente ao par (u,v).
    Retorna:
      (path, direction)
    onde direction é:
      - "U__V" se o arquivo encontrado foi label_u__label_v.zarr
      - "V__U" se foi label_v__label_u.zarr
    """
    p_uv = matches_dir / f"{label_u}__{label_v}.zarr"
    if p_uv.exists():
        return p_uv, "U__V"

    p_vu = matches_dir / f"{label_v}__{label_u}.zarr"
    if p_vu.exists():
        return p_vu, "V__U"

    raise FileNotFoundError(
        f"Não encontrei match zarr para o par: {label_u} <-> {label_v} "
        f"em {matches_dir} (nem {p_uv.name} nem {p_vu.name})."
    )


def _read_dxdy_from_match_zarr(zarr_path: Path) -> Tuple[float, float]:
    """
    Lê attrs['translation_matrix'] do .zarr.
    No seu pipeline, normalmente fica em <zarr>/matches attrs.
    Então: tenta no root e, se não achar, tenta no subgrupo 'matches'.
    """
    # 1) tenta root
    root = zarr.open_group(str(zarr_path), mode="r")
    if "translation_matrix" in root.attrs:
        M = np.array(root.attrs["translation_matrix"], dtype=float)
    else:
        # 2) tenta subgrupo matches
        try:
            matches_grp = zarr.open_group(str(zarr_path / "matches"), mode="r")
        except Exception as e:
            raise KeyError(
                f"{zarr_path.name} não tem attrs['translation_matrix'] no root "
                f"e não consegui abrir subgrupo 'matches': {e}"
            )

        if "translation_matrix" not in matches_grp.attrs:
            raise KeyError(
                f"{zarr_path.name} não tem attrs['translation_matrix'] nem no root nem em 'matches/'."
            )

        M = np.array(matches_grp.attrs["translation_matrix"], dtype=float)

    if M.shape != (3, 3):
        raise ValueError(f"{zarr_path.name}: translation_matrix shape inválido: {M.shape} (esperado 3x3).")

    dx = float(M[0, 2])
    dy = float(M[1, 2])
    return dx, dy



def verify_translation_sign_convention(
    topology_graph_path: Optional[Path] = None,
    matches_dir: Optional[Path] = None,
    *,
    pairs: Optional[List[Pair]] = None,
    max_print: int = 30,
) -> Dict[str, int]:
    """
    Verifica se os dx/dy salvos nos matches (.zarr) têm sinal compatível com o deslocamento esperado
    pelo grid (node=(x,y)).

    Como funciona:
      - Para cada aresta (u,v) do grafo topológico (ou para cada par fornecido em `pairs`):
        - encontra o arquivo .zarr do par (label_u__label_v ou label_v__label_u)
        - lê dx,dy do attrs['translation_matrix']
        - normaliza para o sentido u→v:
            se arquivo é V__U, então dx_uv = -dx, dy_uv = -dy
        - calcula o sinal esperado:
            expected_dx_sign = sign(v.x - u.x)  (direita => +, esquerda => -)
            expected_dy_sign = sign(v.y - u.y)  (baixo => +, cima => -)
        - compara apenas quando a diferença em x ou y for != 0

    Retorna contagem de ok/mismatch e imprime amostras.

    Interpretação:
      - Se muitos pares "direita" (v.x>u.x) têm dx_uv < 0, então o dx salvo está invertido
        para uso como offset de canvas. Nesse caso, você provavelmente precisa usar (-dx,-dy)
        como deslocamento no grafo geométrico (ou na propagação do globalpos).
    """
    topology_graph_path = topology_graph_path or Config.TOPOLOGY_GRAPH_FILE
    matches_dir = matches_dir or (Config.BASE_DIR / "matches")

    G_top = load_graph(topology_graph_path)

    # Se pairs não foi passado, usamos as arestas do grafo topológico
    if pairs is None:
        # G_top pode ser Graph/DiGraph; pegamos pares únicos
        pairs = [(u, v) for (u, v) in G_top.edges()]

    total = 0
    ok = 0
    mismatch = 0
    missing = 0
    printed = 0

    def sgn(z: int) -> int:
        return 0 if z == 0 else (1 if z > 0 else -1)

    for u, v in pairs:
        total += 1

        # Esperado pelo grid
        exp_dx_sign = sgn(v[0] - u[0])
        exp_dy_sign = sgn(v[1] - u[1])

        # label precisa existir no grafo topológico (se não existir, dá pra adaptar)
        try:
            label_u = G_top.nodes[u]["label"]
            label_v = G_top.nodes[v]["label"]
        except Exception as e:
            missing += 1
            if printed < max_print:
                logger.warning(f"[SKIP] Nó sem label no grafo topológico: u={u}, v={v}. Erro: {e}")
                printed += 1
            continue

        # acha zarr do match
        try:
            zarr_path, direction = _find_match_zarr_for_pair(matches_dir, label_u, label_v)
        except FileNotFoundError as e:
            missing += 1
            if printed < max_print:
                logger.warning(f"[MISSING] {e}")
                printed += 1
            continue

        # lê dx/dy "como salvo"
        try:
            dx, dy = _read_dxdy_from_match_zarr(zarr_path)
        except Exception as e:
            missing += 1
            if printed < max_print:
                logger.warning(f"[MISSING/INVALID] {zarr_path.name}: {e}")
                printed += 1
            continue

        # normaliza para u→v
        dx_uv, dy_uv = (dx, dy) if direction == "U__V" else (-dx, -dy)

        # compara sinal (somente nos eixos relevantes)
        axis_mismatch = False

        if exp_dx_sign != 0:
            got_dx_sign = sgn(int(np.sign(dx_uv)))
            if got_dx_sign != exp_dx_sign:
                axis_mismatch = True

        if exp_dy_sign != 0:
            got_dy_sign = sgn(int(np.sign(dy_uv)))
            if got_dy_sign != exp_dy_sign:
                axis_mismatch = True

        if axis_mismatch:
            mismatch += 1
            if printed < max_print:
                logger.warning(
                    f"[MISMATCH] u={u} ({label_u}) -> v={v} ({label_v}) | "
                    f"expected_sign(dx,dy)=({exp_dx_sign},{exp_dy_sign}) | "
                    f"got(dx_uv,dy_uv)=({dx_uv:.2f},{dy_uv:.2f}) | file={zarr_path.name} ({direction})"
                )
                printed += 1
        else:
            ok += 1

    logger.info(f"[VERIFY] total={total} ok={ok} mismatch={mismatch} missing={missing}")

    # Heurística simples: se mismatch domina, provavelmente precisa inverter no uso como offset de canvas
    if ok > 0 and mismatch > ok:
        logger.info(
            "[VERIFY] Muitos mismatches. Forte indicação de que (dx,dy) salvo é mais 'mapeamento pixel' "
            "do que 'offset no canvas'. Provável correção: usar (-dx,-dy) como deslocamento u->v "
            "na construção do grafo geométrico (ou subtrair no globalpos)."
        )

    return {"total": total, "ok": ok, "mismatch": mismatch, "missing": missing}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] - %(message)s")
    verify_translation_sign_convention()
