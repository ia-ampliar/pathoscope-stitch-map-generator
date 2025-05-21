import argparse
import os
import time

import cv2

from .registry import DETECTOR_REGISTRY


def main():
    # Argumentos de linha de comando
    parser = argparse.ArgumentParser(
        description="Detecta keypoints com múltiplos algoritmos e salva resultados."
    )
    parser.add_argument(
        "image_path", type=str, help="Caminho da imagem de entrada (grayscale)"
    )
    args = parser.parse_args()

    # Carrega imagem
    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erro: não foi possível carregar a imagem '{args.image_path}'")
        return

    # Cria pasta de saída
    output_dir = "output/tmp/detector_benchmark"
    os.makedirs(output_dir, exist_ok=True)

    # Loop por todos os detectores registrados
    for name, DetectorClass in DETECTOR_REGISTRY.items():
        print(f"\nExecutando detector: {name.upper()}")

        # Parâmetros específicos por detector
        params = {
            "sift": {"nfeatures": 10000},
            "orb": {"nfeatures": 10000},
            "akaze": {},
        }.get(name, {})

        try:
            detector = DetectorClass(**params)

            start = time.perf_counter()
            keypoints, descriptors = detector.detect_and_compute(img)
            elapsed = time.perf_counter() - start

            print(f" - Keypoints detectados: {len(keypoints)}")
            print(f" - Tempo de execução: {elapsed:.4f} segundos")

            # Desenha e salvar imagem
            img_with_kp = detector.draw_keypoints(img, keypoints)
            output_path = os.path.join(output_dir, f"{name}_keypoints.jpg")
            cv2.imwrite(output_path, img_with_kp)
            print(f" - Imagem salva em: {output_path}")

        except Exception as e:
            print(f"Erro ao executar {name}: {e}")


if __name__ == "__main__":
    main()
