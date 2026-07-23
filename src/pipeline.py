import subprocess
import sys
import time
from pathlib import Path

# Caminhos lidos do Config (centralizado). Fallback robusto para o caso de o
# orquestrador ser executado como script (python src/pipeline.py), quando o
# pacote `src` pode não estar importável.
try:
    from src.config.config import Config

    NORMALIZED_DIR = Config.NORMALIZED_DIR
    SUPPORTED_EXTENSIONS = tuple(Config.SUPPORTED_EXTENSIONS)
except Exception:
    NORMALIZED_DIR = Path("output") / "tiles" / "normalized"
    SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def preflight() -> None:
    """Valida pré-condições antes de iniciar o pipeline automático.

    O pipeline assume que os tiles já estão normalizados em NORMALIZED_DIR
    (a etapa de preprocessing é manual/opcional). Falhar cedo aqui evita erros
    crípticos em etapas posteriores (ex.: dataset.json ausente no detect).
    """
    if not NORMALIZED_DIR.exists():
        print(
            f"✗ Pré-condição falhou: diretório de tiles normalizados não existe: {NORMALIZED_DIR}\n"
            f"  Execute 'python -m src.modules.tile.preprocessing.preprocesser' "
            f"ou popule {NORMALIZED_DIR} antes de rodar o pipeline."
        )
        sys.exit(1)

    has_images = any(
        p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        for p in NORMALIZED_DIR.iterdir()
    )
    if not has_images:
        print(
            f"✗ Pré-condição falhou: nenhum tile encontrado em {NORMALIZED_DIR} "
            f"(extensões aceitas: {list(SUPPORTED_EXTENSIONS)}).\n"
            f"  Execute o preprocesser ou popule o diretório antes de rodar o pipeline."
        )
        sys.exit(1)

    print(f"✓ Pré-condição OK: tiles normalizados encontrados em {NORMALIZED_DIR}.")


steps = [
    # "src.modules.tile.preprocessing.preprocesser",
    "src.modules.tile.fetch.fetch",
    "src.modules.tile.classify.classifier",
    "src.modules.graph.graph",
    "src.modules.features.detect.detect",
    "src.modules.features.match.match",
    "src.modules.graph.geograph",
    "src.modules.graph.globalpos",
    "src.modules.canvas.create_geom",
    "src.modules.canvas.populate_geom",
]

preflight()

start_time = time.perf_counter()

for step in steps:
    print(f"\n▶ Executando: {step}")
    result = subprocess.run([sys.executable, "-m", step])
    if result.returncode != 0:
        print(f"✗ Falhou em: {step}")
        sys.exit(result.returncode)

print("\n✓ Pipeline concluído com sucesso!")

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"Tempo total de execução: {elapsed:.2f} segundos")