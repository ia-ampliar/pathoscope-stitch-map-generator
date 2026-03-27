import subprocess
import sys
import time

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