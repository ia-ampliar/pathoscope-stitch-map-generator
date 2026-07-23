**Preprocesser.py — Documentação Detalhada

**Visão Geral**
- **Arquivo:** `src/modules/tile/preprocessing/preprocesser.py`
- **Propósito:** Fornece utilitários e pipeline para pré-processamento de tiles de imagem: leitura, cálculo de imagem média (normalização), normalização individual de tiles e orquestração do fluxo completo de pré-processamento. Também disponibiliza um utilitário para organização de arquivos (`move_tiles_to_raw`), que **não** é chamado automaticamente pelo fluxo principal.
- **Efeito colateral principal:** escreve arquivos em diretórios controlados por `Config` (AVERAGE_DIR, NORMALIZED_DIR).

**Principais responsabilidades**
- Ler imagens com segurança (`imread`).
- Calcular imagem média a partir de um conjunto de tiles (`average_images`).
- Normalizar imagens com base na imagem média (`normalize_image`).
- Processar e salvar imagens normalizadas em paralelo (`process_and_save` + `Parallel`).
- Orquestrar o pré-processamento completo (`run_preprocessing`).
- (Utilitário avulso) Mover arquivos soltos para a pasta `raw` (`move_tiles_to_raw`) — **não** é chamado por `run_preprocessing`.

**Dependências e configuração**
- Usa OpenCV (`cv2`) para leitura/escrita e operações de imagem.
- Usa `ThreadPoolExecutor` para leitura paralela em `average_images` e `joblib.Parallel` para normalizar e salvar tiles.
- Utiliza `Config` (de `src.config.config`) para caminhos e constantes:
  - `Config.TILES_DIR`, `Config.RAW_DIR`, `Config.AVERAGE_DIR`, `Config.NORMALIZED_DIR`
  - `Config.SUPPORTED_IMAGE_EXTENSIONS`, `Config.DEFAULT_BRIGHTNESS_FACTOR`, `Config.DEFAULT_EPSILON`

**Funções e comportamento detalhado**

- `imread(filepath: Union[str, Path]) -> Optional[np.ndarray]`
  - Propósito: Leitura segura de imagem. Lança `FileNotFoundError` se o caminho não existe.
  - Retorno: Matriz NumPy da imagem (BGR) ou levanta exceção se arquivo ausente.
  - Observação: Usa `cv2.imread(str(filepath))` — se a leitura falhar por outro motivo, retorna `None`.

- `average_images(image_folder: Union[str, Path]) -> np.ndarray`
  - Propósito: Calcula a imagem média (pixel-wise) de todas as imagens JPG (ou extensões configuradas) presentes na pasta.
  - Comportamento:
    - Verifica existência da pasta; levanta `FileNotFoundError` se não existir.
    - Enumera imagens com base em `Config.SUPPORTED_IMAGE_EXTENSIONS`.
    - Lê imagens em paralelo com `ThreadPoolExecutor` para reduzir I/O-bound latency.
    - Filtra leituras `None`.
    - Normaliza o tamanho das imagens para a dimensão da primeira imagem (faz `cv2.resize` se necessário).
    - Soma as imagens em float64, divide pelo número de imagens e converte para `uint8` arredondando.
  - Erros possíveis:
    - `ValueError` se nenhuma imagem for encontrada ou se nenhuma imagem puder ser lida.
  - Retorno: `np.ndarray` com a imagem média pronta para ser usada como imagem de normalização.
  - Notas de robustez: O método assume que a primeira imagem define as dimensões alvo; imagens com dimensões diferentes são redimensionadas.

- `normalize_image(image: np.ndarray, normalization_image: np.ndarray, brightness_factor: float = Config.DEFAULT_BRIGHTNESS_FACTOR, epsilon: float = Config.DEFAULT_EPSILON) -> np.ndarray`
  - Propósito: Normaliza um tile usando a imagem média.
  - Fórmula básica:
    - converte para float: image_f = image.astype(float32)
    - normalized = image_f / (norm_f + epsilon) * 255.0
    - aplica `brightness_factor`, aplica `np.clip` e converte para `uint8`.
  - Parâmetros:
    - `brightness_factor`: fator multiplicativo para ajuste de brilho final.
    - `epsilon`: evita divisão por zero quando `normalization_image` tem pixels muito baixos.
  - Retorno: imagem normalizada em `uint8`.
  - Observação: A operação preserva proporção local de intensidade e corrige variações de iluminação por divisão ponto-a-ponto.

- `move_tiles_to_raw(tiles_dir: Union[str, Path]) -> None`
  - Propósito: Move arquivos soltos dentro da pasta `tiles` para `Config.RAW_DIR`.
  - Comportamento: cria `RAW_DIR` se necessário e usa `shutil.move` para cada arquivo.
  - Erro: Lança `FileNotFoundError` se `tiles_dir` não existir.
  - Uso típico: utilitário avulso para organização manual de arquivos antes do processamento. **Não é chamado por `run_preprocessing`.**

- `process_and_save(img_path, norm_image, brightness_factor)`
  - Função auxiliar usada por `Parallel` que:
    - Lê a imagem (`imread`) — ignora se `None`.
    - Normaliza chamando `normalize_image`.
    - Grava resultado em `Config.NORMALIZED_DIR / img_path.name` com `cv2.imwrite`.
  - Observações de falhas: silenciosamente retorna se `imread` falhar; não lança exceções aqui.

- `run_preprocessing(tiles_dir: Union[str, Path], brightness_factor: float = Config.DEFAULT_BRIGHTNESS_FACTOR) -> Path`
  - Propósito: Orquestra pipeline completo de pré-processamento.
  - Passos realizados:
    1. Cria diretórios necessários (`RAW_DIR`, `AVERAGE_DIR`, `NORMALIZED_DIR`).
    2. Enumera arquivos em `Config.TILES_DIR` usando `Config.SUPPORTED_IMAGE_EXTENSIONS`.
    3. Levanta `ValueError` se não houver imagens.
    4. Calcula `norm_image` chamando `average_images(Config.TILES_DIR)`.
    5. Salva `norm_image` em `Config.AVERAGE_DIR/normalize.jpg`.
    6. Normaliza e salva todas as imagens em paralelo usando `joblib.Parallel(n_jobs=-1)` e `delayed(process_and_save)`.
  - Retorno: `Config.NORMALIZED_DIR` (Path para os arquivos normalizados).
  - Observações:
    - **Nota:** `move_tiles_to_raw` **não** é chamada dentro de `run_preprocessing`. É um utilitário separado.
    - Paralelização em `Parallel(n_jobs=-1)` usa todos os núcleos disponíveis; pode aumentar uso de I/O simultâneo.
    - A função trata inconsistências de tamanho de imagem ao delegar a `average_images`.
    - Este módulo **não faz parte do pipeline automático** (`pipeline.py`). Deve ser executado manualmente quando necessário.

- Bloco `if __name__ == "__main__":`
  - Executa `run_preprocessing(Config.TILES_DIR)` com medição de tempo e configuração mínima de `logging`.
  - Útil para executar módulo diretamente durante testes/diagnóstico.

**Entradas e Saídas (I/O)**
- Entradas:
  - Arquivos de imagem em `Config.TILES_DIR` (extensões em `Config.SUPPORTED_IMAGE_EXTENSIONS`).
- Saídas (arquivos gravados):
  - `Config.AVERAGE_DIR/normalize.jpg` — imagem média usada para normalização.
  - Arquivos normalizados em `Config.NORMALIZED_DIR/*` (mesmos nomes dos arquivos originais).
  - `Config.RAW_DIR/` se `move_tiles_to_raw` for utilizada.

**Paralelismo e desempenho**
- Leitura das imagens para cálculo da média: `ThreadPoolExecutor` — bom para I/O-bound.
- Normalização/salvamento: `joblib.Parallel(n_jobs=-1)` — aproveita múltiplos núcleos; cuidado com I/O concorrente e disco.
- Recomendações:
  - Para grandes volumes e discos lentos, limitar `n_jobs` em `Parallel` (ex.: `n_jobs=4`).
  - Monitorar RAM: `average_images` carrega todas as imagens na memória; para datasets muito grandes, implementar streaming ou usar blocos.

**Erros e robustez**
- O módulo lança exceções claras quando diretórios/arquivos não existem (`FileNotFoundError`, `ValueError`).
- `process_and_save` ignora imagens que não puderam ser lidas (comportamento tolerante), mas isso pode mascarar problemas de permissão ou corrupção de arquivo.
- `normalize_image` usa `epsilon` para evitar divisão por zero.

**Boas práticas e sugestões de melhoria**
- Serializar leitura e normalização por lotes quando > centenas de imagens para evitar uso excessivo de RAM.
- Expor `n_jobs` como argumento de `run_preprocessing` (atualmente hard-coded como `-1`).
- Adicionar logs informativos dentro de `Parallel` (ex.: progresso por lote) ou utilizar `tqdm` para feedback.
- Tratar explicitamente erros de `cv2.imwrite` (retorno booleano) para detectar falhas de disco.

**Exemplos de uso**
- Executar módulo diretamente (linha de comando):

  python -m src.modules.tile.preprocessing.preprocesser

- Em código Python, chamar: 

  from src.modules.tile.preprocessing.preprocesser import run_preprocessing
  run_preprocessing(Config.TILES_DIR, brightness_factor=1.0)

**Onde procurar no código**
- Implementação completa: [src/modules/tile/preprocessing/preprocesser.py](src/modules/tile/preprocessing/preprocesser.py)
- Configurações relacionadas: `src/config/config.py` — ver constantes de diretórios e extensões.

**Resumo rápido**
- `preprocesser.py` prepara os tiles do pipeline: cria imagem média para normalização, normaliza todos os tiles em paralelo e grava os resultados em `Config.NORMALIZED_DIR`. É eficiente para workflows moderados, mas pode precisar de ajustes para datasets muito grandes (limitar `n_jobs`, leitura por blocos).

---

Se desejar, eu salvo este documento no repositório (docs/PREPROCESSER_DETAILED.md) e/ou abro o arquivo para revisão. Também posso ajustar a seção de desempenho para valores concretos de `n_jobs` e exemplos de profiling.