# Plano de Ação — Pipeline de Stitching

> Documento derivado do diagnóstico arquitetural do repositório. Cada ação cita a evidência no código (`caminho:linha`), define a intervenção concreta e o critério de aceitação.
>
> **Escopo:** correções e refatorações incrementais, de risco controlado. Nenhuma reescrita total.
>
> **Como usar:** marque as caixas conforme o progresso. As ações estão ordenadas por horizonte e, dentro de cada horizonte, por relação impacto/esforço.

---

## Legenda

- **Impacto:** Alto / Médio / Baixo — quanto o problema afeta correção, onboarding ou escala.
- **Esforço:** Alto / Médio / Baixo — estimativa relativa de trabalho.
- **Status:** ⬜ pendente · 🟡 em andamento · ✅ concluído
- **Tipo:** `fato` (observado no código) · `hipótese` · `decisão de produto` (requer aprovação antes de executar).

---

## Resumo de prioridades (Top 5)

| Ordem | Ação | Impacto | Esforço | Horizonte |
|-------|------|---------|---------|-----------|
| 1 | H1-01 · Eliminar falha silenciosa no `fetch` | Alto | Baixo | 1 |
| 2 | H1-03 · Limpar duplicatas e centralizar magic numbers no `Config` | Médio | Baixo | 1 |
| 3 | H2-03 · Otimizar I/O do matching (store/keypoints) | Médio | Baixo/Médio | 2 |
| 4 | H2-04 · Exportação do canvas por blocos (não materializar em RAM) | Alto | Médio | 2 |
| 5 | H1-02 · Validar contrato de diretórios (`src` vs `normalized`) | Alto | Médio | 1 |

---

## Horizonte 1 — Correções imediatas

> Corrigem comportamentos que arriscam ou mascaram o resultado. Prioridade máxima.

### ✅ H1-01 — Propagar falhas no `fetch` (fim da falha silenciosa)
- **Tipo:** `fato` · **Impacto:** Alto · **Esforço:** Baixo
- **Problema:** `fetch.main()` captura `Exception` e apenas registra no log, retornando com código de saída 0. O orquestrador só aborta em `returncode != 0`, então o pipeline **prossegue sobre um `dataset.json` inexistente** e só quebra depois, em `detect`, com mensagem enganosa.
- **Evidência:** `src/modules/tile/fetch/fetch.py:58-59`; `src/pipeline.py:20-23`.
- **Intervenção:**
  1. No bloco `except`, registrar o erro e re-lançar (`raise`) ou encerrar com `sys.exit(1)`.
  2. Ao final de cada etapa, validar a existência do artefato de saída esperado.
- **Critério de aceitação:** com `NORMALIZED_DIR` vazio/ausente, `python -m src.modules.tile.fetch.fetch` retorna código ≠ 0 e o `pipeline.py` aborta na 1ª etapa com mensagem apontando a causa raiz.

### ✅ H1-02 — Validar contrato de diretórios (`src` vs `normalized`)
- **Tipo:** `fato` (contradição com README) · **Impacto:** Alto · **Esforço:** Médio
- **Problema:** o README trata a normalização como opcional, mas `fetch`, `classify`, `detect` e `populate_geom` leem de `NORMALIZED_DIR`, que só é populado pela etapa manual `preprocesser`. Além disso, `create_geom` lê a amostra de `TILES_DIR` (src) enquanto `populate_geom` cola de `NORMALIZED_DIR` — dupla dependência.
- **Evidência:** `src/modules/tile/classify/classifier.py:44`; `src/modules/canvas/populate_geom.py:194`; `src/pipeline.py:6` (preprocesser comentado); `src/modules/canvas/create_geom.py` (`list_tile_images(Config.TILES_DIR)`); `src/modules/initializer/initialize_structure.py:11` (não cria `tiles/src` nem `tiles/normalized`).
- **Intervenção:**
  1. Guard-clause no início do `pipeline.py`: validar que `NORMALIZED_DIR` existe e é não-vazio; caso contrário, mensagem acionável ("execute o preprocesser ou popule `output/tiles/normalized`").
  2. Unificar a fonte de tiles entre `create_geom` e `populate_geom` (ler ambos do mesmo diretório).
- **Critério de aceitação:** executar o pipeline sem `NORMALIZED_DIR` populado produz erro claro imediato; `create_geom` e `populate_geom` referenciam o mesmo diretório de tiles.

### ✅ H1-03 — Limpar duplicatas e centralizar magic numbers no `Config`
- **Tipo:** `fato` · **Impacto:** Médio · **Esforço:** Baixo
- **Problema:** `Config` define `CLASSIFIED_DIR` e `FEATURES_DIR` duas vezes (a 2ª vence, silenciosamente), mantém o atributo com typo `BASE_DIRTILES_DIR` sem uso, e parâmetros de algoritmo estão como literais espalhados fora do `Config`.
- **Evidência:** `src/config/config.py:21` e `:37` (`CLASSIFIED_DIR`); `:24` e `:52` (`FEATURES_DIR`); `:15` (`BASE_DIRTILES_DIR`); magic numbers em `src/modules/tile/classify/threshold.py:7,10`; `src/modules/features/match/match.py:69,97-100`; `src/modules/features/detect/detect.py:35`.
- **Intervenção:**
  1. Remover as definições duplicadas e o atributo com typo.
  2. Mover para `Config` (agrupados por etapa): `MIN_BLACK_RATIO`, `THRESHOLD_VALUE`, parâmetros do RANSAC (`ransacReprojThreshold`, `maxIters`, `confidence`, `refineIters`), o mínimo de matches (`< 4`) e o mínimo de keypoints (`> 4`).
- **Critério de aceitação:** cada parâmetro existe uma única vez no `Config`; nenhum literal de threshold permanece embutido nos módulos de etapa.

### ✅ H1-04 — Remover/isolar código morto e quebrado
- **Tipo:** `fato` · **Impacto:** Baixo · **Esforço:** Baixo
- **Problema:** código que não roda e confunde diagnóstico e manutenção.
- **Evidência:** `src/modules/canvas/create_geom.py:171` (`da.full` sem `import dask` → `NameError`); `src/modules/canvas/create_grid.py:64`; `src/modules/features/match/ransac.py:19` (artefatos de citação `[2, 6]` no meio do código); `src/modules/canvas/populate.py` (colagem em grade, legado).
- **Intervenção:** remover a função `create()` de `create_geom.py`; mover utilitários legados/experimentais para um diretório `legacy/` ou excluí-los, conforme decisão do mantenedor.
- **Critério de aceitação:** nenhum símbolo referenciado no fluxo aponta para código com import ausente; `create_geom.py` mantém apenas `create_blank_canvas_geom` + `main`.

### ✅ H1-05 — Fixar seed para reprodutibilidade do RANSAC
- **Tipo:** `recomendação` · **Impacto:** Baixo · **Esforço:** Baixo
- **Problema:** o RANSAC é estocástico e não há semente fixada, tornando execuções não determinísticas.
- **Evidência:** ausência de `random.seed`/`cv2.setRNGSeed` no repositório; uso do RANSAC em `src/modules/features/match/match.py:93`.
- **Intervenção:** fixar `cv2.setRNGSeed(...)` (e `np.random.seed(...)`) num ponto único de inicialização, com o valor exposto no `Config`.
- **Critério de aceitação:** duas execuções com os mesmos insumos produzem as mesmas matrizes de translação.

---

## Horizonte 2 — Refatorações estruturais

> Orquestração, configuração e contratos de dados. Reduzem acoplamento e custo de manutenção.

### ✅ H2-01 — Padronizar logging e observabilidade
- **Tipo:** `fato` · **Impacto:** Médio · **Esforço:** Baixo
- **Problema:** observabilidade inconsistente — alguns módulos usam `logging`, outros usam `print`.
- **Evidência:** `src/modules/features/match/match.py` e `src/modules/canvas/populate.py` usam `print`; `fetch`, `classifier`, `graph`, `geograph`, `globalpos` usam `logging`.
- **Intervenção:** substituir `print` por `logging` com formato/nível padronizados; centralizar a configuração de logging.
- **Critério de aceitação:** todas as etapas emitem logs pelo mesmo mecanismo, com nível configurável.

### ✅ H2-02 — Contratos de dados entre etapas (validação de schema)
- **Tipo:** `fato` · **Impacto:** Médio · **Esforço:** Médio
- **Problema:** a comunicação entre etapas é 100% por disco (pickle/zarr/json/npy/memmap) sem validação de esquema; leituras assumem estrutura implicitamente.
- **Evidência:** `src/modules/graph/geograph.py:165-166` lê `attrs["translation_matrix"]` sem checagem de versão/forma; leitura de `dataset.json`/`valid_tiles.json` sem schema.
- **Intervenção:** criar uma camada fina de I/O (uma função de leitura/escrita por artefato) que valide forma/versão e falhe cedo com mensagem clara.
- **Critério de aceitação:** artefato malformado/ausente gera erro explícito e localizado, não `KeyError`/`IndexError` genérico em etapa posterior.

### ✅ H2-03 — Otimizar I/O do matching
- **Tipo:** `fato` + `hipótese` · **Impacto:** Médio · **Esforço:** Baixo/Médio
- **Problema:** `match_pair` reabre o store zarr e reinstancia o matcher a cada par; cada tile tem seus keypoints relidos ~4× (uma vez por vizinho).
- **Evidência:** `src/modules/features/match/match.py:42` (matcher por par), `:48` (store por par), `load_keypoints_and_descriptors` chamado por par.
- **Intervenção:** abrir store e instanciar matcher uma vez por worker; cachear keypoints/descritores por tile (ex.: LRU) para evitar releitura.
- **Ganho esperado:** para grid N×N, reduzir de ~2N aberturas de store e ~4N leituras de keypoints para ~N leituras (≈ −75% de I/O na etapa).
- **Critério de aceitação:** número de aberturas de store por execução proporcional a workers/tiles, não a pares; tempo de matching reduzido em dataset de referência.

### ✅ H2-04 — Exportação do canvas por blocos (streaming)
- **Tipo:** `fato` · **Impacto:** Alto (mosaicos grandes) · **Esforço:** Médio
- **Problema:** a exportação materializa o canvas inteiro em RAM via `np.asarray(canvas)`, anulando o design memmap. Para grid 13×13 de tiles ~2000px, o pico é da ordem de ~2 GB.
- **Evidência:** `src/modules/canvas/populate_geom.py:150` (preview), `:167` (BigTIFF).
- **Intervenção:** escrever o BigTIFF por tiles/blocos com `tifffile`; gerar o preview por downsampling em streaming (sem carregar o mosaico inteiro).
- **Critério de aceitação:** pico de RAM na exportação passa a ser O(bloco), não O(mosaico); exportação funciona para canvas maior que a RAM disponível.

### ✅ H2-05 — Retomada incremental por etapa
- **Tipo:** `fato` · **Impacto:** Médio · **Esforço:** Médio
- **Problema:** etapas recomputam do zero; não há detecção de trabalho já feito.
- **Evidência:** `src/modules/features/detect/detect.py:113` (abre zarr em modo `"w"`); `src/modules/features/match/match.py:168` (recria diretório de matches).
- **Intervenção:** pular pares/tiles cujos artefatos já existem e estão íntegros (com flag `--force` para recomputar).
- **Critério de aceitação:** reexecutar `match`/`detect` após conclusão parcial processa apenas o que falta.

### ✅ H2-06 — Suíte de testes + CI
- **Tipo:** `fato` · **Impacto:** Alto · **Esforço:** Alto
- **Problema:** não há testes nem integração contínua.
- **Evidência:** ausência de `tests/` e de qualquer `import pytest`; ausência de `.github/`; linters já configurados em `pyproject.toml` (black/isort/flake8/interrogate).
- **Intervenção:**
  1. Criar fixture sintética (grid 3×3 de tiles gerados proceduralmente com sobreposição conhecida).
  2. Testes por etapa validando contrato de entrada/saída e um teste end-to-end sobre a fixture.
  3. CI executando lint + testes a cada push/PR.
- **Critério de aceitação:** `pytest` verde localmente e no CI; a fixture reconstrói o mosaico com erro de posição abaixo de um limiar definido.

---

## Horizonte 3 — Evolução

> Qualidade algorítmica, escala e qualidade visual. Após estabilizar orquestração e contratos.

### ⬜ H3-01 — Blending no preenchimento do canvas
- **Tipo:** `fato` · **Impacto:** Médio · **Esforço:** Médio/Alto
- **Problema:** a colagem é overwrite puro, sem tratamento de costuras.
- **Evidência:** `src/modules/canvas/populate_geom.py` (`paste_tiles_overwrite`).
- **Intervenção:** adicionar modo de blending opcional (feathering nas bordas; futuramente multiband), com parâmetro `mode='overwrite'|'blend'`.
- **Critério de aceitação:** zonas de sobreposição sem costura visível dura; modo overwrite permanece disponível.

### ⬜ H3-02 — Estender abstração de detector/matcher
- **Tipo:** `fato` · **Impacto:** Médio · **Esforço:** Médio
- **Problema:** as interfaces `FeatureDetector`/`FeatureMatcher` existem, mas os registries cobrem poucos algoritmos.
- **Evidência:** `src/modules/features/detect/registry.py` (só `sift`/`orb`); `src/modules/features/match/registry.py` (só `bf`).
- **Intervenção:** documentar e validar o ponto de extensão; adicionar suporte a matchers/detectores aprendidos (ex.: SuperPoint/LoFTR) como novos registros, sem alterar o fluxo.
- **Critério de aceitação:** trocar o detector/matcher por configuração (sem editar as etapas) funciona de ponta a ponta.

### ⬜ H3-03 — Métrica de fechamento de ciclo e diagnóstico de qualidade
- **Tipo:** `recomendação` · **Impacto:** Médio · **Esforço:** Médio
- **Problema:** há gating por resíduo, mas não uma métrica explícita de erro de fechamento de ciclo do grafo geométrico por execução.
- **Evidência:** solver e gating em `src/modules/graph/globalpos.py:419-423` e no loop IRLS subsequente.
- **Intervenção:** calcular e registrar, por execução, o erro médio de fechamento de ciclos e o resíduo global como métrica de qualidade.
- **Critério de aceitação:** relatório por execução com métricas comparáveis entre rodadas.

### ⬜ H3-04 — Escala do solver global
- **Tipo:** `hipótese` · **Impacto:** Médio · **Esforço:** Alto
- **Problema:** o solver não paraleliza internamente; grafos muito grandes podem custar.
- **Evidência:** `src/modules/graph/globalpos.py:419-423` (chamadas a `lsqr` por iteração, sem paralelismo).
- **Intervenção:** segmentar por clusters/sub-regiões e compor as transformações entre clusters; avaliar solvers esparsos alternativos.
- **Critério de aceitação:** tempo de resolução escala aceitavelmente para grafos com dezenas de milhares de arestas.

---

## Itens que exigem decisão de produto (aguardar aprovação)

- **Remoção definitiva de código legado** (`create_grid.py`, `populate.py`, `ransac.py`, `read.py`, `print_zarr_matches.py`, `draw_matches.py`, `draw_mosaic.py`): manter em `legacy/` ou excluir? (relaciona-se a H1-04)
- **Correção do `initialize_structure.py`** para criar `tiles/src` e `tiles/normalized`: alterar o initializer (mudança de código) ou apenas documentar? (relaciona-se a H1-02)
- **Formato de serialização dos grafos/posições**: manter `pickle` ou migrar para formato versionado (impacta compatibilidade de artefatos existentes).

---

## Checklist consolidado

**Horizonte 1**
- [x] H1-01 Propagar falhas no `fetch`
- [x] H1-02 Validar contrato de diretórios `src`/`normalized`
- [x] H1-03 Limpar duplicatas e centralizar magic numbers
- [x] H1-04 Remover/isolar código morto e quebrado
- [x] H1-05 Fixar seed do RANSAC

**Horizonte 2**
- [x] H2-01 Padronizar logging
- [x] H2-02 Contratos de dados com validação de schema
- [x] H2-03 Otimizar I/O do matching
- [x] H2-04 Exportação do canvas por blocos
- [x] H2-05 Retomada incremental por etapa
- [x] H2-06 Suíte de testes + CI

**Horizonte 3**
- [ ] H3-01 Blending no canvas
- [ ] H3-02 Estender abstração de detector/matcher
- [ ] H3-03 Métrica de fechamento de ciclo
- [ ] H3-04 Escala do solver global

---

## Rastreabilidade

Este plano deriva do diagnóstico arquitetural do repositório. As evidências citadas referem-se ao estado do código no momento da análise; ao executar cada ação, reconfirme as linhas, pois podem ter mudado.
