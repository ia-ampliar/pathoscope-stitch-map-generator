# PROJECT_CONTEXT.md

## Visão Geral

Este projeto implementa um pipeline robusto para **reconstrução geométrica (stitching)** de mosaicos de imagens histopatológicas a partir de tiles sobrepostos, utilizando:

- correspondência visual entre tiles adjacentes,
- modelagem do problema como um **grafo geométrico de restrições relativas**,
- resolução global por **mínimos quadrados robustos (IRLS + Huber)**,
- e colagem final em um **canvas global memmap**, exportável em BigTIFF.

O pipeline foi projetado para ser **determinístico, auditável e extensível**, priorizando estabilidade geométrica e qualidade visual do mosaico final.

---

## Estado Atual do Pipeline (Resumo)

✅ **Costura geométrica correta**

✅ **Sem artefatos visíveis entre tiles**

✅ **Sem deslocamentos espúrios em cantos ou bordas**

✅ **Pipeline estável para grids completos (ex.: 13×13 = 169 tiles)**

O problema residual observado anteriormente em tiles de borda foi **eliminado** após a introdução de **normalização fotométrica prévia das imagens**, evidenciando que a falha não era geométrica, mas **de qualidade de matching visual**.

---

## Arquitetura do Pipeline

### 1. Pré-processamento de Imagens (`preprocesser.py`)

Antes de qualquer matching, os tiles passam por um estágio de **normalização fotométrica**, cujo objetivo é:

- reduzir variações de iluminação,
- equalizar contraste local,
- facilitar detecção de keypoints consistentes,
- aumentar a confiabilidade dos deslocamentos estimados (`dx`, `dy`).

> 🔑 **Decisão chave do projeto**
> 
> 
> A normalização ocorre **antes** da geração do grafo geométrico.
> 
> Todo o pipeline subsequente assume que as imagens já estão normalizadas.
> 

Esse estágio foi decisivo para:

- eliminar mismatches em áreas de fundo,
- reduzir outliers extremos,
- estabilizar a solução global sem necessidade de heurísticas adicionais agressivas.

---

### 2. Geração do Grafo Geométrico (`geograph.py` / `create_geom.py`)

Cada tile é representado como um **nó** identificado por `(row, col)`.

As arestas representam **restrições geométricas relativas** entre tiles vizinhos:

```
x_v - x_u ≈ dx
y_v - y_u ≈ dy
```

Cada aresta carrega:

- `dx`, `dy` — deslocamento estimado entre tiles,
- `weight` — confiança do match (derivada da qualidade dos keypoints),
- metadados (arquivo `.zarr`, ids, etc.).

### Filtros aplicados na criação do grafo

Durante a leitura dos matches:

- ❌ deslocamentos fisicamente impossíveis são descartados (`|shift| > MAX_SHIFT`),
- ❌ deslocamentos nulos (`dx ≈ 0 && dy ≈ 0`) são descartados,
- ✔ apenas vizinhança válida é considerada.

Esses filtros são **hard gates geométricos** e não são revertidos posteriormente.

---

### 3. Resolução Global das Posições (`globalpos.py`)

O grafo geométrico é resolvido por um solver global robusto:

- **IRLS (Iteratively Reweighted Least Squares)**,
- **loss de Huber** para atenuar outliers,
- âncora fixa (`root`) para eliminar grau de liberdade global.

### Características do solver

- resolve simultaneamente **todos os ciclos do grafo**,
- reduz drift acumulado típico de BFS,
- usa pesos base + pesos robustos (Huber),
- permite *gating* por resíduo após iterações iniciais.

### Diagnóstico

O solver produz logs detalhados contendo:

- estatísticas de resíduos (mean, median, p90, p95, max),
- número de arestas atenuadas,
- nós inalcançáveis (se existirem),
- verificação da posição do root,
- offset global aplicado para normalização positiva.

Esses logs são parte fundamental do processo de validação.

---

### 4. Gating Pós-Solver (Controle de Qualidade)

Após a convergência inicial:

- arestas com **resíduo acima de um limiar físico (ex.: 200 px)** podem ser removidas,
- o solver é reexecutado com o subconjunto consistente.

> ⚠️ Importante
> 
> 
> Tentativas de impor regras como “grau mínimo por nó” mostraram-se **desnecessárias** após a normalização das imagens e foram **abandonadas** como regra obrigatória.
> 

O pipeline atual confia prioritariamente em:

- qualidade fotométrica,
- pesos de match,
- robustez do solver global.

---

### 5. Colagem no Canvas Global (`populate_geom.py`)

As posições globais são usadas para colar os tiles em um **canvas geométrico memmap**, criado previamente:

- escrita direta em `np.memmap` (baixo uso de RAM),
- estratégia **overwrite** (sem blending),
- arredondamento de coordenadas globais,
- recorte automático em bordas.

O resultado pode ser exportado como:

- preview JPG,
- BigTIFF RGB (compatível com visualizadores WSI).

---

## Lições Aprendidas (Decisões Importantes)

### ✔ Normalização > Heurísticas Geométricas

O principal erro de costura **não era geométrico**, mas causado por:

- baixa textura,
- fundo homogêneo,
- variações de iluminação.

A normalização:

- aumentou a estabilidade dos keypoints,
- reduziu outliers extremos,
- eliminou a necessidade de regras artificiais no grafo.

### ✔ Solver global funciona quando os dados são bons

O IRLS + Huber mostrou-se suficiente quando:

- os matches são razoáveis,
- os pesos refletem qualidade real,
- não há viés sistemático nas imagens.

---

## Estado Atual: Conclusão

📌 O pipeline encontra-se em um **estado estável e correto**, com:

- arquitetura clara,
- responsabilidades bem separadas,
- comportamento previsível,
- resultados visuais de alta qualidade.

Próximos passos naturais (opcionais):

- versionar parâmetros de normalização,
- salvar métricas de matching por tile,
- adicionar validação automática por densidade de keypoints,
- integrar com WSI viewers (OME-TIFF / pyramidal).