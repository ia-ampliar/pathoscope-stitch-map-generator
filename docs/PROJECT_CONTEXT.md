# PROJECT_CONTEXT.md

# 📌 CONTEXTO DO PROJETO — PIPELINE DE STITCHING GEOMÉTRICO

## Visão geral

Projeto de **stitching de tiles histopatológicos** usando **grafo geométrico** e **translações estimadas por keypoints**, com foco em modularidade, robustez e escalabilidade para mosaicos grandes.

O pipeline **já está funcional**, incluindo cálculo correto de posições globais e colagem dos tiles no canvas.

---

## Estrutura atual do pipeline

### 1️⃣ `match.py`

- Extrai keypoints e descritores dos tiles (armazenados em `.zarr`).
- Calcula correspondências entre tiles vizinhos.
- Usa `cv2.estimateAffinePartial2D` / RANSAC.
- Salva em cada `.zarr`:
    - `translation_matrix` (3×3 homogênea, apenas translação relevante).
- **Observação importante**:
    - O `dx, dy` salvo representa **mapeamento de pixels entre tiles**, **não** deslocamento físico no canvas.

---

### 2️⃣ `geograph.py`

- Constrói o **grafo geométrico dirigido** a partir do grafo topológico.
- Cada aresta armazena `(dx, dy)` como deslocamento entre tiles.

### 🔑 Correção crítica aplicada

Foi necessário **inverter o sinal da translação** para alinhar com o canvas:

```python
if direction =="A__B":
    dx, dy = -dx, -dy

```

Motivo:

- `match.py` estima transformação de coordenadas (pixel → pixel).
- O canvas precisa de deslocamento físico global.
- Sem essa inversão, o mosaico ficava desalinhado.

Após essa correção, o grafo geométrico passou a representar corretamente:

pos(v)=pos(u)+Δuv\text{pos}(v) = \text{pos}(u) + \Delta_{uv}

pos(v)=pos(u)+Δuv

---

### 3️⃣ `globalpos.py`

- Calcula **posições globais (X, Y)** dos tiles.
- Usa BFS (com `collections.deque`) a partir de um root escolhido.
- Propaga posições usando o grafo geométrico.
- Normaliza para coordenadas positivas.
- Salva resultado em:

```
output/result/global_positions.pkl

```

Formato:

```python
{ node: (X, Y) }

```

---

### 4️⃣ `create_geom.py`

- Cria um **canvas geométrico global** via `numpy.memmap`.
- Calcula bounding box global a partir das posições.
- Usa:
    - `CANVAS_CHUNK_SIZE`
    - `CANVAS_FILL_VALUE`
- Salva:
    - `canvas_geom.dat`
    - `canvas_geom_shape.npy`

---

### 5️⃣ `populate_geom.py`

- Abre o canvas geométrico.
- Carrega:
    - `global_positions.pkl`
    - `graph_geometric.gpickle`
- Resolve mapeamento:

```
node ->label -> arquivo de imagem (label.*)

```

- Cola tiles no canvas via **overwrite direto**.
- Exporta:
    - preview JPG
    - BigTIFF final

✅ Após a correção em `geograph.py`, o mosaico passou a ser colado **corretamente alinhado**.

---

## Problema atual identificado

Quando há tiles com **muito branco / pouca textura**:

- O matching gera poucos inliers.
- Algumas translações ficam ruins (ou próximas de zero).
- Essas arestas “podres” causam erro na propagação das posições globais.

Atualmente:

- Todas as arestas são usadas igualmente no BFS.
- Não há filtragem ou agregação de múltiplas estimativas.

---

## Próximo passo planejado

Tornar o cálculo das posições globais **mais robusto**, especialmente para tiles com pouco conteúdo.

Abordagem discutida e considerada ideal:

### ✅ Agregação por vizinhos confiáveis

Para um tile `v`:

- Cada vizinho `u` fornece uma estimativa:

p^v(u)=pu+Δuv\hat{p}_v^{(u)} = p_u + \Delta_{uv}

p^v(u)=pu+Δuv

- A posição final de `v` deve ser:
    - **mediana** das estimativas, ou
    - **média ponderada** (peso = inliers / inlier_ratio)

Antes disso:

- Filtrar arestas ruins (ex: `inliers < limiar`).
- Evitar usar arestas com peso zero.
- Preferir **não criar a aresta** a criar uma aresta inválida.

---

## Estado atual

✔️ Pipeline funcional

✔️ Geometria correta

✔️ Canvas colado corretamente

🔜 Robustez do grafo geométrico e cálculo de posições globais

---

### 👉 Instrução para nova conversa