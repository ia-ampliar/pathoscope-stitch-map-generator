# Stitch map generator

Cria um mapa com os tiles dispostos num canva seguindo a ordem de posicinamento do mosaico.

## 1. Configurando o repositório

Após realizar o git clone do repositório, crie uma venv e com a venv atividada, execute o pip install passando como parâmetro o requirements.txt. Para ambiente de dev, passe o requirements-dev.txt

### Cria o ambiente virtual

```bash
python -m venv venv
```
crie o ambiente com uma versão do python acima da 3.10. Por exemplo

```bash
py -3.11 -m venv venv
```

### Ativa o ambiente virtual

```bash
.\venv\Scripts\activate
```

### Atualiza o pip da venv

```bash
python -m pip install --upgrade pip
```

### Instala as dependências

```bash
python -m pip install -r .\requirements\requirements.txt
```

## 2. Iniciando o projeto (Cria estrutura de diretórios)

Para criar a estrutura de diretórios necessária para a execução do pipeline execute o módulo initializer

```bash
python -m src.modules.initializer.initialize_structure
```

Após a criação da estrutura de diretórios, insira os tiles em `output/tiles`

## 3. Normalizaço dos tiles

Para realizar a normalização dos tiles execute o módulo tile.preprocessing.preprocessing

```bash
python -m src.modules.tile.preprocessing.preprocesser
```

## 4. Extração de metadados dos tiles

Para realizar a extração dos dados dos tiles (nome, caminho e coordenadas), execute o módulo tile.fetch

```bash
python -m src.modules.tile.fetch.fetch
```

## 5. Classificação dos tiles candidatos

Para realizar a classificação dos tiles candidatos, execute o módulo tile_classifier

```bash
python -m src.modules.tile.classify.classifier
```

## 6. Geração do Grafo Topológico

Para criar a imagem do grafo com nós e arestas indicando o posicionamento dos tiles assim como candidatos ou não, execute o módulo

```bash
python -m src.modules.graph.graph
```

## 7. Detecção de features (keypoints e descriptors)

Para realizar a detecção de features no dataset dos tiles (somente nos candidatos), execute o módulo

```bash
python -m src.modules.features.detect.detect
```

## 8. Realização dos Matches entre tiles

Para realizar o match entre os tiles execute o módulo

```bash
python -m src.modules.features.match.match
```

## 9. Geração do Grafo Geométrico

Para gerar o grafo geométrico bidericional execute o módulo

```bash
python -m src.modules.graph.geograph
```

## 10. Calcula as posições globais

Para calcular as posições globais a partir do grafo geométrico bidericional execute o módulo

```bash
python -m src.modules.graph.globalpos
```

## 11. Criação do Canvas

Para realizar a criação do canvas em branco (para ser preenchido posteriormente) execute o módulo

```bash
python -m src.modules.canvas.create_geom
```

## 12. Preenchimento do Canvas

Para inserir os tiles no canvas execute o módulo

```bash
python -m src.modules.canvas.populate_geom
```

