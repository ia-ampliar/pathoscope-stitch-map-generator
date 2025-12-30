import zarr
import numpy as np

def ler_pontos_correspondencia(caminho_zarr):
    """
    Abre o arquivo Zarr e retorna os dados do array 'matches/matches'.
    """
    try:
        # Abre o store Zarr no modo de leitura ('r') [3, 4]
        # O zarr.open permite acessar o grupo raiz do diretório [5]
        root = zarr.open(caminho_zarr, mode='r')
        
        # Acessa o array específico dentro da hierarquia matches/matches [6, 7]
        # Como visto na sua estrutura anterior, o array está nesse caminho lógico
        z_array = root['matches/matches']
        
        # O fatiamento [:] carrega os dados do store para um array NumPy em memória [4, 8]
        # Isso é necessário para que as bibliotecas de visão computacional processem os pontos
        dados = z_array[:]
        
        print(f"Dados carregados com sucesso. Formato: {dados.shape}")
        return dados

    except KeyError:
        print("Erro: O caminho 'matches/matches' não foi encontrado no arquivo Zarr.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo: {e}")
        return None

# Exemplo de uso:
# pontos = ler_pontos_correspondencia('seu_arquivo.zarr')