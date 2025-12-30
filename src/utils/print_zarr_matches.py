import zarr
import time
import os

from src.modules.features.match.read import ler_pontos_correspondencia
from src.modules.features.match.ransac import transformar_e_salvar_zarr

def imprimir_estrutura_zarr(caminho):
    """
    Abre um arquivo Zarr e imprime sua estrutura hierárquica.
    """
    try:
        # Abre o store Zarr no modo de leitura ('r') [1]
        # O zarr.open é capaz de identificar se o caminho é um grupo ou array [3]
        dados = zarr.open(caminho, mode='r')
        
        print(f"--- Explorando o arquivo: {caminho} ---")
        
        # O método .tree() exibe visualmente a organização de grupos e arrays [1, 2]
        # Isso é útil para estruturas complexas como a sua (matches/matches) [4, 5]
        print(dados.tree())
        
        # Opcional: Mostra informações técnicas como compressão e formato [6, 7]
        print("\n--- Informações Adicionais ---")
        print(dados.info)

    except FileNotFoundError:
        print(f"Erro: O caminho '{caminho}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo Zarr: {e}")

def main():

    caminho_do_arquivo = 'output/matches/00001_x1_y1_zp1__00002_x2_y1_zp1.zarr'

    # Ler pontos de correspondência usando a função existente
    dados = ler_pontos_correspondencia(caminho_do_arquivo)

    # Aplica RANSAC e salva a transformação em um novo arquivo Zarr
    if dados is not None:
        caminho_saida = 'output/ransac/transformacao_afim_ransac.zarr'
        transformar_e_salvar_zarr(dados, caminho_saida)

    # Defina aqui o caminho para o seu arquivo .zarr
    # Usei o exemplo da estrutura de pastas que você forneceu anteriormente
    base_dir = 'output/ransac'
    if not os.path.exists(base_dir):
        print(f"O diretório '{base_dir}' não existe.")
        return
    else:
        dirs_zarr = [f for f in os.listdir(base_dir) if f.endswith('.zarr')]
        caminho_do_arquivo = os.path.join(base_dir, dirs_zarr[1]) if dirs_zarr else None
    
    # Chama a função de inspeção
    imprimir_estrutura_zarr(caminho_do_arquivo)

# Bloco padrão para garantir que o código só execute se o script for chamado diretamente
if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")