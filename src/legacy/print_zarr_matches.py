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

        # Acessa o grupo 'matches' e imprime seus atributos
        mg = dados["matches"]
        print("\n--- attrs do grupo 'matches' ---")
        print(dict(mg.attrs))

        if "matches" in mg:
            arr = mg["matches"]
            print("\n--- array matches/matches ---")
            print("shape:", arr.shape, "dtype:", arr.dtype)

        
        # Opcional: Mostra informações técnicas como compressão e formato [6, 7]
        print("\n--- Informações Adicionais ---")
        print(dados.info)

    except FileNotFoundError:
        print(f"Erro: O caminho '{caminho}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo Zarr: {e}")

# Função principal para explorar a estrutura dos arquivos .zarr
def main():
    # Definir o diretório onde os arquivos .zarr estão localizados
    base_dir = 'output/matches'  # Diretório onde os arquivos .zarr estão
    dirs_zarr = [f for f in os.listdir(base_dir) if f.endswith('.zarr')]

    if not dirs_zarr:
        print("Nenhum arquivo .zarr encontrado no diretório especificado.")
        return

    # Aqui estamos pegando o primeiro arquivo .zarr encontrado no diretório
    caminho_do_arquivo = os.path.join(base_dir, dirs_zarr[0])
    print(f"Inspecionando o arquivo .zarr: {caminho_do_arquivo}")
    
    # Chamar a função para imprimir a estrutura do arquivo .zarr
    imprimir_estrutura_zarr(caminho_do_arquivo)

# Bloco padrão para garantir que o código só execute se o script for chamado diretamente
if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")