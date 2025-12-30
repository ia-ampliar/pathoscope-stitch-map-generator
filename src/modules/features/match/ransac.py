import zarr
import cv2
import numpy as np

def transformar_e_salvar_zarr(dados_matches, caminho_saida):
    """
    Aplica RANSAC para calcular a transformação afim e salva o resultado em Zarr.
    """
    # 1. Preparação dos pontos (Source e Destination)
    # Assumindo que os dados lidos são (N, 4) -> [x1, y1, x2, y2]
    src_pts = dados_matches[:, :2].astype(np.float32).reshape(-1, 1, 2) [2, 6]
    dst_pts = dados_matches[:, 2:].astype(np.float32).reshape(-1, 1, 2) [6]

    # 2. Aplicação do RANSAC para Transformação Afim
    # Nota: Fora das fontes, o cv2.estimateAffine2D é usado para garantir H[7]=0, H[3, 7]=0, H[7]=1.
    # As fontes descrevem o RANSAC via cv2.findHomography para modelos projetivos [2, 6].
    
    # Usando o conceito de RANSAC para remover outliers conforme as fontes [4, 5]:
    matrix_afim, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    if matrix_afim is not None:
        # Criar a matriz 3x3 completa com a última linha fixa [3]
        h_matrix = np.eye(3)
        h_matrix[:2, :] = matrix_afim
        
        # 3. Salvar os resultados utilizando Zarr
        # Zarr permite criar arrays com dtypes específicos do NumPy [1, 8]
        try:
            # Criar um novo store no disco [8, 9]
            # O zarr.create_array escreve imediatamente no disco [10]
            z = zarr.create_array(
                store=caminho_saida, 
                shape=h_matrix.shape, 
                chunks=h_matrix.shape, 
                dtype='f8', 
                overwrite=True
            ) [8, 11]
            
            # Atribuição dos dados (escrita) [12, 13]
            z[:] = h_matrix 
            
            # Adicionar metadados (atributos) sobre a transformação [12, 14]
            z.attrs['tipo_transformacao'] = 'Afim (RANSAC)' [12]
            z.attrs['outliers_removidos'] = int(np.sum(mask == 0))
            
            print(f"Transformação salva com sucesso em: {caminho_saida}")
            print("Matriz calculada:\n", h_matrix)
            
        except Exception as e:
            print(f"Erro ao salvar no Zarr: {e}")
    else:
        print("Erro: Não foi possível estimar a transformação.")

# Exemplo de fluxo completo
# 1. Ler dados (função criada anteriormente)
# 2. Chamar: transformar_e_salvar_zarr(meus_dados, 'resultado_transformacao.zarr')