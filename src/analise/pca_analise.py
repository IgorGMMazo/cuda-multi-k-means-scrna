import numpy as np
import matplotlib.pyplot as plt

def plotar_pca(caminho_arquivo):
    # 1. Carregar os dados do arquivo .npy
    dados_pca = np.load(caminho_arquivo)
    
    print(f"Formato original dos dados: {dados_pca.shape}")
    
    # Validação simples para garantir que temos pelo menos 2 dimensões
    if dados_pca.shape[1] < 2:
        print("Erro: Os dados precisam ter pelo menos 2 componentes para um plot 2D.")
        return

    # 2. Extrair a Primeira e a Segunda Componente Principal (PC1 e PC2)
    # Pegamos todas as linhas (amostras) das colunas 0 e 1
    pc1 = dados_pca[:, 0]
    pc2 = dados_pca[:, 1]

    # 3. Configurar e gerar o gráfico
    plt.figure(figsize=(10, 6))
    
    # Criando o gráfico de dispersão (scatter plot)
    plt.scatter(pc1, pc2, alpha=0.7, edgecolors='w', linewidth=0.5)

    # Estilização
    plt.title('Visualização do PCA (2D)', fontsize=14, fontweight='bold')
    plt.xlabel('Componente Principal 1 (PC1)', fontsize=12)
    plt.ylabel('Componente Principal 2 (PC2)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Exibir o plot
    plt.tight_layout()
    plt.show()

# Executando a função com o caminho que aparece na sua imagem
plotar_pca(r'C:\Users\igorm\Documents\programacao\trabalho-cad-v2\dataset\pca\pca.npy')