"""
umap_vs_pca_hvg.py — versao HVG do umap_vs_pca.py, estilo Fig.5A do paper.

Mesma ideia (PC1xPC2 vs UMAP dos 50 PCs, mesmas celulas, mesma cor), mas
sobre o espaco PCA construido em cima do HVG (dataset/pca_hvg/pca.npy) em
vez do PCA com todos os genes. A coloracao usa o clustering final REAL do
pipeline (clustering-finalv2.py rodado sobre pca_hvg/pca.npy), lido direto
de metricas-etapas/hvg/labels_multi_k.npz — nao um k-means rapido novo.

O paper (Hicks et al. 2021, Fig. 5A) usa K=15; o nosso sweep de K no HVG
so cobre valores pares (2..30), entao usamos K=16 (o mais proximo ja
calculado) em vez de rodar um k-means a parte so pra bater o numero exato.

Renderizacao: a Fig.5A do paper NAO e um scatter — e um HEXBIN ("Hexbin
plot of the UMAP representation of the 1.3 million cells, color coded by
the clusters found via mbkmeans"), feito com o pacote R `schex`
(Freytag & Lister, 2020) especificamente pra evitar overplotting num
scatter de mais de 1M pontos. Cada hexagono agrega varias celulas e e
colorido pelo cluster MAJORITARIO daquele bin — isso "derrete" a fronteira
ruidosa entre clusters em regioes lisas, bem diferente de um scatter de
pontos individuais semi-transparentes. Reproduzimos isso aqui com
matplotlib.hexbin + reduce_C_function=moda.

Pre-requisito: dataset/pca_hvg/pca.npy e metricas-etapas/hvg/labels_multi_k.npz
precisam ser da versao CORRIGIDA do pca-gpu.py (covariancia em chunks
float64 — ver nota de precisao no topo de src/paralelizacao/pca-gpu.py).
A versao anterior (matmul unico) sofria cancelamento catastrofico e dava
PC1+PC2=7.6% em vez de 36.4%.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

SEED = 42
N_SUB = 60_000
K = 16   # mais proximo do K=15 do paper dentre os K's ja rodados no HVG (K_VALUES pares)
GRIDSIZE = 70

PCA_HVG    = r"dataset/pca_hvg/pca.npy"
LABELS_HVG = r"metricas-etapas/hvg/labels_multi_k.npz"
OUT        = r"metricas-etapas/hvg/umap_vs_pca_hvg_k16_hexbin.png"


def moda(a):
    """Cluster majoritario de um bin hexagonal (equivalente ao 'color coded by
    the clusters' do schex: cada hexagono mostra 1 cor, a mais frequente nele).
    matplotlib passa `a` como lista python, nao array."""
    return np.bincount(np.asarray(a, dtype=np.int64)).argmax()


print("[1/5] carregando 50 PCs (HVG, corrigido)...")
X = np.load(PCA_HVG).astype(np.float32)
n = X.shape[0]
print(f"      {n:,} celulas x {X.shape[1]} PCs")

# --- espectro: fracao de variancia ENTRE os 50 PCs ---
var = X.var(0)
frac = var / var.sum()
print("[2/5] fracao de variancia (entre os 50 PCs, HVG):")
print("      PC1..PC6:", np.round(frac[:6], 4))
print(f"      PC1+PC2 = {frac[:2].sum()*100:.1f}%  (de 50 PCs);  "
      f"top10 = {frac[:10].sum()*100:.1f}%")

print(f"[3/5] carregando clustering final ja rodado (k={K}) de {LABELS_HVG}...")
labels = np.load(LABELS_HVG)[f"k{K}"]
assert len(labels) == n, "labels_multi_k.npz nao bate com pca_hvg/pca.npy (rode clustering-finalv2.py de novo)"

rng = np.random.default_rng(SEED)
idx = rng.choice(n, N_SUB, replace=False)
Xs, ls = X[idx], labels[idx]

print(f"[4/5] UMAP de {N_SUB:,} celulas (50 PCs HVG -> 2D)...")
emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean",
                random_state=SEED).fit_transform(Xs)

print("[5/5] plotando lado a lado (hexbin, estilo Fig.5A / schex)...")
fig, (axp, axu) = plt.subplots(1, 2, figsize=(16, 7))
cmap = "tab20"

hb1 = axp.hexbin(Xs[:, 0], Xs[:, 1], C=ls, gridsize=GRIDSIZE,
                  reduce_C_function=moda, cmap=cmap, vmin=0, vmax=K - 1)
axp.set_title(f"PC1 x PC2 (HVG)\nk={K} clusters", fontweight="bold")
axp.set_xlabel("PC1"); axp.set_ylabel("PC2")

hb2 = axu.hexbin(emb[:, 0], emb[:, 1], C=ls, gridsize=GRIDSIZE,
                  reduce_C_function=moda, cmap=cmap, vmin=0, vmax=K - 1)
axu.set_title(f"UMAP dos 50 PCs (HVG) - hexbin estilo Fig.5A do paper\nk={K} clusters",
              fontweight="bold")
axu.set_xlabel("UMAP-1"); axu.set_ylabel("UMAP-2")

fig.suptitle("HVG + PCA (corrigido) + clustering final: mesmas celulas, mesmos clusters, lentes diferentes",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"      salvo -> {OUT}")
