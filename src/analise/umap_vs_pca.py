"""
umap_vs_pca.py — Demonstra que o "borrão" da Fig. 2 (PC1xPC2) vs. as "ilhas"
da Fig. 5A do paper (UMAP) é diferença de LENTE, não de dados.

Gera, nas MESMAS células e MESMA coloração de clusters (k=15, como o paper):
  - esquerda : projeção PC1 x PC2  (o que vocês plotaram)
  - direita  : UMAP dos 50 PCs     (o que o paper plotou)
E imprime a fração de variância dos 50 PCs (espectro plano = sem estrutura).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import umap

SEED = 42
N_SUB = 60_000
K = 15

print("[1/5] carregando 50 PCs...")
X = np.load(r"dataset/pca/pca.npy").astype(np.float32)
n = X.shape[0]
print(f"      {n:,} celulas x {X.shape[1]} PCs")

# --- espectro: fracao de variancia ENTRE os 50 PCs (plano => isotropico) ---
var = X.var(0)
frac = var / var.sum()
print("[2/5] fracao de variancia (entre os 50 PCs):")
print("      PC1..PC6:", np.round(frac[:6], 4))
print(f"      PC1+PC2 = {frac[:2].sum()*100:.1f}%  (de 50 PCs);  "
      f"top10 = {frac[:10].sum()*100:.1f}%")

print(f"[3/5] k-means k={K} (MiniBatch) sobre os 50 PCs...")
km = MiniBatchKMeans(n_clusters=K, batch_size=10_000, random_state=SEED, n_init=3)
labels = km.fit_predict(X)

rng = np.random.default_rng(SEED)
idx = rng.choice(n, N_SUB, replace=False)
Xs, ls = X[idx], labels[idx]

print(f"[4/5] UMAP de {N_SUB:,} celulas (50 PCs -> 2D)...")
emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean",
                random_state=SEED).fit_transform(Xs)

print("[5/5] plotando lado a lado...")
fig, (axp, axu) = plt.subplots(1, 2, figsize=(16, 7))
cmap = "tab20"
axp.scatter(Xs[:, 0], Xs[:, 1], c=ls, cmap=cmap, s=2, alpha=0.5, linewidths=0)
axp.set_title(f"PC1 x PC2 (o que voces plotaram)\nk={K} clusters", fontweight="bold")
axp.set_xlabel("PC1"); axp.set_ylabel("PC2")
axu.scatter(emb[:, 0], emb[:, 1], c=ls, cmap=cmap, s=2, alpha=0.5, linewidths=0)
axu.set_title(f"UMAP dos 50 PCs (o que o paper plotou, Fig.5A)\nk={K} clusters",
              fontweight="bold")
axu.set_xlabel("UMAP-1"); axu.set_ylabel("UMAP-2")
fig.suptitle("Mesmas celulas, mesmos clusters, lentes diferentes", fontsize=14,
             fontweight="bold")
fig.tight_layout()
out = r"metricas-etapas/umap_vs_pca.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"      salvo -> {out}")
