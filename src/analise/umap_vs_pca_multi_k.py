"""
umap_vs_pca_multi_k.py — versao em loop do umap_vs_pca.py.

Em vez de um unico k=15, varre K de 1 a 20 e gera, para cada K, a figura
lado a lado (PC1xPC2 | UMAP dos 50 PCs) com as MESMAS celulas.

Ponto importante de desempenho: o embedding UMAP NAO depende de K
(ele so projeta os 50 PCs -> 2D). Por isso ele e calculado UMA unica vez,
fora do loop, e reaproveitado em todos os K — o que muda a cada K e apenas
a coloracao (os labels do k-means). Recomputar o UMAP a cada K so repetiria
a parte cara sem necessidade.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import umap

SEED = 42
N_SUB = 60_000
K_MIN, K_MAX = 1, 20          # varredura de K (inclusive)
OUTDIR = r"metricas-etapas/umap_multi_k"

os.makedirs(OUTDIR, exist_ok=True)

print("[1/4] carregando 50 PCs...")
X = np.load(r"dataset/pca/pca.npy").astype(np.float32)
n = X.shape[0]
print(f"      {n:,} celulas x {X.shape[1]} PCs")

# --- espectro: fracao de variancia ENTRE os 50 PCs (plano => isotropico) ---
var = X.var(0)
frac = var / var.sum()
print("[2/4] fracao de variancia (entre os 50 PCs):")
print("      PC1..PC6:", np.round(frac[:6], 4))
print(f"      PC1+PC2 = {frac[:2].sum()*100:.1f}%  (de 50 PCs);  "
      f"top10 = {frac[:10].sum()*100:.1f}%")

# --- subamostra fixa (mesmas celulas em todos os K) ---
rng = np.random.default_rng(SEED)
idx = rng.choice(n, N_SUB, replace=False)
Xs = X[idx]

# --- UMAP calculado UMA vez (independe de K) e reaproveitado no loop ---
print(f"[3/4] UMAP de {N_SUB:,} celulas (50 PCs -> 2D), calculado uma unica vez...")
emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean",
                random_state=SEED).fit_transform(Xs)

print(f"[4/4] varrendo K de {K_MIN} a {K_MAX}...")
cmap = "tab20"
for K in range(K_MIN, K_MAX + 1):
    km = MiniBatchKMeans(n_clusters=K, batch_size=10_000,
                         random_state=SEED, n_init=3)
    labels = km.fit_predict(X)
    ls = labels[idx]

    fig, (axp, axu) = plt.subplots(1, 2, figsize=(16, 7))
    axp.scatter(Xs[:, 0], Xs[:, 1], c=ls, cmap=cmap, s=2, alpha=0.5, linewidths=0)
    axp.set_title(f"PC1 x PC2 (o que voces plotaram)\nk={K} clusters", fontweight="bold")
    axp.set_xlabel("PC1"); axp.set_ylabel("PC2")
    axu.scatter(emb[:, 0], emb[:, 1], c=ls, cmap=cmap, s=2, alpha=0.5, linewidths=0)
    axu.set_title(f"UMAP dos 50 PCs (o que o paper plotou, Fig.5A)\nk={K} clusters",
                  fontweight="bold")
    axu.set_xlabel("UMAP-1"); axu.set_ylabel("UMAP-2")
    fig.suptitle(f"Mesmas celulas, mesmos clusters, lentes diferentes  (k={K})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    out = os.path.join(OUTDIR, f"umap_vs_pca_k{K:02d}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      k={K:2d} -> {out}")

print("OK -> figuras em", OUTDIR)
