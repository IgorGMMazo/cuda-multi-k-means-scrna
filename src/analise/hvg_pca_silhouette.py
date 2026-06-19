"""
hvg_pca_silhouette.py — Fecha a brecha do revisor: e se selecionarmos HVG antes do PCA?

Compara DUAS construcoes do espaco de 50 PCs:
  A) TODOS os genes  -> reaproveita o pca.npy de voces (baseline do artigo)
  B) top-2000 HVG    -> seleciona genes variaveis (scanpy/Seurat) e refaz o PCA

Para cada espaco, roda k-means (k=2 pico do artigo, e k=15 do paper) e mede
silhouette amostrado. Gera tambem UMAP lado a lado (todos os genes vs HVG).
Se o silhouette continuar baixo no espaco HVG, a conclusao de nao-clusterabilidade
nao e artefato de usar todos os genes.
"""
import numpy as np
import scipy.sparse as sp
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import umap

SEED, N_SUB, N_SIL, N_HVG = 42, 60_000, 5_000, 2_000
KS = [2, 15]
NORM = r"dataset/normalizado/normalizado_scran.h5ad"
ALLPCA = r"dataset/pca/pca.npy"

def hvg_seurat(X, n_top, n_bins=20, bs=50_000):
    """HVG estilo Seurat/scanpy em chunks: momentos de expm1(log) sem densificar.
    dispersion=var/mean nas contagens normalizadas; z-score por bin de media; top-N."""
    n, g = X.shape
    s1 = np.zeros(g, np.float64)   # sum de expm1(data)
    s2 = np.zeros(g, np.float64)   # sum de expm1(data)^2
    for s in range(0, n, bs):
        Xb = X[s:s+bs]
        e = np.expm1(Xb.data.astype(np.float64))            # so nos nao-zeros
        s1 += np.asarray(sp.csr_matrix((e,    Xb.indices, Xb.indptr), shape=Xb.shape).sum(0)).ravel()
        s2 += np.asarray(sp.csr_matrix((e*e,  Xb.indices, Xb.indptr), shape=Xb.shape).sum(0)).ravel()
    mean = s1 / n
    var  = s2 / n - mean**2
    mean_safe = np.where(mean == 0, 1e-12, mean)
    disp = var / mean_safe
    disp[disp == 0] = np.nan
    disp = np.log(disp)
    lmean = np.log1p(mean)
    # bins por media; normaliza dispersao dentro do bin (scanpy 'seurat')
    bins = np.minimum((n_bins * np.argsort(np.argsort(lmean)) // g), n_bins-1)
    norm_disp = np.full(g, np.nan)
    for b in range(n_bins):
        m = bins == b
        d = disp[m]
        mu_b, sd_b = np.nanmean(d), np.nanstd(d)
        if not np.isfinite(sd_b) or sd_b == 0: sd_b = 1.0
        norm_disp[m] = (disp[m] - mu_b) / sd_b
    order = np.argsort(np.nan_to_num(norm_disp, nan=-np.inf))[::-1]
    mask = np.zeros(g, bool); mask[order[:n_top]] = True
    return mask


def covariance_pca(X, k=50, bs=20_000):
    """PCA so-centralizacao (igual pca-gpu.py), em CPU: M=XtX-n*mu mu^T, eigh, projeta."""
    n, p = X.shape
    S = np.zeros((p, p), np.float64); colsum = np.zeros(p, np.float64)
    for s in range(0, n, bs):
        Xb = X[s:s+bs]
        S += (Xb.T @ Xb).toarray().astype(np.float64)
        colsum += np.asarray(Xb.sum(0)).ravel()
    mu = colsum / n
    M = S - n * np.outer(mu, mu)
    w, V = np.linalg.eigh(M)
    Vk = V[:, -k:][:, ::-1].astype(np.float32)
    muV = (mu.astype(np.float32) @ Vk)
    scores = np.empty((n, k), np.float32)
    for s in range(0, n, bs):
        scores[s:s+bs] = X[s:s+bs] @ Vk - muV[None, :]
    return scores

def sil_at(scores, k, rng):
    km = MiniBatchKMeans(n_clusters=k, batch_size=10_000, random_state=SEED, n_init=3)
    lab = km.fit_predict(scores)
    idx = rng.choice(len(scores), N_SIL, replace=False)
    s = silhouette_score(scores[idx], lab[idx], metric="euclidean")
    return float(s), lab

print("[1/6] carregando normalizado (log1p)...")
adata = ad.read_h5ad(NORM)
X = adata.X.tocsr().astype(np.float32)
print(f"      {X.shape[0]:,} celulas x {X.shape[1]:,} genes")

print(f"[2/6] selecionando top-{N_HVG} HVG (Seurat, em chunks)...")
mask = hvg_seurat(X, N_HVG)
print(f"      {int(mask.sum())} genes HVG selecionados")
Xh = X[:, mask]

print("[3/6] PCA (50 PCs) no espaco HVG...")
scores_h = covariance_pca(Xh, k=50)
var_h = scores_h.var(0); frac_h = var_h / var_h.sum()
print(f"      HVG: PC1+PC2={frac_h[:2].sum()*100:.1f}%  top10={frac_h[:10].sum()*100:.1f}%")

print("[4/6] carregando baseline todos-os-genes (pca.npy)...")
scores_a = np.load(ALLPCA).astype(np.float32)

print("[5/6] silhouette comparado (todos vs HVG)...")
rng = np.random.default_rng(SEED)
res = {}
lab_a15 = lab_h15 = None
for name, sc_ in [("todos_genes", scores_a), ("hvg_2000", scores_h)]:
    res[name] = {}
    for k in KS:
        s, lab = sil_at(sc_, k, np.random.default_rng(SEED))
        res[name][k] = s
        if name == "todos_genes" and k == 15: lab_a15 = lab
        if name == "hvg_2000" and k == 15: lab_h15 = lab

print("\n  ==================== SILHOUETTE ====================")
print(f"  {'espaco':>14} | {'k=2':>8} | {'k=15':>8}")
print("  " + "-"*40)
for name in res:
    print(f"  {name:>14} | {res[name][2]:>8.4f} | {res[name][15]:>8.4f}")
print("  ====================================================\n")

print("[6/6] UMAP lado a lado (todos vs HVG), k=15...")
idx = rng.choice(X.shape[0], N_SUB, replace=False)
emb_a = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(scores_a[idx])
emb_h = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(scores_h[idx])
fig, (axa, axh) = plt.subplots(1, 2, figsize=(16, 7))
axa.scatter(emb_a[:,0], emb_a[:,1], c=lab_a15[idx], cmap="tab20", s=2, alpha=.5, linewidths=0)
axa.set_title(f"UMAP todos os genes (13897)\nk=15  sil={res['todos_genes'][15]:.3f}", fontweight="bold")
axh.scatter(emb_h[:,0], emb_h[:,1], c=lab_h15[idx], cmap="tab20", s=2, alpha=.5, linewidths=0)
axh.set_title(f"UMAP top-2000 HVG\nk=15  sil={res['hvg_2000'][15]:.3f}", fontweight="bold")
fig.suptitle("Selecao de HVG muda a separabilidade?", fontsize=14, fontweight="bold")
fig.tight_layout()
out = r"metricas-etapas/hvg_vs_allgenes_umap.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"      salvo -> {out}")
