"""
investiga_variancia_hvg.py — por que PC1+PC2 caiu de 36.4% (exploratorio,
hvg_pca_silhouette.py) para 7.6% (producao, pca-gpu.py)?

Hipotese: os DOIS scripts fazem S = X^T X em float32 (dtype nativo do
h5ad) e so depois convertem pra float64. A diferenca e COMO somam:
  - pca-gpu.py:            UM matmul so sobre as 1.148.558 celulas inteiras,
                            acumulado internamente em float32 pelo cuSPARSE,
                            e SO NO FINAL convertido pra float64.
  - hvg_pca_silhouette.py: soma p/ CHUNKS de 20k linhas, cada chunk convertido
                            pra float64 e acumulado ANTES do proximo chunk.
Isso muda a ordem de grandeza do erro de arredondamento acumulado em S ANTES
da subtracao M = S - n*mu*mu^T (centralizacao) — que e exatamente o tipo de
operacao (diferenca de dois numeros grandes e parecidos) que amplifica erro
de arredondamento (cancelamento catastrofico).

Este script usa o MESMO hvg_mask.npy ja salvo pela pca-gpu.py (mesmos genes,
elimina a hipotese "genes diferentes") e recalcula a covariancia em CHUNKS
com acumulacao float64 (igual ao script exploratorio, so que em cima do
mask real de producao). Se a fracao de variancia voltar pra perto de ~36%,
confirma que o bug e de PRECISAO NUMERICA no matmul unico da pca-gpu.py,
nao selecao de genes diferente.
"""
import time
import numpy as np
import scipy.sparse as sp
import anndata as ad
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
NORM = BASE / "dataset" / "normalizado" / "normalizado_scran.h5ad"
HVG_MASK = BASE / "dataset" / "pca_hvg" / "hvg_mask.npy"

BUILD_BS = 100_000   # leitura do disco, blocos grandes (igual pca-gpu.py)
COV_BS   = 20_000    # chunk da soma S += (Xb.T@Xb) em float64 (igual ao script exploratorio)
K = 50

t0 = time.perf_counter()
print("[1/5] abrindo dataset (backed) e mask HVG ja salvo pela pca-gpu.py...")
adata = ad.read_h5ad(NORM, backed="r")
n, p_full = adata.shape
mask = np.load(HVG_MASK)
p = int(mask.sum())
print(f"      {n:,} celulas x {p_full:,} genes -> {p:,} HVG (mask reaproveitado, MESMOS genes da producao)")

print("[2/5] montando matriz HVG completa em RAM (blocos grandes, so leitura)...")
blocos = []
for s in range(0, n, BUILD_BS):
    e = min(s + BUILD_BS, n)
    Xb = adata.X[s:e]
    Xb = Xb if sp.issparse(Xb) else sp.csr_matrix(np.asarray(Xb))
    blocos.append(Xb.tocsr()[:, mask])
X = sp.vstack(blocos, format="csr")
del blocos
print(f"      X_hvg: {X.shape}, {X.data.nbytes/1024**3:.2f} GB, dtype={X.dtype}  ({time.perf_counter()-t0:.1f}s)")

print(f"[3/5] covariancia S=X^T X EM CHUNKS de {COV_BS:,} linhas (acumulado em float64 a cada chunk)...")
S = np.zeros((p, p), np.float64)
colsum = np.zeros(p, np.float64)
for s in range(0, n, COV_BS):
    Xb = X[s:s+COV_BS]
    S += (Xb.T @ Xb).toarray().astype(np.float64)
    colsum += np.asarray(Xb.sum(0)).ravel()
mu = colsum / n
M = S - n * np.outer(mu, mu)
print(f"      pronto ({time.perf_counter()-t0:.1f}s)")

print("[4/5] eigh (float64, top-k) + projecao...")
w, V = np.linalg.eigh(M)
wk = w[-K:][::-1]
Vk = V[:, -K:][:, ::-1].astype(np.float32)
muV = (mu.astype(np.float32) @ Vk)
scores = np.empty((n, K), np.float32)
for s in range(0, n, COV_BS):
    scores[s:s+COV_BS] = X[s:s+COV_BS] @ Vk - muV[None, :]
print(f"      pronto ({time.perf_counter()-t0:.1f}s)")

print("[5/5] fracao de variancia (mesmo metodo usado pra medir 7.6% e 36.4%)...")
var = scores.var(0)
frac = var / var.sum()
print(f"      PC1..PC6:  {np.round(frac[:6], 4)}")
print(f"      PC1+PC2  = {frac[:2].sum()*100:.1f}%")
print(f"      top10    = {frac[:10].sum()*100:.1f}%")
print(f"\n      eigenvalues wk (top-6, chunked-float64): {np.round(wk[:6], 4)}")
print(f"      (compare com wk/(n-1) do metricas_pca_hvg.json atual, escala ~1e-13 -> bug)")

np.save(BASE / "metricas-etapas" / "hvg" / "scores_diagnostico_chunked.npy", scores)
print(f"\nTOTAL: {time.perf_counter()-t0:.1f}s")
