"""
pca_gpu.py — PCA na GPU por covariância (CuPy + cuSOLVER)
=========================================================

Espelha o paper EXATAMENTE para comparação justa: todos os genes, apenas
centralização (sem z-score), top 50 PCs. Substitui o IRLBA out-of-core (96h)
por covariância gene×gene + eigendecomposição na GPU.

Por que isto é rápido: com n_células (1,1M) >> n_genes (13.897), o PCA sai da
matriz de covariância p×p (~772 MB, cabe na VRAM), não de Lanczos iterativo
relendo a matriz do disco. Dois passes lineares pelos dados:
  Passe 1: acumula S = XᵀX (esparso, por batch) e as somas de coluna → média
  M = S - n·μμᵀ  (= (X-μ)ᵀ(X-μ), sem densificar X)
  eigh(M) → top 50 autovetores = loadings
  Passe 2: scores = (X - μ)·V  (streaming)

Validado: idêntico ao sklearn PCA (corr 1.000000 nos 50 PCs).
"""

import json
import time
from pathlib import Path

import anndata as ad
import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
import scipy.sparse as sp
from cupyx.scipy.sparse.linalg import eigsh

BASE = Path(__file__).resolve().parents[2]   # raiz do repositório (trabalho-cad-v2/)
CAMINHO_ENTRADA = BASE / "dataset" / "normalizado" / "normalizado_scran.h5ad"
SAIDA_SCORES    = BASE / "dataset" / "pca" / "pca.npy"   # arquivo (antes apontava p/ o diretório)
METRICAS_JSON   = BASE / "metricas-etapas" / "metricas_pca.json"

CFG = {"n_components": 50, "batch_size": 7_000}


def main():
    etapas, t_ini, t0 = [], time.perf_counter(), time.perf_counter()
    def marca(nome):
        nonlocal t0
        el = time.perf_counter() - t0
        etapas.append({"nome": nome, "elapsed_s": round(el, 3),
                       "total_s": round(time.perf_counter() - t_ini, 3)})
        free, tot = cp.cuda.runtime.memGetInfo()
        print(f"  [{el:7.2f}s | VRAM {(tot-free)/1024**3:.2f}/{tot/1024**3:.1f}GB] {nome}")
        t0 = time.perf_counter()

    print("=" * 64, "\n  PCA GPU (covariância + cuSOLVER) — 50 PCs, só centralização\n", "=" * 64, sep="")

    adata = ad.read_h5ad(CAMINHO_ENTRADA)
    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    X = X.astype(np.float32, copy=False)
    n, p = X.shape

    # guard: dado de entrada não pode ter NaN/Inf (propagariam p/ todos os PCs)
    nan_x = int(np.isnan(X.data).sum()); inf_x = int(np.isinf(X.data).sum())
    if nan_x > 0 or inf_x > 0:
        raise ValueError(f"Dado de entrada corrompido: {nan_x} NaN, {inf_x} Inf em X.data. "
                         "Re-execute o scran_gpu.py para regenerar o normalizado_scran.h5ad.")
    k, bs = CFG["n_components"], CFG["batch_size"]
    print(f"  {n:,} células × {p:,} genes | {k} PCs\n")
    marca("Dataset carregado")

    # ── Passe 1: S = XᵀX (esparso, por batch) + somas de coluna ──────────────────
    S      = cp.zeros((p, p), dtype=cp.float64)     # ~1.5 GB
    colsum = cp.zeros(p, dtype=cp.float64)
    for s in range(0, n, bs):
        e  = min(s + bs, n)
        Xb = cpsp.csr_matrix(X[s:e])                # batch b×p na GPU
        prod = (Xb.T.tocsr() @ Xb)                  # esparso p×p (XᵀX do batch)
        S += prod.toarray().astype(cp.float64)
        colsum += Xb.sum(axis=0).ravel()
        del Xb, prod
        cp.get_default_memory_pool().free_all_blocks()
    mu = colsum / n
    marca("Passe 1: covariancia XtX")

    # ── M = (X-μ)ᵀ(X-μ) = S - n·μμᵀ ; eigh float64 → top-k loadings ────────────
    # eigh garante ordem ASCENDENTE: w[-1] é o maior autovalor.
    # Usamos V[:, -k:] diretamente (sem argsort) — mais estável com espectro plano.
    M = S - n * cp.outer(mu, mu)                     # float64, (p, p)
    del S
    cp.get_default_memory_pool().free_all_blocks()
    w, V = cp.linalg.eigh(M)                         # w em ordem ascendente
    wk = cp.ascontiguousarray(w[-k:][::-1])          # top-k descrescente
    Vk = cp.ascontiguousarray(V[:, -k:][:, ::-1].astype(cp.float32))   # (p, k)
    # sinal determinístico (maior |loading| positivo, igual sklearn)
    signs = cp.sign(Vk[cp.argmax(cp.abs(Vk), axis=0), cp.arange(k)])
    Vk *= signs[None, :]
    bad = int(cp.isnan(Vk).sum() + cp.isinf(Vk).sum())
    if bad > 0:
        raise ValueError(f"Eigendecomposicao produziu {bad} NaN/Inf nos loadings.")
    del M, V, w
    cp.get_default_memory_pool().free_all_blocks()
    marca("Eigendecomposicao (eigh float64, top-k)")

    # ── Passe 2: scores = (X - μ)·V  (streaming) ─────────────────────────────────
    muV    = (mu.astype(cp.float32) @ Vk)            # (k,)
    scores = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, bs):
        e  = min(s + bs, n)
        Xb = cpsp.csr_matrix(X[s:e])
        scores[s:e] = cp.asnumpy(Xb @ Vk - muV[None, :])
        del Xb
        cp.get_default_memory_pool().free_all_blocks()
    # pós-ordenação por variância empírica — rede de segurança p/ espectro plano
    # (autovalores próximos podem ter ordenação instável; a variância empírica é a verdade)
    var_pc  = scores.var(0)
    ord_var = np.argsort(var_pc)[::-1]
    scores  = np.ascontiguousarray(scores[:, ord_var])
    print(f"  var por PC (top-6): {np.round(scores.var(0)[:6], 4)}")
    marca("Passe 2: projecao + reordenacao")

    # ── Saída ────────────────────────────────────────────────────────────────────
    SAIDA_SCORES.parent.mkdir(parents=True, exist_ok=True)
    np.save(SAIDA_SCORES, scores)
    var = cp.asnumpy(wk / (n - 1))
    METRICAS_JSON.write_text(json.dumps(
        {"etapas": etapas, "explained_variance": var.tolist(),
         "var_ratio_top10": (var[:10] / var.sum()).tolist()}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    marca("Scores salvos")

    print(f"\n  scores: {scores.shape}  | var. explicada PC1..PC3: "
          f"{var[0]:.3f}, {var[1]:.3f}, {var[2]:.3f}")
    print(f"  → {SAIDA_SCORES}")


if __name__ == "__main__":
    main()
