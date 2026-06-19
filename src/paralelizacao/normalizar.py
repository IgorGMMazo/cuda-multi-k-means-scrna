"""
scran_gpu.py — Normalização por deconvolution (scran) COMPLETA em CuPy/CUDA
===========================================================================

Roda 100% na GPU via CuPy. Toma o dataset filtrado + os labels de cluster
(coluna `cluster_all_genes` gerada pelo clustering) e produz size factors por
célula, gravando counts log-normalizados.

Lógica validada em NumPy contra size factors conhecidos:
  - 1 cluster      : corr 0.99, erro ~6%   (vs ~23% library-size)
  - multi-cluster  : corr 0.997, erro ~2.6% (vs ~15% library-size)

A mediana-por-pool está em CuPy (vetorizada). Quando o kernel .cu
`pool_size_factors` estiver validado, é só trocar `median_pools` por ele —
o resto do pipeline não muda.

Fluxo por bloco de cluster (<= MAX_BLOCK células):
  prescaling → ordena por lib → referência → cumsum (P) → pools deslizantes
  → mediana(pool/ref) → resolve A·beta=R (lsqr GPU) → theta → pseudo-célula
Depois: reescala entre blocos, centraliza em média 1, aplica log1p(counts/sf).
"""

import json
import time
from pathlib import Path

import anndata as ad
import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
import pandas as pd
import scipy.sparse as sp
try:
    from cupyx.scipy.sparse.linalg import lsqr as gpu_lsqr
except Exception:
    gpu_lsqr = None        # versões antigas de CuPy: cai p/ equações normais

# ─── Config ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]   # raiz do repositório (trabalho-cad-v2/)
CAMINHO_ENTRADA = BASE / "dataset" / "clusterizado-inicial" / "clustering_all_genes.h5ad"
CAMINHO_SAIDA   = BASE / "dataset" / "normalizado" / "normalizado_scran.h5ad"
METRICAS_JSON   = BASE / "metricas-etapas" / "metricas_norm.json"
COL_CLUSTER     = "cluster_all_genes"

CFG = {
    "max_block": 3_000,                  # scran quebra clusters grandes nisso p/ o solve
    "sizes":     list(range(21, 102, 5)),# janelas do pooling (padrão scran p/ UMI)
    "min_mean":  0.1,                    # filtro de genes de baixa expressão
    "min_cells": 50,                     # bloco menor que isso → library-size
    "pool_chunk": 4_000,                 # pools por vez na mediana (controla VRAM)
}


# ─── Monta os pools de um bloco: starts/widths + incidência COO ─────────────────
def build_pools(nb: int, sizes):
    starts = [np.arange(nb, dtype=np.int32)]
    widths = [np.ones(nb, np.int32)]
    coo_r  = [np.arange(nb, dtype=np.int32)]      # pool i ← célula i (eq. por célula)
    coo_c  = [np.arange(nb, dtype=np.int32)]
    pid = nb
    for w in sizes:
        if w > nb:
            continue
        s    = np.arange(nb, dtype=np.int32)
        cols = (s[:, None] + np.arange(w, dtype=np.int32)[None, :]) % nb   # (nb, w) circular
        rows = (pid + s)[:, None].repeat(w, 1)
        starts.append(s); widths.append(np.full(nb, w, np.int32))
        coo_r.append(rows.ravel()); coo_c.append(cols.ravel())
        pid += nb
    return (np.concatenate(starts), np.concatenate(widths),
            np.concatenate(coo_r), np.concatenate(coo_c))


# ─── Mediana por pool (CuPy vetorizado). P[0]=0 → não-wrap soma 0 no termo extra ─
def median_pools(P, ref, starts, widths, nb, chunk):
    npools = int(starts.shape[0])
    R = cp.empty(npools, cp.float32)
    for a in range(0, npools, chunk):
        b   = min(a + chunk, npools)
        st  = starts[a:b]
        en  = st + widths[a:b]
        wrap = en > nb
        en_eff = cp.minimum(en, nb)
        extra  = cp.where(wrap, en - nb, 0)               # P[0]=0 p/ não-wrap
        pooled = P[en_eff] - P[st] + P[extra]             # (chunk, g')
        R[a:b] = cp.median(pooled / ref[None, :], axis=1)
    return R


# ─── Mínimos quadrados na GPU, robusto a versões do CuPy ────────────────────────
def solve_lsq(A, R, nb):
    if gpu_lsqr is not None:
        try:
            sol = gpu_lsqr(A, R)
            return sol[0] if isinstance(sol, tuple) else sol
        except Exception:
            pass
    # fallback: equações normais (AᵀA·beta = AᵀR) — robusto p/ nb <= 3000
    AtA = (A.T @ A).toarray()
    Atb = A.T @ R
    return cp.linalg.solve(AtA + 1e-6 * cp.eye(nb, dtype=cp.float32), Atb)


# ─── Deconvolution de um bloco → size factors (centrados em 1) + pseudo-célula ──
def deconv_block(Bg, sizes, min_mean, pool_chunk):
    nb, g = Bg.shape
    lib   = Bg.sum(1)
    t     = lib / lib.mean()
    Z     = Bg / t[:, None]

    order = cp.argsort(lib)
    Zo    = Z[order]
    ref   = Zo.mean(0)
    mask  = ref >= min_mean
    Zo, refm = Zo[:, mask], ref[mask]

    zeros = cp.zeros((1, int(mask.sum())), cp.float32)
    P     = cp.concatenate([zeros, cp.cumsum(Zo, axis=0)], 0)   # (nb+1, g')

    starts, widths, coo_r, coo_c = build_pools(nb, sizes)
    starts_d = cp.asarray(starts); widths_d = cp.asarray(widths)
    R = median_pools(P, refm, starts_d, widths_d, nb, pool_chunk)

    A = cpsp.csr_matrix(
        (cp.ones(coo_r.shape[0], cp.float32), (cp.asarray(coo_r), cp.asarray(coo_c))),
        shape=(int(starts.shape[0]), nb))
    beta = solve_lsq(A, R, nb)                                 # mínimos quadrados na GPU

    theta = cp.empty(nb, cp.float32)
    theta[order] = beta.astype(cp.float32) * t[order]
    theta = cp.maximum(theta, 1e-8)
    theta /= theta.mean()

    pseudo = (Bg / theta[:, None]).mean(0)                     # pseudo-célula (genes completos)
    return theta, pseudo


# ─── Pipeline ───────────────────────────────────────────────────────────────────
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

    print("=" * 64, "\n  NORMALIZAÇÃO SCRAN (deconvolution) — CuPy/CUDA\n", "=" * 64, sep="")

    adata = ad.read_h5ad(CAMINHO_ENTRADA)
    marca("Dataset carregado")

    X = adata.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    X = X.astype(np.float32, copy=False)
    n, g = X.shape
    labels = adata.obs[COL_CLUSTER].to_numpy()
    print(f"  {n:,} células × {g:,} genes | {len(np.unique(labels))} clusters\n")
    marca("Matriz CSR float32")

    # blocos: cada cluster fatiado em <= max_block células
    blocks = []
    for c in np.unique(labels):
        cells = np.where(labels == c)[0]
        for s in range(0, len(cells), CFG["max_block"]):
            blocks.append(cells[s:s + CFG["max_block"]])
    print(f"  {len(blocks)} blocos (<= {CFG['max_block']} células cada)\n")

    sf      = np.empty(n, np.float32)
    pseudos = []
    block_cells = []
    for bi, cells in enumerate(blocks):
        Bg = cp.asarray(X[cells].toarray())               # (nb, g) na GPU
        if len(cells) < CFG["min_cells"]:                 # bloco pequeno → library-size
            lib = Bg.sum(1); theta = lib / lib.mean()
            pseudo = (Bg / cp.maximum(theta, 1e-8)[:, None]).mean(0)
        else:
            theta, pseudo = deconv_block(Bg, CFG["sizes"], CFG["min_mean"], CFG["pool_chunk"])
        sf[cells] = cp.asnumpy(theta)
        pseudos.append(pseudo); block_cells.append(cells)
        del Bg
        cp.get_default_memory_pool().free_all_blocks()
        if (bi + 1) % 25 == 0 or bi + 1 == len(blocks):
            print(f"    bloco {bi+1}/{len(blocks)}")
    marca("Deconvolution por bloco")

    # reescala entre blocos: tudo relativo ao maior bloco
    ref_idx = int(np.argmax([len(c) for c in block_cells]))
    ref_ps  = pseudos[ref_idx]
    for cells, ps in zip(block_cells, pseudos):
        m = (ps > 0) & (ref_ps > 0)
        rescale = float(cp.median(ps[m] / ref_ps[m]))
        sf[cells] *= rescale
    sf /= sf.mean()
    marca("Reescala entre blocos")

    # aplica log1p(counts / sf) IN-PLACE, em chunks (sem duplicar a matriz)
    inv = (1.0 / sf).astype(np.float32)
    CH = 20_000
    for s in range(0, n, CH):
        e = min(s + CH, n)
        a, b = int(X.indptr[s]), int(X.indptr[e])
        rows = np.repeat(np.arange(s, e, dtype=np.int32), np.diff(X.indptr[s:e + 1]))
        X.data[a:b] *= inv[rows]
        np.log1p(X.data[a:b], out=X.data[a:b])
        del rows
    marca("log1p(counts/sf) aplicado")

    adata.X = X
    adata.obs["size_factors"] = sf
    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(CAMINHO_SAIDA)
    METRICAS_JSON.write_text(json.dumps(etapas, indent=2, ensure_ascii=False))
    marca("Resultado salvo")

    print(f"\n  size factors: média={sf.mean():.4f}  mediana={np.median(sf):.4f}  "
          f"min={sf.min():.3f}  max={sf.max():.3f}")
    print(f"  → {CAMINHO_SAIDA}")


if __name__ == "__main__":
    main()