import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anndata as ad
import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
import pandas as pd
import scipy.sparse as sp

# ─── Configuração ─────────────────────────────────────────────────────────────

CAMINHO_ENTRADA = Path(r"C:\Users\igorm\Documents\programacao\trabalho-cad-v2\dataset\pre-processado\dataset_filtrado.h5ad")
CAMINHO_SAIDA   = Path(r"C:\Users\igorm\Documents\programacao\trabalho-cad-v2\dados\clustering_all_genes.h5ad")
METRICAS_JSON   = Path(r"C:\Users\igorm\Documents\programacao\trabalho-cad-v2\dados\metricas_clustering.json")

CFG = {
    "n_clusters":  15,
    "batch_size":  5_000,    # paper recomenda 500–10000; 5k é seguro p/ 8 GB
    "max_iter":    2_000,    # teto de iterações de mini-batch
    "tol":         3e-4,     # shift relativo de Frobenius p/ considerar convergido
    "patience":    5,       # nº de checagens seguidas abaixo de tol p/ parar
    "eval_every":  1,        # checa convergência a cada N iterações
    "random_seed": 42,
}


# ─── Monitor ──────────────────────────────────────────────────────────────────

@dataclass
class Monitor:
    etapas:  list  = field(default_factory=list, init=False)
    _inicio: float = field(default_factory=time.perf_counter, init=False)
    _t0:     float = field(default_factory=time.perf_counter, init=False)

    def marca(self, nome: str) -> float:
        agora   = time.perf_counter()
        elapsed = agora - self._t0
        total   = agora - self._inicio
        try:
            free, tv = cp.cuda.runtime.memGetInfo()
            vram = f"VRAM {(tv-free)/1024**3:.2f}/{tv/1024**3:.1f}GB"
        except Exception:
            vram = ""
        self.etapas.append({"nome": nome, "elapsed_s": round(elapsed, 3), "total_s": round(total, 3)})
        print(f"  [{elapsed:7.2f}s | total {total:7.2f}s | {vram}] {nome}")
        self._t0 = agora
        return elapsed

    def imprimir(self):
        sep = "=" * 64
        print(f"\n{sep}\n  MÉTRICAS\n{sep}")
        for e in self.etapas:
            print(f"  {e['nome']:<44} {e['elapsed_s']:7.2f}s")
        if self.etapas:
            total = self.etapas[-1]["total_s"]
            print(f"  {'─'*52}")
            print(f"  {'TOTAL':<44} {total:7.2f}s  ({total/60:.2f} min)")
        print(sep)

    def salvar(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.etapas, indent=2, ensure_ascii=False))


# ─── Norms via reduceat — zero PCIe, O(NNZ) (igual à sua versão) ─────────────

def _norms_cpu(X: sp.csr_matrix, chunk_rows: int = 50_000) -> cp.ndarray:
    """||x_i||² direto de X.data via np.add.reduceat. O(NNZ), sem transfer."""
    n     = X.shape[0]
    norms = np.zeros(n, dtype=np.float32)
    print("  Norms via reduceat (CPU, sem PCIe)... ", end="", flush=True)
    t = time.perf_counter()
    for s in range(0, n, chunk_rows):
        e       = min(s + chunk_rows, n)
        nnz_s   = int(X.indptr[s]); nnz_e = int(X.indptr[e])
        data_sq = X.data[nnz_s:nnz_e] ** 2
        starts  = (X.indptr[s:e] - nnz_s).astype(np.intp)
        norms[s:e] = np.add.reduceat(data_sq, starts)
    print(f"{time.perf_counter()-t:.2f}s")
    return cp.asarray(norms)


class PinnedBuffers:
    def __init__(self, max_nnz: int, max_rows: int):
        self.max_nnz, self.max_rows = max_nnz, max_rows
        self._buf_data   = cp.cuda.alloc_pinned_memory(max_nnz * 4)
        self._buf_idx    = cp.cuda.alloc_pinned_memory(max_nnz * 4)
        self._buf_indptr = cp.cuda.alloc_pinned_memory((max_rows + 1) * 4)
        self.data   = np.frombuffer(self._buf_data,   dtype=np.float32)
        self.idx    = np.frombuffer(self._buf_idx,    dtype=np.int32)
        self.indptr = np.frombuffer(self._buf_indptr, dtype=np.int32)
        total_mb = (max_nnz * 8 + (max_rows + 1) * 4) / 1024**2
        print(f"  Pinned buffers alocados: {total_mb:.0f} MB page-locked")

    def transfer(self, batch_cpu: sp.csr_matrix, n_genes: int) -> cpsp.csr_matrix:
        nnz, b = batch_cpu.nnz, batch_cpu.shape[0]
        assert nnz <= self.max_nnz,  f"NNZ {nnz} > max_nnz {self.max_nnz}"
        assert b   <= self.max_rows, f"rows {b} > max_rows {self.max_rows}"
        data = batch_cpu.data;    data = data if data.dtype == np.float32 else data.astype(np.float32)
        idx  = batch_cpu.indices; idx  = idx  if idx.dtype  == np.int32   else idx.astype(np.int32)
        self.data[:nnz]   = data
        self.idx[:nnz]    = idx
        self.indptr[:b+1] = batch_cpu.indptr.astype(np.int32)
        cp_data   = cp.array(self.data[:nnz])     # DMA
        cp_idx    = cp.array(self.idx[:nnz])      # DMA
        cp_indptr = cp.array(self.indptr[:b+1])   # DMA
        return cpsp.csr_matrix((cp_data, cp_idx, cp_indptr), shape=(b, n_genes))


# ─── Assignment (distâncias) — batch fica esparso, só o (b,k) é denso ─────────

def _assign(batch_sp: cpsp.csr_matrix, C: cp.ndarray,
            bnorms: cp.ndarray, cnorms: cp.ndarray) -> cp.ndarray:
    # ||x-c||² = ||x||² - 2 x·c + ||c||²  →  argmin ignora ||x||² (constante por linha)
    XC    = batch_sp.dot(C.T)                       # (b,G)·(G,k) → (b,k) denso pequeno
    dists = bnorms[:, None] - 2.0 * XC + cnorms[None, :]
    return cp.argmin(dists, axis=1).astype(cp.int32)


# ─── Cluster-sums via one-hot ESPARSO (sem densificar o batch) ────────────────

def _cluster_sums(batch_sp: cpsp.csr_matrix, asgn: cp.ndarray, k: int):
    b = batch_sp.shape[0]
    # P (k×b) esparsa: P[c, j] = 1 se célula j foi atribuída ao cluster c
    P = cpsp.csr_matrix(
        (cp.ones(b, cp.float32), (asgn, cp.arange(b, dtype=cp.int32))),
        shape=(k, b),
    )
    cluster_sums = (P @ batch_sp).toarray()          # (k×b)·(b×G) → (k,G) denso ~1 MB
    counts       = cp.bincount(asgn, minlength=k).astype(cp.float32)
    return cluster_sums, counts


# ─── K-Means++ (igual à sua versão) ──────────────────────────────────────────

def _kmeans_pp_init(X: sp.csr_matrix, k: int, rng: np.random.Generator) -> cp.ndarray:
    n_sub   = min(10_000, X.shape[0])
    idx     = rng.choice(X.shape[0], size=n_sub, replace=False)
    sub_gpu = cp.asarray(np.asarray(X[idx].toarray(), dtype=np.float32))
    cents   = [sub_gpu[int(rng.integers(n_sub))]]
    print(f"  K-Means++ ({k} centroides, subset {n_sub:,})... ", end="", flush=True)
    t = time.perf_counter()
    for _ in range(1, k):
        C          = cp.stack(cents)
        cent_norms = cp.sum(C ** 2, axis=1)
        sub_norms  = cp.sum(sub_gpu ** 2, axis=1)
        dists      = sub_norms[:, None] - 2.0 * sub_gpu.dot(C.T) + cent_norms[None, :]
        cp.maximum(dists, 0.0, out=dists)
        min_d  = cp.asnumpy(dists.min(axis=1))
        probs  = min_d / (min_d.sum() + 1e-10)
        cents.append(sub_gpu[int(rng.choice(n_sub, p=probs))])
    print(f"{time.perf_counter()-t:.2f}s")
    return cp.stack(cents)        # FICA NA GPU (k,G)


# ─── Fit: mini-batch convergente ──────────────────────────────────────────────

def fit_minibatch(
    X: sp.csr_matrix, k: int, batch_size: int, rng: np.random.Generator,
    max_iter: int, tol: float, patience: int, eval_every: int,
    monitor: Optional[Monitor] = None,
):
    n_cells, n_genes = X.shape
    _m = monitor.marca if monitor else (lambda s: None)

    cell_norms = _norms_cpu(X);                                   _m("Norms (CPU reduceat)")
    C          = _kmeans_pp_init(X, k, rng)                       # (k,G) GPU
    v          = cp.zeros(k, dtype=cp.float64)                    # contagens acumuladas
    _m("Centroides K-Means++")

    max_row_nnz = int(np.diff(X.indptr).max())                   # worst-case real
    pinned = PinnedBuffers(max_nnz=batch_size * max_row_nnz, max_rows=batch_size)
    _m("Pinned buffers alocados")

    print(f"\n  Mini-batch convergente — b={batch_size:,}, max_iter={max_iter}, "
          f"tol={tol:.0e}, patience={patience}\n")

    ewa = None
    t_loop = time.perf_counter()
    for t in range(1, max_iter + 1):
        idx       = rng.choice(n_cells, size=batch_size, replace=False)  # sem reposição
        batch_sp  = pinned.transfer(X[idx], n_genes)
        bnorms    = cp.asarray(cell_norms[idx])
        cnorms    = cp.sum(C ** 2, axis=1)

        asgn      = _assign(batch_sp, C, bnorms, cnorms)
        cs, bc    = _cluster_sums(batch_sp, asgn, k)

        # ── Update agregado de Sculley (tudo na GPU) ─────────────────────────
        v        += bc.astype(cp.float64)
        lr        = (bc / cp.maximum(v.astype(cp.float32), 1.0))          # (k,) lr por centro
        bm        = cs / cp.maximum(bc, 1.0)[:, None]                     # (k,G) média do batch
        C_old     = C
        C         = C + (lr[:, None] * (bm - C)).astype(cp.float32)

        # ── Convergência: shift relativo de Frobenius ────────────────────────
        if t % eval_every == 0:
            shift = float(cp.linalg.norm(C - C_old) / (cp.linalg.norm(C_old) + 1e-12))
            alpha = min(1.0, 5.0 * batch_size / n_cells)        # suaviza ~ N/(5b) iters
            ewa   = shift if ewa is None else (1 - alpha) * ewa + alpha * shift
            if t % 50 == 0:
                print(f"  iter {t:4d} | shift {shift:.3e} | ewa {ewa:.3e}")
            if ewa < tol:
                print(f"\n  ✓ Convergiu (EWA) em {t} (ewa={ewa:.3e})")
                break

        del batch_sp, asgn, cs
    else:
        print(f"\n  ⚠ Atingiu max_iter={max_iter} sem convergir (shift={shift:.3e})")

    print(f"  Loop: {time.perf_counter()-t_loop:.1f}s")
    _m("Mini-batch convergido")

    # ── Assignment final em TODAS as células (streaming) ─────────────────────
    print("  Assignment final... ", end="", flush=True)
    t_f    = time.perf_counter()
    cnorms = cp.sum(C ** 2, axis=1)
    labels = np.empty(n_cells, dtype=np.int32)
    for s in range(0, n_cells, batch_size):
        e        = min(s + batch_size, n_cells)
        batch_sp = pinned.transfer(X[s:e], n_genes)
        bnorms   = cell_norms[s:e]
        labels[s:e] = cp.asnumpy(_assign(batch_sp, C, bnorms, cnorms))
        del batch_sp
        cp.get_default_memory_pool().free_all_blocks()
    print(f"{time.perf_counter()-t_f:.2f}s")
    _m("Assignment final")

    return labels, cp.asnumpy(C)


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main():
    monitor = Monitor()
    sep = "=" * 64
    print(f"\n{sep}\n  CLUSTERING ALL GENES — Mini-Batch CONVERGENTE (CuPy)\n{sep}\n")
    try:
        free, total = cp.cuda.runtime.memGetInfo()
        print(f"  VRAM: {total/1024**3:.1f} GB | livre: {free/1024**3:.1f} GB\n")
    except Exception:
        pass

    print(f"  Carregando {CAMINHO_ENTRADA.name}...")
    adata = ad.read_h5ad(CAMINHO_ENTRADA)
    monitor.marca("Dataset carregado")

    X = adata.X
    if not sp.issparse(X):           X = sp.csr_matrix(X)
    if not isinstance(X, sp.csr_matrix): X = X.tocsr()
    X = X.astype(np.float32, copy=False)
    print(f"  Shape: {X.shape[0]:,} × {X.shape[1]:,} | NNZ: {X.nnz:,}\n")
    monitor.marca("Matriz CSR float32")

    rng = np.random.default_rng(CFG["random_seed"])
    labels, _ = fit_minibatch(
        X=X, k=CFG["n_clusters"], batch_size=CFG["batch_size"], rng=rng,
        max_iter=CFG["max_iter"], tol=CFG["tol"], patience=CFG["patience"],
        eval_every=CFG["eval_every"], monitor=monitor,
    )

    unique, cnts = np.unique(labels, return_counts=True)
    print(f"\n  Distribuição dos {CFG['n_clusters']} clusters:")
    bar_max = cnts.max()
    for c, n in zip(unique, cnts):
        bar = "█" * max(1, int(25 * n / bar_max))
        print(f"    {c:2d}: {n:>8,}  ({100*n/len(labels):5.1f}%)  {bar}")

    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    adata.obs["cluster_all_genes"] = pd.Categorical(labels.astype(str))
    adata.uns["mbkmeans_cfg"]      = {k: str(v) for k, v in CFG.items()}
    adata.write_h5ad(CAMINHO_SAIDA)
    monitor.marca("Resultado salvo")
    monitor.salvar(METRICAS_JSON)
    monitor.imprimir()
    print(f"\n  → {CAMINHO_SAIDA}\n")


if __name__ == "__main__":
    main()