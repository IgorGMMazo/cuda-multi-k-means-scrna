import os
import gc
import time
import json
import threading

import h5py
import cupy as cp
import scipy.sparse as sp
import numpy as np
import numba as nb
from numba import njit, prange
import scanpy as sc
import pandas as pd
import psutil

# ══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════

def configurar_threads() -> int:
    logical  = os.cpu_count()
    physical = psutil.cpu_count(logical=False)
    n        = min(logical, physical + 6)
    os.environ["OMP_NUM_THREADS"]      = str(n)
    os.environ["MKL_NUM_THREADS"]      = str(n)
    os.environ["NUMBA_NUM_THREADS"]    = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    print(f"  Threads : {n}  (logical={logical}, physical={physical})")
    return n

N_THREADS = configurar_threads()

cp.cuda.set_pinned_memory_allocator(None)
cp.get_default_memory_pool().set_limit(size=6 * 1024**3)

CHUNK_SIZE    = 100_000
MIN_GENES     = 500
MIN_CELLS     = 3
MAX_PCT_MT    = 5.0
MT_PREFIXES   = ("MT-", "mt-")
MEAN_EXPR_MIN = 0.0027


# ══════════════════════════════════════════════════════════════
# MONITOR DE MÉTRICAS
# ══════════════════════════════════════════════════════════════

class MetricsMonitor:
    def __init__(self, sample_interval: float = 0.5):
        self._interval   = sample_interval
        self._process    = psutil.Process(os.getpid())
        self._ram_samples: list[float] = []
        self._timestamps:  list[float] = []
        self._marks:       list[tuple]  = []
        self._running      = False
        self._thread       = None

    def _sample_loop(self):
        while self._running:
            rss = self._process.memory_info().rss / (1024 ** 3)
            self._ram_samples.append(rss)
            self._timestamps.append(time.perf_counter())
            time.sleep(self._interval)

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        self.mark("inicio")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self.mark("fim")

    def mark(self, label: str):
        self._marks.append((label, time.perf_counter()))

    def _stage_times(self) -> dict[str, float]:
        stages = {}
        for i in range(1, len(self._marks)):
            label_prev, t_prev = self._marks[i - 1]
            label_curr, t_curr = self._marks[i]
            key = f"{label_prev} → {label_curr}"
            stages[key] = round(t_curr - t_prev, 3)
        return stages

    def _ram_stats(self) -> dict:
        if not self._ram_samples:
            return {}
        arr = np.array(self._ram_samples)
        return {
            "pico_gb":    round(float(arr.max()),  2),
            "media_gb":   round(float(arr.mean()), 2),
            "minimo_gb":  round(float(arr.min()),  2),
            "n_amostras": len(arr),
            "intervalo_s": self._interval,
        }

    def _total_time(self) -> float:
        if len(self._marks) < 2:
            return 0.0
        return round(self._marks[-1][1] - self._marks[0][1], 3)

    def _build_report(self) -> dict:
        return {
            "total_segundos":  self._total_time(),
            "etapas_segundos": self._stage_times(),
            "ram":             self._ram_stats(),
        }

    def salvar(self, path_txt: str, path_json: str):
        report = self._build_report()
        ram    = report["ram"]
        stages = report["etapas_segundos"]
        total  = report["total_segundos"]

        linhas = [
            "=" * 60,
            "  RELATÓRIO DE PERFORMANCE — PRÉ-PROCESSAMENTO",
            "=" * 60,
            "",
            f"  Tempo total           : {total:.2f}s  ({total/60:.1f} min)",
            "",
            "  Tempo por etapa:",
        ]
        for label, secs in stages.items():
            linhas.append(f"    {label:<45} {secs:>8.2f}s")

        linhas += [
            "",
            "  RAM (amostrada a cada {:.1f}s):".format(ram.get("intervalo_s", 0)),
            f"    Pico    : {ram.get('pico_gb',  0):.2f} GB",
            f"    Média   : {ram.get('media_gb', 0):.2f} GB",
            f"    Mínimo  : {ram.get('minimo_gb',0):.2f} GB",
            f"    Amostras: {ram.get('n_amostras', 0)}",
            "",
            "=" * 60,
        ]

        os.makedirs(os.path.dirname(path_txt) or ".", exist_ok=True)
        with open(path_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(linhas))

        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n  Métricas salvas:")
        print(f"    {path_txt}")
        print(f"    {path_json}")

    def imprimir(self):
        report = self._build_report()
        ram    = report["ram"]
        print("\n" + "=" * 60)
        print("  MÉTRICAS DE PERFORMANCE")
        print("=" * 60)
        print(f"  Tempo total : {report['total_segundos']:.2f}s")
        print("\n  Por etapa:")
        for label, secs in report["etapas_segundos"].items():
            print(f"    {label:<45} {secs:>8.2f}s")
        print(f"\n  RAM — pico: {ram.get('pico_gb',0):.2f} GB  "
              f"| média: {ram.get('media_gb',0):.2f} GB  "
              f"| mín: {ram.get('minimo_gb',0):.2f} GB")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════

def _timer(label: str, t0: float) -> float:
    print(f"      ✓ {label:<52} {time.perf_counter() - t0:.2f}s")
    return time.perf_counter()


# ══════════════════════════════════════════════════════════════
# GPU — QC via prefix sum (sem construção de matriz)
# ══════════════════════════════════════════════════════════════

def _qc_metrics_gpu(data_cpu, indices_cpu, indptr_cpu, mt_indices):
    data_gpu    = cp.array(data_cpu,    dtype=cp.float32)
    indices_gpu = cp.array(indices_cpu, dtype=cp.int32)
    indptr_gpu  = cp.array(indptr_cpu,  dtype=cp.int64)

    n_genes_cell = cp.diff(indptr_gpu).astype(cp.int32)

    cs           = cp.concatenate([cp.zeros(1, cp.float32), cp.cumsum(data_gpu)])
    total_counts = cs[indptr_gpu[1:]] - cs[indptr_gpu[:-1]]
    del cs

    is_mt     = cp.isin(indices_gpu, cp.array(mt_indices, dtype=cp.int32)).astype(cp.float32)
    cs_mt     = cp.concatenate([cp.zeros(1, cp.float32), cp.cumsum(data_gpu * is_mt)])
    mt_counts = cs_mt[indptr_gpu[1:]] - cs_mt[indptr_gpu[:-1]]
    del data_gpu, indices_gpu, is_mt, cs_mt

    pct_mt    = cp.where(total_counts > 0, mt_counts / total_counts * 100.0, 0.0)
    valid_gpu = cp.where((n_genes_cell >= MIN_GENES) & (pct_mt < MAX_PCT_MT))[0]
    valid_cpu = valid_gpu.get()

    del n_genes_cell, total_counts, mt_counts, pct_mt, valid_gpu, indptr_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return valid_cpu


# ══════════════════════════════════════════════════════════════
# OMP — Numba kernels
#
# FIX: gene_mask passado como uint8 (não bool_)
#      Numba tem comportamento inconsistente com np.bool_ em
#      alguns ambientes — uint8 é equivalente e sempre confiável.
#
# FIX: dst_starts e out_indptr em int64
#      total_nnz_upper (NNZ antes do filtro de genes) pode
#      ultrapassar 2.147B (limite int32), causando overflow
#      silencioso e indptr corrompido.
# ══════════════════════════════════════════════════════════════

@njit(parallel=True, cache=False, fastmath=True)
def _count_chunk_output_nnz(
    indptr:      np.ndarray,   # (n_chunk+1,) int32
    valid_local: np.ndarray,   # (n_valid,)   int32
    gene_mask:   np.ndarray,   # (n_genes,)   uint8  ← uint8, não bool
    indices:     np.ndarray,   # (NNZ,)       int32
) -> np.ndarray:
    """
    Conta NNZ filtrados por gene_mask para cada célula válida.
    Paralelo por célula — cada thread processa uma célula independente.
    Retorna array int64 para evitar overflow no acumulador.
    """
    n_valid   = len(valid_local)
    nnz_count = np.zeros(n_valid, dtype=np.int64)   # int64 seguro
    for i in prange(n_valid):
        cell  = valid_local[i]
        count = np.int64(0)
        for j in range(indptr[cell], indptr[cell + 1]):
            if gene_mask[indices[j]] != np.uint8(0):  # uint8: 1=True, 0=False
                count += np.int64(1)
        nnz_count[i] = count
    return nnz_count


@njit(parallel=True, cache=False, fastmath=True)
def _fill_output_chunk(
    data:        np.ndarray,   # (NNZ,)       float32
    indices:     np.ndarray,   # (NNZ,)       int32
    indptr:      np.ndarray,   # (n_chunk+1,) int32
    valid_local: np.ndarray,   # (n_valid,)   int32
    gene_mask:   np.ndarray,   # (n_genes,)   uint8  ← uint8, não bool
    gene_remap:  np.ndarray,   # (n_genes,)   int32
    out_data:    np.ndarray,   # (NNZ_out,)   float32
    out_indices: np.ndarray,   # (NNZ_out,)   int32
    dst_starts:  np.ndarray,   # (n_valid,)   int64  ← int64 para não overflow
) -> None:
    """
    Copia NNZ filtrados diretamente no buffer de saída pré-alocado.
    Paralelo por célula — sem race condition (ranges não se sobrepõem).
    """
    n_valid = len(valid_local)
    for i in prange(n_valid):
        cell = valid_local[i]
        dst  = dst_starts[i]          # int64
        for j in range(indptr[cell], indptr[cell + 1]):
            g = indices[j]
            if gene_mask[g] != np.uint8(0):
                out_data[dst]    = data[j]
                out_indices[dst] = gene_remap[g]
                dst             += np.int64(1)


def _warmup_numba():
    print("Aquecendo compilador Numba (JIT)...")
    d   = np.ones(10,  dtype=np.float32)
    ix  = np.zeros(10, dtype=np.int32)
    ip  = np.array([0, 5, 10], dtype=np.int32)
    vl  = np.array([0, 1],     dtype=np.int32)
    gm  = np.ones(10,  dtype=np.uint8)    # uint8 igual ao pipeline real
    gr  = np.arange(10, dtype=np.int32)
    dst = np.zeros(2,  dtype=np.int64)    # int64 igual ao pipeline real
    od  = np.empty(10, dtype=np.float32)
    oi  = np.empty(10, dtype=np.int32)
    _count_chunk_output_nnz(ip, vl, gm, ix)
    _fill_output_chunk(d, ix, ip, vl, gm, gr, od, oi, dst)
    print("  JIT compilado.\n")


# ══════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL — DOIS PASSES
# ══════════════════════════════════════════════════════════════

def preprocessing_h5_gpu(
    caminho_h5: str,
    monitor:    MetricsMonitor,
    chunk_size: int = CHUNK_SIZE,
) -> sc.AnnData:

    _warmup_numba()
    t_pipeline = time.perf_counter()
    print("=" * 64)

    # ── Metadados ──────────────────────────────────────────────
    monitor.mark("metadados_inicio")
    print("[Meta] Lendo metadados do H5...")
    t = time.perf_counter()

    with h5py.File(caminho_h5, 'r') as f:
        if 'matrix' in f:
            grp        = f['matrix']
            gene_names = grp['features']['name'][:].astype(str)
            gene_ids   = grp['features']['id'][:].astype(str)
        else:
            gk         = next(k for k in f.keys() if 'data' in f[k])
            grp        = f[gk]
            gene_names = grp['gene_names'][:].astype(str)
            gene_ids   = grp['genes'][:].astype(str)

        n_genes, n_cells = int(grp['shape'][0]), int(grp['shape'][1])
        barcodes         = grp['barcodes'][:].astype(str)
        indptr           = grp['indptr'][:].astype(np.int64)

    t = _timer(f"Metadados ({n_cells:,} células × {n_genes:,} genes)", t)

    mt_mask    = np.array([n.startswith(MT_PREFIXES) for n in gene_names])
    mt_indices = np.where(mt_mask)[0]
    t = _timer(f"Máscara MT ({mt_mask.sum()} genes)", t)

    n_chunks     = (n_cells + chunk_size - 1) // chunk_size
    chunk_starts = [i * chunk_size for i in range(n_chunks)]

    # ════════════════════════════════════════════════════════
    # PASS 1 — QC + contagem de genes
    # ════════════════════════════════════════════════════════
    monitor.mark("pass1_inicio")
    print(f"\n[Pass 1/2] QC chunked: H5 → GPU → contagem de genes...")
    t_pass1 = time.perf_counter()

    valid_per_chunk: list[np.ndarray] = []
    gene_counts = np.zeros(n_genes, dtype=np.int64)
    gene_totals = np.zeros(n_genes, dtype=np.float64)

    with h5py.File(caminho_h5, 'r') as f:
        grp = f['matrix'] if 'matrix' in f else f[next(k for k in f.keys() if 'data' in f[k])]

        for idx in range(n_chunks):
            start   = chunk_starts[idx]
            end     = min(start + chunk_size, n_cells)
            t_chunk = time.perf_counter()
            n_chunk = end - start

            nnz_s = int(indptr[start])
            nnz_e = int(indptr[end])

            data_cpu    = grp['data'][nnz_s:nnz_e].astype(np.float32)
            indices_cpu = grp['indices'][nnz_s:nnz_e].astype(np.int32)
            indptr_cpu  = (indptr[start:end + 1] - nnz_s).astype(np.int32)
            t_io        = time.perf_counter() - t_chunk

            t_gpu_s     = time.perf_counter()
            valid_local = _qc_metrics_gpu(data_cpu, indices_cpu, indptr_cpu, mt_indices)
            t_gpu       = time.perf_counter() - t_gpu_s

            if len(valid_local) > 0:
                cell_nnz  = np.diff(indptr_cpu)
                cell_ids  = np.repeat(np.arange(n_chunk, dtype=np.int32), cell_nnz)
                vmask     = np.zeros(n_chunk, dtype=bool)
                vmask[valid_local] = True
                nnz_valid = vmask[cell_ids]

                gene_counts += np.bincount(
                    indices_cpu[nnz_valid], minlength=n_genes
                )
                gene_totals += np.bincount(
                    indices_cpu[nnz_valid],
                    weights=data_cpu[nnz_valid].astype(np.float64),
                    minlength=n_genes
                )

            valid_per_chunk.append(valid_local)

            del data_cpu, indices_cpu, indptr_cpu
            gc.collect()

            kept_pct = len(valid_local) / n_chunk * 100
            elapsed  = time.perf_counter() - t_chunk
            print(
                f"  Chunk {idx+1:>3}/{n_chunks}"
                f"  [{start:>8,}:{end:>8,}]"
                f"  {len(valid_local):>6,} ({kept_pct:5.1f}%)"
                f"  I/O:{t_io:.1f}s  GPU:{t_gpu:.1f}s"
                f"  {elapsed:.2f}s"
            )

    print(f"\n  Total pass 1 : {time.perf_counter() - t_pass1:.2f}s")
    monitor.mark("pass1_fim")

    # ── Gene mask ──────────────────────────────────────────────
    monitor.mark("gene_mask_inicio")

    all_valid_global = np.concatenate([
        chunk_starts[i] + vl
        for i, vl in enumerate(valid_per_chunk)
        if len(vl) > 0
    ]).astype(np.int64)

    n_cells_out = len(all_valid_global)
    gene_mean   = gene_totals / n_cells_out

    gene_mask_bool = (gene_counts >= MIN_CELLS) & (gene_mean >= MEAN_EXPR_MIN)

    # FIX: converte para uint8 antes de passar ao Numba
    # np.bool_ tem comportamento inconsistente em algumas versões do Numba
    gene_mask_u8 = gene_mask_bool.astype(np.uint8)

    gene_remap = np.full(n_genes, -1, dtype=np.int32)
    gene_remap[gene_mask_bool] = np.arange(int(gene_mask_bool.sum()), dtype=np.int32)
    n_genes_out    = int(gene_mask_bool.sum())
    valid_barcodes = barcodes[all_valid_global].tolist()

    total_nnz_upper = int(np.sum(np.diff(indptr)[all_valid_global]))
    del all_valid_global

    print(f"\n  Células válidas : {n_cells_out:,}")
    print(f"  Genes mantidos  : {n_genes_out:,} / {n_genes:,}")
    print(f"  NNZ upper bound : {total_nnz_upper:,}")

    monitor.mark("gene_mask_fim")

    # ════════════════════════════════════════════════════════
    # PASS 2 — Monta matriz filtrada em buffer pré-alocado
    # ════════════════════════════════════════════════════════
    monitor.mark("pass2_inicio")
    print(f"\n[Pass 2/2] Construindo matriz filtrada (row+col inline)...")
    t_pass2 = time.perf_counter()

    out_data    = np.empty(total_nnz_upper, dtype=np.float32)
    out_indices = np.empty(total_nnz_upper, dtype=np.int32)

    # FIX: int64 — total_nnz_upper pode ultrapassar 2.147B (limite int32)
    out_indptr  = np.zeros(n_cells_out + 1, dtype=np.int64)

    row_ptr = 0    # próxima linha no output
    nnz_ptr = 0    # próximo NNZ no output

    with h5py.File(caminho_h5, 'r') as f:
        grp = f['matrix'] if 'matrix' in f else f[next(k for k in f.keys() if 'data' in f[k])]

        for idx in range(n_chunks):
            start       = chunk_starts[idx]
            end         = min(start + chunk_size, n_cells)
            valid_local = valid_per_chunk[idx]

            if len(valid_local) == 0:
                continue

            t_chunk = time.perf_counter()

            nnz_s = int(indptr[start])
            nnz_e = int(indptr[end])

            data_cpu    = grp['data'][nnz_s:nnz_e].astype(np.float32)
            indices_cpu = grp['indices'][nnz_s:nnz_e].astype(np.int32)
            indptr_cpu  = (indptr[start:end + 1] - nnz_s).astype(np.int32)
            t_io        = time.perf_counter() - t_chunk

            t_omp_s = time.perf_counter()

            # Conta NNZ filtrados por célula — retorna int64
            nnz_counts = _count_chunk_output_nnz(
                indptr_cpu, valid_local, gene_mask_u8, indices_cpu
            )

            # FIX: dst_starts em int64 — evita overflow quando nnz_ptr > 2.147B
            dst_starts = (
                np.concatenate([[0], np.cumsum(nnz_counts[:-1])]) + nnz_ptr
            ).astype(np.int64)

            # Preenche buffer de saída em paralelo
            _fill_output_chunk(
                data_cpu, indices_cpu, indptr_cpu,
                valid_local, gene_mask_u8, gene_remap,
                out_data, out_indices, dst_starts,
            )

            # Atualiza out_indptr (int64 — sem risco de overflow)
            cum = np.cumsum(nnz_counts)
            out_indptr[row_ptr + 1 : row_ptr + len(valid_local) + 1] = (
                out_indptr[row_ptr] + cum
            )

            n_valid     = len(valid_local)
            nnz_written = int(nnz_counts.sum())
            row_ptr    += n_valid
            nnz_ptr    += nnz_written
            t_omp       = time.perf_counter() - t_omp_s

            del data_cpu, indices_cpu, indptr_cpu, nnz_counts, dst_starts, cum
            gc.collect()

            elapsed = time.perf_counter() - t_chunk
            print(
                f"  Chunk {idx+1:>3}/{n_chunks}"
                f"  [{start:>8,}:{end:>8,}]"
                f"  {n_valid:>6,} células"
                f"  {nnz_written:>10,} NNZ"
                f"  I/O:{t_io:.1f}s  OMP:{t_omp:.1f}s"
                f"  {elapsed:.2f}s"
            )

    print(f"\n  Total pass 2 : {time.perf_counter() - t_pass2:.2f}s")
    monitor.mark("pass2_fim")

    # ── Trim, validação e AnnData ───────────────────────────────
    monitor.mark("anndata_inicio")

    actual_nnz = nnz_ptr
    if actual_nnz < total_nnz_upper:
        out_data    = out_data[:actual_nnz]
        out_indices = out_indices[:actual_nnz]

    # Validação de integridade: indptr[-1] deve bater com NNZ real
    if int(out_indptr[-1]) != actual_nnz:
        print(f"  ⚠ indptr[-1]={int(out_indptr[-1])} ≠ actual_nnz={actual_nnz} — corrigindo")
        out_indptr[-1] = actual_nnz

    print(f"  NNZ real : {actual_nnz:,}")
    print("\n[Final] Construindo AnnData...")
    t = time.perf_counter()

    # scipy aceita int64 indptr a partir da versão 1.8
    X = sp.csr_matrix(
        (out_data, out_indices, out_indptr),
        shape=(n_cells_out, n_genes_out)
    )

    var_out = pd.DataFrame(
        index=gene_names[gene_mask_bool],
        data={
            "gene_ids": gene_ids[gene_mask_bool],
            "mt":       mt_mask[gene_mask_bool],
        }
    )
    adata = sc.AnnData(
        X   = X,
        obs = pd.DataFrame(index=valid_barcodes),
        var = var_out,
    )
    adata.var_names_make_unique()

    t = _timer("AnnData construído", t)
    monitor.mark("anndata_fim")

    t_total = time.perf_counter() - t_pipeline
    print(f"\n{'=' * 64}")
    print(f"  PIPELINE TOTAL   : {t_total:.2f}s  ({t_total/60:.1f} min)")
    print(f"  Células          : {adata.n_obs:,}")
    print(f"  Genes            : {adata.n_vars:,}")
    print(f"{'=' * 64}\n")

    return adata


# ══════════════════════════════════════════════════════════════
# I/O FINAL
# ══════════════════════════════════════════════════════════════

def salvar_dataset(
    adata:             sc.AnnData,
    diretorio_destino: str,
    monitor:           MetricsMonitor,
    nome_arquivo:      str = "dataset_filtrado.h5ad",
) -> str:
    monitor.mark("io_save_inicio")
    print("[I/O] Gravando no SSD...")
    t = time.perf_counter()

    os.makedirs(diretorio_destino, exist_ok=True)
    caminho_final = os.path.join(diretorio_destino, nome_arquivo)
    adata.write_h5ad(caminho_final, compression=None)

    elapsed = time.perf_counter() - t
    tamanho = os.path.getsize(caminho_final) / (1024 ** 2)
    print(f"✓ Salvo em {elapsed:.2f}s  |  {tamanho:.0f} MB")
    print(f"  {caminho_final}\n")

    monitor.mark("io_save_fim")
    return caminho_final


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    CAMINHO_ENTRADA = (
        r"C:\Users\igorm\Documents\programacao\trabalho-cad-v2"
        r"\dataset\antigo\1M_neurons_filtered_gene_bc_matrices_h5.h5"
    )
    DIRETORIO_SAIDA = (
        r"C:\Users\igorm\Documents\programacao\trabalho-cad-v2"
        r"\dataset\pre-processado"
    )
    METRICAS_TXT  = os.path.join(DIRETORIO_SAIDA, "metricas.txt")
    METRICAS_JSON = os.path.join(DIRETORIO_SAIDA, "metricas.json")

    monitor = MetricsMonitor(sample_interval=0.5)
    monitor.start()

    try:
        adata = preprocessing_h5_gpu(CAMINHO_ENTRADA, monitor)
        salvar_dataset(adata, DIRETORIO_SAIDA, monitor)
    finally:
        monitor.stop()
        monitor.imprimir()
        monitor.salvar(METRICAS_TXT, METRICAS_JSON)
