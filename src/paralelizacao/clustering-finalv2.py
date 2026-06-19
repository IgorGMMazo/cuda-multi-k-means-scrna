"""
clustering_final.py — Multi-K Means GPU (Lloyd's completo + Gumbel-Max + Prefix Reuse)
============================================================================
"""

import json
import threading
import time
from pathlib import Path

import cupy as cp
import cupyx
import matplotlib.pyplot as plt
import numpy as np

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE           = Path(__file__).resolve().parents[2]   # raiz do repositório (trabalho-cad-v2/)
PCA_SCORES     = str(BASE / "dataset" / "pca" / "pca.npy")
SAIDA_LABELS   = str(BASE / "metricas-etapas" / "labels_multi_k.npz")
SAIDA_JSON     = str(BASE / "metricas-etapas" / "resultados_multi_kv4.json")
SAIDA_PLOT     = str(BASE / "metricas-etapas" / "grafico_metricasv4.png")

# ── Config ────────────────────────────────────────────────────────────────────
K_VALUES      = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
N_INIT        = 3          # restarts p/ evitar mínimos locais
MAX_ITER      = 300        # teto de iterações Lloyd
TOL           = 1e-4       # critério de convergência (shift de centroides)
N_SIL         = 5_000      # células amostradas p/ silhouette
SEED          = 42


# ── K-Means++ com Gumbel-Max Trick (100% GPU) ─────────────────────────────────
def kpp_init_gumbel(X: cp.ndarray, xnorms: cp.ndarray, k: int, seed: int) -> cp.ndarray:
    cp.random.seed(seed)
    n = len(X)
    
    C = cp.zeros((k, X.shape[1]), dtype=cp.float32)
    # Primeiro centroide aleatório
    C[0] = X[int(cp.random.randint(0, n))]
    
    min_d = cp.full(n, cp.inf, dtype=cp.float32)
    
    for i in range(1, k):
        c = C[i-1]
        d = xnorms - 2.0 * (X @ c) + float((c ** 2).sum())
        cp.maximum(d, 0.0, out=d)
        cp.minimum(min_d, d, out=min_d)
        
        # Gumbel-Max trick para amostragem ponderada
        U = cp.random.rand(n, dtype=cp.float32)
        gumbel_noise = -cp.log(-cp.log(U + 1e-10))
        
        # O score é log(prob) + ruido. A prob é proporcional a min_d.
        scores = cp.log(min_d + 1e-10) + gumbel_noise
        next_idx = int(cp.argmax(scores))
        
        C[i] = X[next_idx]
        
    return C


# ── Lloyd's completo ──────────────────────────────────────────────────────────
def lloyd(X: cp.ndarray, xnorms: cp.ndarray, C_init: cp.ndarray,
          max_iter: int = MAX_ITER, tol: float = TOL):
    C   = C_init.copy()
    k   = len(C)
    n   = len(X)
    D_mat = None
    for it in range(max_iter):
        cn    = (C ** 2).sum(1)                                    # (k,)
        D_mat = xnorms[:, None] - 2.0 * (X @ C.T) + cn[None, :]    # (n,k)
        labels = cp.argmin(D_mat, axis=1).astype(cp.int32)         # (n,)

        C_new  = cp.zeros((k, X.shape[1]), dtype=cp.float32)
        counts = cp.bincount(labels, minlength=k).astype(cp.float32)
        cupyx.scatter_add(C_new, labels, X)               
        C_new /= cp.maximum(counts[:, None], 1.0)         

        empty = counts == 0
        C_new[empty] = C[empty]

        shift = float(cp.linalg.norm(C_new - C) / (cp.linalg.norm(C) + 1e-12))
        C = C_new
        if shift < tol:
            break

    wcss = float(D_mat[cp.arange(n), labels].clip(0).sum())
    return labels, C, it + 1, wcss


# ── Avalia K reaproveitando o prefixo da cadeia ───────────────────────────────
def fit_k_from_chains(X: cp.ndarray, xnorms: cp.ndarray, k: int, n_init: int, kpp_chains: list):
    best = None
    for r in range(n_init):
        # O pulo do gato: Pega apenas os 'k' primeiros centroides da cadeia longa
        C0 = kpp_chains[r][:k]
        lab, C, iters, wcss = lloyd(X, xnorms, C0)
        
        if best is None or wcss < best["wcss"]:
            best = {"labels": lab, "centroids": C, "iters": iters, "wcss": wcss}
    return best


# ── Silhouette amostrado ──────────────────────────────────────────────────────
def silhouette_sampled(X: cp.ndarray, labels_cpu: np.ndarray,
                       k: int, n_s: int = N_SIL, seed: int = SEED) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(labels_cpu), min(n_s, len(labels_cpu)), replace=False)
    Xs  = X[idx]                                        
    ls  = cp.asarray(labels_cpu[idx])               

    xn  = (Xs ** 2).sum(1)
    D   = cp.maximum(xn[:, None] + xn[None, :] - 2.0 * (Xs @ Xs.T), 0.0)

    a   = cp.zeros(len(Xs), dtype=cp.float64)
    b   = cp.full(len(Xs), cp.inf, dtype=cp.float64)

    for c in range(k):
        in_c = ls == c
        cnt  = int(in_c.sum())
        if cnt == 0:
            continue
        D_c  = D[:, in_c]
        b_c  = D_c.sum(1) / cnt
        if cnt > 1:
            a_c = D_c.sum(1) / (cnt - 1)
            a   = cp.where(in_c, a_c.astype(cp.float64), a)
        b = cp.where(~in_c, cp.minimum(b, b_c.astype(cp.float64)), b)

    valid = cp.isfinite(b)
    if not int(valid.sum()):
        return 0.0
    denom = cp.maximum(cp.maximum(a[valid], b[valid]), 1e-10)
    return float(((b[valid] - a[valid]) / denom).mean())


# ── Elbow: 2ª derivada discreta ───────────────────────────────────────────────
def find_elbow(ks: list, wcss: list) -> int:
    w  = np.array(wcss, float)
    d2 = np.gradient(np.gradient(w))
    return int(np.array(ks)[np.argmax(d2)])


# ── Plotagem de Gráficos ──────────────────────────────────────────────────────
def plot_metrics(ks, wcss, sil, elbow_k, best_sil_k, save_path):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Número de Clusters (K)', fontweight='bold')
    ax1.set_ylabel('WCSS / n', color=color, fontweight='bold')
    ax1.plot(ks, wcss, marker='o', color=color, linewidth=2, label='WCSS')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axvline(x=elbow_k, color='blue', linestyle='--', alpha=0.5, label=f'Elbow (K={elbow_k})')

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Silhouette Score', color=color, fontweight='bold')
    ax2.plot(ks, sil, marker='s', color=color, linewidth=2, label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axvline(x=best_sil_k, color='orange', linestyle='--', alpha=0.5, label=f'Best Sil (K={best_sil_k})')

    plt.title('Análise de Métricas Multi-K Means', fontweight='bold', fontsize=14)
    fig.tight_layout()
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # Unificando legendas
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\n  Gráfico salvo → {Path(save_path).name}")


# ── Worker por thread ─────────────────────────────────────────────────────────
# ── Worker por thread ─────────────────────────────────────────────────────────
def worker(k: int, X_gpu: cp.ndarray, xnorms: cp.ndarray, kpp_chains: list,
           results: dict, lock: threading.Lock, t_ini: float):
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        t0   = time.perf_counter()
        best = fit_k_from_chains(X_gpu, xnorms, k, N_INIT, kpp_chains)
        sil  = silhouette_sampled(X_gpu, cp.asnumpy(best["labels"]), k)
        stream.synchronize()
        elapsed = time.perf_counter() - t0

    with lock:
        results[k] = {
            "wcss":       best["wcss"],
            "wcss_n":     best["wcss"] / len(X_gpu),
            "silhouette": sil,
            "n_iters":    best["iters"],
            "elapsed_s":  round(elapsed, 2),
            "labels":     cp.asnumpy(best["labels"]),
        }

    # --- A MÁGICA DA LIMPEZA AQUI ---
    # 1. Apaga a referência das matrizes da GPU que estão na variável local
    del best
    
    # 2. Força o CuPy a esvaziar o Pool e devolver a memória pra placa de vídeo
    cp.get_default_memory_pool().free_all_blocks()
    
    # 3. Mede a VRAM *depois* da limpeza para o log ficar real
    free, tot = cp.cuda.runtime.memGetInfo()
    with lock:
        print(f"  k={k:2d} | WCSS/n={results[k]['wcss_n']:.4f} | "
              f"sil={sil:.4f} | iters={results[k]['n_iters']} | "
              f"{elapsed:.1f}s | VRAM {(tot-free)/1024**3:.2f}GB")


# ── Pipeline ──────────────────────────────────────────────────────────────────
def main():
    t_ini = time.perf_counter()
    print("=" * 64)
    print("  MULTI-K MEANS GPU — Gumbel-Max + Prefix Reuse")
    print(f"  K values: {K_VALUES}  |  n_init={N_INIT}  |  tol={TOL}")
    print("=" * 64)

    # 1. Carrega dados
    scores_np = np.load(PCA_SCORES).astype(np.float32)
    n, d      = scores_np.shape
    X_gpu     = cp.asarray(scores_np);  del scores_np
    xnorms    = (X_gpu ** 2).sum(1)
    
    # 2. Pré-computa cadeias K-Means++ (Reuso de prefixo)
    k_max = max(K_VALUES)
    print(f"  Pré-computando {N_INIT} cadeias K-Means++ até K={k_max}...")
    kpp_chains = [kpp_init_gumbel(X_gpu, xnorms, k_max, SEED + r * 1000) for r in range(N_INIT)]
    print(f"  Cadeias prontas em {time.perf_counter()-t_ini:.1f}s.\n")

    # 3. Dispara threads
    results: dict = {}
    lock    = threading.Lock()
    threads = []
    
    t_par = time.perf_counter()
    for k in K_VALUES:
        t = threading.Thread(
            target=worker,
            args=(k, X_gpu, xnorms, kpp_chains, results, lock, t_ini),
            daemon=True,
        )
        threads.append(t)

    for t in threads: t.start()
    for t in threads: t.join()
    t_par = time.perf_counter() - t_par

    # 4. Resultados e Métricas
    ks_sorted  = sorted(results)
    wcss_list  = [results[k]["wcss_n"] for k in ks_sorted] # plot usa WCSS/n para normalizar
    sil_list   = [results[k]["silhouette"] for k in ks_sorted]
    elbow_k    = find_elbow(ks_sorted, wcss_list)
    best_sil_k = ks_sorted[int(np.argmax(sil_list))]

    print(f"\n  {'K':>3}  {'WCSS/n':>10}  {'Silhouette':>11}  {'Iters':>6}  {'Tempo':>7}")
    print(f"  {'-'*46}")
    for k in ks_sorted:
        r   = results[k]
        tag = " ← elbow" if k == elbow_k else (" ← melhor sil" if k == best_sil_k else "")
        print(f"  {k:>3}  {r['wcss_n']:>10.4f}  {r['silhouette']:>11.4f}  "
              f"{r['n_iters']:>6}  {r['elapsed_s']:>6.1f}s{tag}")

    # 5. Salva saídas e gera Gráfico
    Path(SAIDA_LABELS).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SAIDA_LABELS, **{f"k{k}": results[k]["labels"] for k in ks_sorted})

    resumo = {
        "k_values":    ks_sorted,
        "elbow_k":     elbow_k,
        "best_sil_k":  best_sil_k,
        "tempo_paralelo_s": round(t_par, 2),
        "resultados": {
            str(k): {k_res: results[k][k_res] for k_res in ["wcss", "wcss_n", "silhouette", "n_iters", "elapsed_s"]}
            for k in ks_sorted
        },
    }
    Path(SAIDA_JSON).write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    
    plot_metrics(ks_sorted, wcss_list, sil_list, elbow_k, best_sil_k, SAIDA_PLOT)

    print(f"  labels  → {Path(SAIDA_LABELS).name}")
    print(f"  JSON    → {Path(SAIDA_JSON).name}")
    print("=" * 64)


if __name__ == "__main__":
    main()