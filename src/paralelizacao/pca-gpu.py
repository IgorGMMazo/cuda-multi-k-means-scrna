"""
pca_gpu.py — PCA na GPU por covariância (CuPy + cuSOLVER), sobre HVG
=====================================================================

Substitui o IRLBA out-of-core (96h) por covariância gene×gene +
eigendecomposição na GPU. Diferença em relação à primeira versão: agora
seleciona os top-N HVG (highly variable genes, estilo Seurat/scanpy) ANTES
do PCA, em vez de rodar sobre todos os ~13,9k genes.

Por quê: o fluxo OSCA/Bioconductor do paper de referência (Hicks et al.,
2021) roda PCA sobre HVGs (modelGeneVar/denoisePCA do scran), não sobre
todos os genes. Rodar sobre todos os genes dilui o sinal biológico com
ruído técnico de genes pouco variáveis — foi isso, não bug, que explicava
o UMAP em "borrão único" em vez de ilhas separadas (ver
metricas-etapas/analise_clusterabilidade.json: com HVG, PC1+PC2 sobem de
5,6% para ~36% da variância e a estrutura de cluster fica bem mais visível).

Por que isto é rápido: com n_células (1,1M) >> n_genes (HVG, ~2k), a matriz de
dados inteira já cortada pra HVG (~2-3 GB) cabe de uma vez só na VRAM (8 GB) —
diferente do caso "todos os genes" (13,9k), que não cabe e obriga a processar
em centenas de blocos pequenos pelo PCIe. Etapas:
  Passe 0: seleciona HVG (dispersão var/mean por bin), streaming do disco
  (só essa etapa precisa ler em blocos — ainda não sabemos quais genes ficam)
  Montagem: lê o disco em poucos blocos GRANDES, já corta pras colunas HVG,
  empilha numa matriz só e sobe pra VRAM de uma vez
  Covariância: S = XᵀX somado em CHUNKS (não um matmul único — ver nota de
  precisão abaixo); M = S - n·μμᵀ (= (X-μ)ᵀ(X-μ), sem densificar X);
  eigh(M) → top 50 loadings
  Projeção: scores = (X - μ)·V também é UM ÚNICO matmul (X já está na VRAM;
  aqui não há cancelamento catastrófico, então um matmul só é seguro)

Validado (sem HVG): idêntico ao sklearn PCA (corr 1.000000 nos 50 PCs). A
seleção de HVG não muda a matemática do PCA, só o conjunto de genes de
entrada — mesmos dados reais, mesmo algoritmo.

CORREÇÃO DE PRECISÃO (importante): a primeira versão fazia S = XᵀX como UM
ÚNICO matmul esparso sobre as 1.148.558 linhas inteiras (dado em float32),
só convertendo pra float64 no final. Como M = S − n·μμᵀ subtrai dois números
GRANDES e parecidos (S é dominado pelo termo de média; a variância real é
uma fração pequena disso), o erro de arredondamento acumulado em float32
durante a redução inteira fica amplificado pelo cancelamento catastrófico
na subtração — contaminando não só os autovalores (ficavam em ~1e-13,
visivelmente errados) mas também os autovetores (loadings) reais usados
pra projetar os scores. Resultado prático: PC1+PC2 saía em 7,6% da
variância em vez dos ~36% esperados (ver
metricas-etapas/analise_clusterabilidade.json e
src/analise/investiga_variancia_hvg.py, que confirma reproduzindo os 36,4%
com os MESMOS genes HVG, só somando S em chunks de 20k linhas convertidos
pra float64 a cada bloco — equivalente numérico de soma hierárquica/
compensada). Por isso agora a covariância é acumulada em CHUNKS: cada
bloco parcial (float32) é convertido pra float64 e somado ANTES do
próximo bloco, em vez de deixar o cuSPARSE reduzir 1,1M linhas inteiras
em float32 de uma vez.
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
# pasta NOVA (pca_hvg/) — não toca em dataset/pca/pca.npy, que é a versão "todos os genes" já existente
SAIDA_SCORES    = BASE / "dataset" / "pca_hvg" / "pca.npy"
HVG_MASK_OUT    = BASE / "dataset" / "pca_hvg" / "hvg_mask.npy"      # bool, shape (n_genes_total,)
HVG_GENES_OUT   = BASE / "dataset" / "pca_hvg" / "hvg_gene_ids.npy"  # ids dos genes selecionados
METRICAS_JSON   = BASE / "metricas-etapas" / "metricas_pca_hvg.json"   # NOVO arquivo — não sobrescreve metricas_pca.json

CFG = {"n_components": 50, "n_hvg": 2_000, "hvg_n_bins": 20,
       "hvg_batch_size": 20_000,     # blocos do Passe 0 (HVG) — varre os 13.897 genes
       "build_batch_size": 100_000,  # blocos só da montagem da matriz HVG (poucos, grandes)
       "cov_batch_size": 20_000}     # blocos da soma S=XtX (float64 por bloco — evita cancelamento catastrófico)


def hvg_seurat_gpu(read_chunk, n, g, n_top, n_bins=20, bs=20_000):
    """HVG estilo Seurat/scanpy, EM STREAMING (lê do disco em blocos, nunca
    materializa a matriz inteira) e com a redução pesada (soma/soma² sobre
    ~1,1M x g) na GPU — só o pós-processamento final (tamanho g, barato)
    fica em NumPy/CPU. Também faz o guard de NaN/Inf por bloco, então não
    precisa mais de uma varredura extra só pra isso.
    Mesmo método já validado em src/analise/hvg_pca_silhouette.py — ver
    metricas-etapas/analise_clusterabilidade.json para a comparação com/sem HVG."""
    s1 = cp.zeros(g, dtype=cp.float64)
    s2 = cp.zeros(g, dtype=cp.float64)
    n_nan = n_inf = 0
    for s in range(0, n, bs):
        e = min(s + bs, n)
        Xb_cpu = read_chunk(s, e)                       # lê só esse bloco do disco
        data = Xb_cpu.data
        n_nan += int(np.isnan(data).sum()); n_inf += int(np.isinf(data).sum())

        exp_data = cp.expm1(cp.asarray(data, dtype=cp.float64))   # só o bloco sobe pra GPU
        indices  = cp.asarray(Xb_cpu.indices)
        indptr   = cp.asarray(Xb_cpu.indptr)
        Eb  = cpsp.csr_matrix((exp_data,      indices, indptr), shape=Xb_cpu.shape)
        Eb2 = cpsp.csr_matrix((exp_data ** 2, indices, indptr), shape=Xb_cpu.shape)
        s1 += cp.asarray(Eb.sum(axis=0)).ravel()
        s2 += cp.asarray(Eb2.sum(axis=0)).ravel()
        del Xb_cpu, exp_data, indices, indptr, Eb, Eb2
        cp.get_default_memory_pool().free_all_blocks()

    if n_nan > 0 or n_inf > 0:
        raise ValueError(f"Dado de entrada corrompido: {n_nan} NaN, {n_inf} Inf. "
                         "Re-execute o scran_gpu.py para regenerar o normalizado_scran.h5ad.")

    mean = cp.asnumpy(s1 / n)
    var  = cp.asnumpy(s2 / n) - mean ** 2
    mean_safe = np.where(mean == 0, 1e-12, mean)
    disp = var / mean_safe
    disp[disp == 0] = np.nan
    disp = np.log(disp)
    lmean = np.log1p(mean)
    bins = np.minimum((n_bins * np.argsort(np.argsort(lmean)) // g), n_bins - 1)
    norm_disp = np.full(g, np.nan)
    for b in range(n_bins):
        m = bins == b
        d = disp[m]
        mu_b, sd_b = np.nanmean(d), np.nanstd(d)
        if not np.isfinite(sd_b) or sd_b == 0:
            sd_b = 1.0
        norm_disp[m] = (disp[m] - mu_b) / sd_b
    order = np.argsort(np.nan_to_num(norm_disp, nan=-np.inf))[::-1]
    mask = np.zeros(g, bool)
    mask[order[:n_top]] = True
    return mask


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

    print("=" * 64, "\n  PCA GPU (covariância + cuSOLVER) — 50 PCs, sobre HVG\n", "=" * 64, sep="")

    # backed="r": NÃO carrega a matriz inteira na RAM, só metadados. Cada bloco é
    # lido do disco sob demanda (mesmo padrão usado em validar_inetgridade.py).
    adata = ad.read_h5ad(CAMINHO_ENTRADA, backed="r")
    n, p_full = adata.shape
    var_names = np.asarray(adata.var_names)

    def read_chunk(s, e):
        Xb = adata.X[s:e]
        Xb = Xb if sp.issparse(Xb) else sp.csr_matrix(np.asarray(Xb))
        return Xb.tocsr().astype(np.float32, copy=False)

    marca("Dataset aberto (backed, sem carregar tudo na RAM)")

    # ── Passe 0: seleção de HVG na GPU, streaming direto do disco ────────────────
    n_hvg = CFG["n_hvg"]
    hvg_mask = hvg_seurat_gpu(read_chunk, n, p_full, n_hvg,
                               n_bins=CFG["hvg_n_bins"], bs=CFG["hvg_batch_size"])
    p = int(hvg_mask.sum())
    HVG_MASK_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(HVG_MASK_OUT, hvg_mask)
    np.save(HVG_GENES_OUT, var_names[hvg_mask])
    marca(f"HVG selecionado ({p:,} de {p_full:,} genes)")

    k = CFG["n_components"]
    print(f"  {n:,} células × {p:,} genes (HVG top-{n_hvg}, de {p_full:,} totais) | {k} PCs\n")

    # ── Montagem: lê o disco em POUCOS blocos grandes, já corta pra HVG, empilha ──
    # Com p=2.000 (em vez de 13.897), a matriz inteira cabe folgado em RAM e em
    # VRAM (poucos GB) — não precisa mais fatiar em ~165 blocos pequenos, cada um
    # pagando overhead de leitura em modo backed. Poucos blocos grandes = muito
    # menos idas ao disco.
    build_bs = CFG["build_batch_size"]
    blocos = []
    for s in range(0, n, build_bs):
        e = min(s + build_bs, n)
        blocos.append(read_chunk(s, e)[:, hvg_mask])
    X_hvg_cpu = sp.vstack(blocos, format="csr")
    del blocos
    marca(f"Montagem: matriz HVG completa em RAM ({X_hvg_cpu.data.nbytes / 1024**3:.2f} GB)")

    # ── Sobe a matriz inteira pra VRAM de uma vez só ──────────────────────────────
    Xg = cpsp.csr_matrix(X_hvg_cpu)
    del X_hvg_cpu
    marca("Matriz HVG inteira na VRAM")

    # ── Covariância: S = XᵀX somado em CHUNKS, cada bloco convertido pra float64
    # ANTES de acumular (não um matmul único em float32 sobre as 1,1M linhas
    # inteiras) — ver nota de precisão no topo do arquivo. Dado já está todo na
    # VRAM, então isto é só re-fatiar o que já está residente; sem I/O extra.
    # Importante: o matmul do bloco é feito DENSO (cuBLAS), não esparso-esparso
    # (cuSPARSE spgemm) — spgemm em fatias (poucas linhas × p colunas) estimou
    # um buffer de ~48 GB e estourou a VRAM; denso é trivial aqui (bloco de
    # 20k×2000 = ~160 MB) e não tem esse problema de estimativa de buffer.
    cov_bs = CFG["cov_batch_size"]
    S = cp.zeros((p, p), dtype=cp.float64)
    colsum = cp.zeros(p, dtype=cp.float64)
    for s in range(0, n, cov_bs):
        e = min(s + cov_bs, n)
        Xb = Xg[s:e].toarray()                         # denso (bs, p), float32 — barato
        S += (Xb.T @ Xb).astype(cp.float64)
        colsum += Xb.sum(axis=0).astype(cp.float64)
    mu = colsum / n
    cp.get_default_memory_pool().free_all_blocks()
    marca("Covariancia XtX (chunks float64, sem cancelamento catastrófico)")

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

    # ── Projeção: UM ÚNICO matmul (Xg ainda está inteira na VRAM) ────────────────
    muV        = (mu.astype(cp.float32) @ Vk)         # (k,)
    scores_gpu = Xg @ Vk - muV[None, :]                # (n, k) — n×2000 @ 2000×50, de uma vez
    scores     = cp.asnumpy(scores_gpu)
    del Xg, scores_gpu
    cp.get_default_memory_pool().free_all_blocks()
    # pós-ordenação por variância empírica — rede de segurança p/ espectro plano
    # (autovalores próximos podem ter ordenação instável; a variância empírica é a verdade)
    var_pc  = scores.var(0)
    ord_var = np.argsort(var_pc)[::-1]
    scores  = np.ascontiguousarray(scores[:, ord_var])
    print(f"  var por PC (top-6): {np.round(scores.var(0)[:6], 4)}")
    marca("Projecao (matmul único) + reordenacao")

    # ── Saída ────────────────────────────────────────────────────────────────────
    SAIDA_SCORES.parent.mkdir(parents=True, exist_ok=True)
    np.save(SAIDA_SCORES, scores)
    var = cp.asnumpy(wk / (n - 1))
    METRICAS_JSON.write_text(json.dumps(
        {"etapas": etapas, "explained_variance": var.tolist(),
         "var_ratio_top10": (var[:10] / var.sum()).tolist(),
         "n_cells": int(n), "n_genes_hvg": p, "n_genes_total": p_full,
         "hvg_top_n": n_hvg}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    marca("Scores salvos")

    print(f"\n  scores: {scores.shape}  | var. explicada PC1..PC3: "
          f"{var[0]:.3f}, {var[1]:.3f}, {var[2]:.3f}")
    print(f"  -> {SAIDA_SCORES}")


if __name__ == "__main__":
    main()
