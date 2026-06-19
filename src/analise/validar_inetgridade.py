"""
validar_integridade.py — Checagem de integridade do pipeline de dados (v2)
============================================================================
v2: processa cada etapa em BLOCOS (streaming), nunca materializando a
matriz inteira na RAM. A v1 carregava raw + filtrado + cluster_init +
normalizado todos de uma vez, inteiros — pra um dataset de ~1,3M células
x ~28k genes isso facilmente passa de 25-30 GB só na matriz raw (mais a
cópia transitória do .T.tocsr()). Em 32 GB de RAM, estoura.

A correção é o mesmo princípio que justifica o uso de HDF5 chunked no
pipeline original: ler em blocos de células, acumular as estatísticas
que importam (NaN, Inf, negativos, soma por célula/gene), descartar o
bloco, seguir pro próximo. Memória de pico fica limitada a ~CHUNK_SIZE
células por vez, não ao dataset inteiro.

Para as comparações ENTRE etapas (n_células, barcodes, genes), só
guardamos metadados leves (MetaEtapa) — nunca a matriz.

Cada checagem te dá um veredito: OK / AVISO / FALHA.
  OK    -> nada a fazer.
  AVISO -> não invalida o pipeline, mas merece sua atenção/justificativa.
  FALHA -> não confie no resultado até resolver isso.

ANTES DE RODAR:
  - Confirme o nome exato do arquivo raw (estava truncado no print do
    VS Code) em RAW_H5.
  - Se ainda estourar RAM, reduza CHUNK_SIZE (linha de config) — vale a
    troca: processamento mais lento, mas memória de pico menor.
  - requer numpy, scipy, h5py. anndata é opcional só pros arquivos .h5ad
    (se não tiver instalado, aquela etapa específica vira FALHA com
    mensagem clara, e o resto do script continua).
"""

import gc
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

# ── Caminhos ──────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\igorm\Documents\programacao\trabalho-cad-v2\dataset")

RAW_H5        = BASE / "inicial" / "1M_neurons_filtered_gene_bc_matrices_h5.h5"   # <-- confirme o nome exato
FILTRADO      = BASE / "pre-processado" / "dataset_filtrado.h5ad"
CLUSTER_INIT  = BASE / "clusterizado-inicial" / "clustering_all_genes.h5ad"
NORMALIZADO   = BASE / "normalizado" / "normalizado_scran.h5ad"
PCA_SCORES    = BASE / "pca" / "pca.npy"
LABELS_FINAIS = BASE.parent / "metricas-etapas" / "labels_multi_k.npz"   # opcional, se já existir
RELATORIO_OUT = BASE.parent / "metricas-etapas" / "relatorio_integridade.json"

N_PCS_ESPERADO = 50
CHUNK_SIZE     = 20_000   # células por bloco. Estourou RAM? Reduza pra 5_000 ou 2_000.


# ── Infraestrutura de relatório ──────────────────────────────────────────
@dataclass
class Resultado:
    etapa: str
    checagem: str
    status: str   # "OK" | "AVISO" | "FALHA"
    detalhe: str


@dataclass
class Relatorio:
    resultados: list = field(default_factory=list)

    def add(self, etapa, checagem, status, detalhe=""):
        self.resultados.append(Resultado(etapa, checagem, status, detalhe))
        print(f"  [{status:<6}] {etapa:<14} {checagem:<32} {detalhe}")

    def resumo(self):
        n_ok = sum(r.status == "OK" for r in self.resultados)
        n_warn = sum(r.status == "AVISO" for r in self.resultados)
        n_fail = sum(r.status == "FALHA" for r in self.resultados)
        return n_ok, n_warn, n_fail

    def salvar_json(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps([r.__dict__ for r in self.resultados], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


@dataclass
class MetaEtapa:
    """Só os metadados leves de uma etapa — nunca a matriz de contagens."""
    nome: str
    n_obs: int
    n_vars: int
    var_names: list = None
    obs_names: set = None


# ── Núcleo: processamento em blocos, memória limitada a CHUNK_SIZE células ──
def processar_em_chunks(leitor_bloco, n_rows, n_cols, chunk_size, esperar_inteiro):
    n_nan = n_inf = n_neg = n_frac = nnz_total = n_celulas_zero = 0
    soma_coluna = np.zeros(n_cols, dtype=np.float64)

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        bloco = leitor_bloco(start, end)
        if not sp.issparse(bloco):
            bloco = sp.csr_matrix(np.asarray(bloco))
        dados = bloco.data

        if dados.dtype.kind == "f":
            n_nan += int(np.isnan(dados).sum())
            n_inf += int(np.isinf(dados).sum())
        n_neg += int((dados < 0).sum())
        if esperar_inteiro and dados.size:
            n_frac += int((np.abs(dados - np.round(dados)) > 1e-6).sum())
        nnz_total += bloco.nnz

        soma_linha = np.asarray(bloco.sum(axis=1)).ravel()
        n_celulas_zero += int((soma_linha == 0).sum())
        soma_coluna += np.asarray(bloco.sum(axis=0)).ravel()

        del bloco, dados, soma_linha   # libera o bloco antes do próximo — isso é o que evita o estouro

    n_genes_zero = int((soma_coluna == 0).sum())
    densidade = nnz_total / (n_rows * n_cols) if n_rows and n_cols else 0.0
    return dict(n_nan=n_nan, n_inf=n_inf, n_neg=n_neg, n_frac=n_frac,
                densidade=densidade, n_celulas_zero=n_celulas_zero, n_genes_zero=n_genes_zero)


def reportar_estatisticas(rel, etapa, stats, esperar_inteiro, negativo_e_erro=True):
    sujo = bool(stats["n_nan"] or stats["n_inf"])
    extra = " — Inf é o sintoma clássico de size factor=0 no scran" if (stats["n_inf"] and not negativo_e_erro) else ""
    rel.add(etapa, "NaN / Inf", "FALHA" if sujo else "OK", f"NaN={stats['n_nan']}, Inf={stats['n_inf']}{extra}")

    if negativo_e_erro:
        rel.add(etapa, "valores negativos", "FALHA" if stats["n_neg"] else "OK",
                f"{stats['n_neg']} negativos (contagem não pode ser negativa)")
    else:
        rel.add(etapa, "valores negativos", "AVISO" if stats["n_neg"] else "OK",
                f"{stats['n_neg']} negativos (ok se centrado/escalado; inesperado se for só log-normalização)")

    if esperar_inteiro:
        rel.add(etapa, "valores inteiros", "AVISO" if stats["n_frac"] else "OK",
                f"{stats['n_frac']} valores não-inteiros (esperado: contagens brutas)")

    rel.add(etapa, "densidade da matriz", "OK", f"{stats['densidade']*100:.3f}% não-zero")
    rel.add(etapa, "células com soma=0", "AVISO" if stats["n_celulas_zero"] else "OK",
            f"{stats['n_celulas_zero']} células")
    rel.add(etapa, "genes com soma=0", "AVISO" if stats["n_genes_zero"] else "OK",
            f"{stats['n_genes_zero']} genes")


# ── Abridores: cada um devolve um leitor de blocos + metadados leves ───────
def abrir_raw_10x(path, rel, etapa):
    if not Path(path).exists():
        rel.add(etapa, "arquivo existe", "FALHA", f"não encontrado: {path}")
        return None
    try:
        f = h5py.File(path, "r")
        grp = f["matrix"]
        n_genes, n_cells = (int(x) for x in grp["shape"][:])
        indptr = grp["indptr"][:]          # leve: n_cells+1 inteiros, ok carregar inteiro
        data_ds, indices_ds = grp["data"], grp["indices"]

        def decode(arr):
            return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

        barcodes = decode(grp["barcodes"][:]) if "barcodes" in grp else None
        if "features" in grp and "id" in grp["features"]:
            genes = decode(grp["features"]["id"][:])
        elif "genes" in grp:
            genes = decode(grp["genes"][:])
        else:
            genes = None

        def leitor_bloco(c_start, c_end):
            p0, p1 = int(indptr[c_start]), int(indptr[c_end])
            data_b = data_ds[p0:p1]            # leitura em fatia -- só esse pedaço vem do disco
            indices_b = indices_ds[p0:p1]
            indptr_b = indptr[c_start:c_end + 1] - indptr[c_start]
            bloco_genes_x_celulas = sp.csc_matrix((data_b, indices_b, indptr_b),
                                                   shape=(n_genes, c_end - c_start))
            return bloco_genes_x_celulas.T.tocsr()   # bloco PEQUENO -- transpor aqui é barato

        rel.add(etapa, "carregamento", "OK", f"{n_cells:,} x {n_genes:,} (leitura em blocos, h5py)")
        return dict(fechar=f.close, leitor=leitor_bloco, n_obs=n_cells, n_vars=n_genes,
                    obs_names=barcodes, var_names=genes)
    except Exception as e:
        rel.add(etapa, "carregamento", "FALHA", f"erro ao abrir: {e}")
        return None


def abrir_h5ad_backed(path, rel, etapa):
    try:
        import anndata as ad
    except ImportError:
        rel.add(etapa, "import anndata", "FALHA", "anndata não instalado")
        return None
    if not Path(path).exists():
        rel.add(etapa, "arquivo existe", "FALHA", f"não encontrado: {path}")
        return None
    try:
        adata = ad.read_h5ad(path, backed="r")   # modo backed: X fica no disco, só lê o que pedir

        def leitor_bloco(start, end):
            bloco = adata.X[start:end]
            return bloco if sp.issparse(bloco) else sp.csr_matrix(np.asarray(bloco))

        rel.add(etapa, "carregamento", "OK", f"{adata.n_obs:,} x {adata.n_vars:,} (modo backed)")
        obs_names = list(adata.obs_names)
        var_names = list(adata.var_names)
        n_obs, n_vars = adata.n_obs, adata.n_vars

        def fechar():
            try:
                adata.file.close()
            except Exception:
                pass

        return dict(fechar=fechar, leitor=leitor_bloco, n_obs=n_obs, n_vars=n_vars,
                    obs_names=obs_names, var_names=var_names)
    except Exception as e:
        rel.add(etapa, "carregamento", "FALHA", f"erro ao abrir: {e}")
        return None


def processar_etapa_matriz(abrir_fn, path, rel, etapa, esperar_inteiro, negativo_e_erro=True):
    info = abrir_fn(path, rel, etapa)
    if info is None:
        return None

    stats = processar_em_chunks(info["leitor"], info["n_obs"], info["n_vars"], CHUNK_SIZE, esperar_inteiro)
    reportar_estatisticas(rel, etapa, stats, esperar_inteiro, negativo_e_erro)

    meta = MetaEtapa(
        nome=etapa, n_obs=info["n_obs"], n_vars=info["n_vars"],
        var_names=info["var_names"],
        obs_names=set(info["obs_names"]) if info["obs_names"] else None,
    )

    info["fechar"]()
    del info, stats
    gc.collect()   # libera de fato a memória do bloco anterior antes da próxima etapa
    return meta


# ── PCA e labels finais: pequenos, carregamento direto sem chunking ────────
def checar_pca(arr, rel, etapa, n_pcs_esperado):
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    sujo = bool(n_nan or n_inf)
    rel.add(etapa, "NaN / Inf", "FALHA" if sujo else "OK", f"NaN={n_nan}, Inf={n_inf}")

    rel.add(etapa, "n_componentes", "OK" if arr.shape[1] == n_pcs_esperado else "AVISO",
            f"{arr.shape[1]} colunas (esperado: {n_pcs_esperado})")

    variancias = np.nanvar(arr, axis=0)
    decrescente = bool(np.all(np.diff(variancias) <= 1e-6))
    rel.add(etapa, "variância decrescente por PC",
            "OK" if (decrescente and not sujo) else "AVISO",
            f"PC1 var={variancias[0]:.3f} ... PC{len(variancias)} var={variancias[-1]:.3f}"
            + ("" if decrescente else " — PCA real é SEMPRE decrescente por construção; "
                                       "se falhar aqui, é sinal forte de colunas fora de ordem")
            + (" [contém NaN/Inf — só informativo]" if sujo else ""))

    medias = np.nanmean(arr, axis=0)
    fora_do_centro = bool((np.abs(medias) > 0.5).any())
    rel.add(etapa, "centragem (média ~0 por PC)",
            "AVISO" if (fora_do_centro or sujo) else "OK",
            f"|média| máxima = {np.abs(medias).max():.4f}"
            + (" [contém NaN/Inf — só informativo]" if sujo else ""))


# ── Consistência ENTRE etapas — só com os metadados leves (MetaEtapa) ──────
def checar_consistencia(rel, metas: dict, n_pca: int):
    raw, filt = metas.get("inicial"), metas.get("filtrado")
    clus, norm = metas.get("cluster_init"), metas.get("normalizado")

    if raw and filt:
        ok = filt.n_obs <= raw.n_obs
        rel.add("pipeline", "raw -> filtrado", "OK" if ok else "FALHA",
                f"{raw.n_obs:,} -> {filt.n_obs:,}" + ("" if ok else " — IMPOSSÍVEL: aumentou"))

    if filt and clus:
        ok = filt.n_obs == clus.n_obs
        rel.add("pipeline", "filtrado == cluster_init", "OK" if ok else "FALHA",
                f"{filt.n_obs:,} vs {clus.n_obs:,}"
                + ("" if ok else " — clustering não deveria alterar n de células"))

    if filt and norm:
        if filt.n_obs == norm.n_obs:
            rel.add("pipeline", "filtrado -> normalizado", "OK", f"{filt.n_obs:,} células preservadas")
        else:
            diff = filt.n_obs - norm.n_obs
            status = "AVISO" if 0 < diff < 0.01 * filt.n_obs else "FALHA"
            rel.add("pipeline", "filtrado -> normalizado", status,
                    f"{filt.n_obs:,} -> {norm.n_obs:,} ({diff:+,}) — confirme se a perda é esperada")

    if norm and n_pca is not None:
        ok = norm.n_obs == n_pca
        rel.add("pipeline", "normalizado == pca [CRÍTICO]", "OK" if ok else "FALHA",
                f"{norm.n_obs:,} vs {n_pca:,}" + ("" if ok else " — MISMATCH"))

    if filt and clus and filt.var_names and clus.var_names:
        iguais = filt.var_names == clus.var_names
        rel.add("pipeline", "genes: filtrado == cluster_init", "OK" if iguais else "FALHA",
                "" if iguais else "ordem ou conjunto de genes mudou")

    if filt and norm and filt.var_names and norm.var_names:
        subset = set(norm.var_names).issubset(set(filt.var_names))
        rel.add("pipeline", "genes: normalizado ⊆ filtrado", "OK" if subset else "FALHA",
                f"{norm.n_vars:,} de {filt.n_vars:,}" + ("" if subset else " — genes inexistentes no filtrado"))

    for nome, m in [("filtrado", filt), ("cluster_init", clus), ("normalizado", norm)]:
        if m is not None and m.obs_names is not None:
            n_dup = m.n_obs - len(m.obs_names)
            rel.add(nome, "barcodes duplicados", "FALHA" if n_dup else "OK", f"{n_dup} duplicados")

    if filt and norm and filt.obs_names and norm.obs_names:
        subset = norm.obs_names.issubset(filt.obs_names)
        rel.add("pipeline", "barcodes: normalizado ⊆ filtrado", "OK" if subset else "FALHA",
                "" if subset else "barcodes em normalizado que não vieram do filtrado")


# ── Pipeline ────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("  VALIDAÇÃO DE INTEGRIDADE (v2 — streaming) — pipeline trabalho-cad-v2")
    print(f"  CHUNK_SIZE = {CHUNK_SIZE:,} células/bloco")
    print("=" * 78)
    rel = Relatorio()
    metas = {}

    print("\n-- inicial (raw 10x) --")
    metas["inicial"] = processar_etapa_matriz(abrir_raw_10x, RAW_H5, rel, "inicial", esperar_inteiro=True)

    print("\n-- pre-processado (filtrado) --")
    metas["filtrado"] = processar_etapa_matriz(abrir_h5ad_backed, FILTRADO, rel, "filtrado", esperar_inteiro=True)

    print("\n-- clusterizado-inicial --")
    metas["cluster_init"] = processar_etapa_matriz(abrir_h5ad_backed, CLUSTER_INIT, rel, "cluster_init",
                                                    esperar_inteiro=True)

    print("\n-- normalizado (scran) --")
    metas["normalizado"] = processar_etapa_matriz(abrir_h5ad_backed, NORMALIZADO, rel, "normalizado",
                                                    esperar_inteiro=False, negativo_e_erro=False)

    print("\n-- pca (pequeno, carregamento direto) --")
    pca_arr = None
    if PCA_SCORES.exists():
        try:
            pca_arr = np.load(PCA_SCORES)
            rel.add("pca", "carregamento", "OK", f"shape={pca_arr.shape}")
            checar_pca(pca_arr, rel, "pca", N_PCS_ESPERADO)
        except Exception as e:
            rel.add("pca", "carregamento", "FALHA", str(e))
    else:
        rel.add("pca", "arquivo existe", "FALHA", f"não encontrado: {PCA_SCORES}")

    print("\n-- consistência entre etapas --")
    checar_consistencia(rel, metas, pca_arr.shape[0] if pca_arr is not None else None)

    if LABELS_FINAIS.exists():
        print("\n-- labels finais de cluster (opcional) --")
        try:
            npz = np.load(LABELS_FINAIS)
            for key in npz.files:
                labels = npz[key]
                if pca_arr is None:
                    status = "AVISO"
                elif len(labels) == pca_arr.shape[0]:
                    status = "OK"
                else:
                    status = "FALHA"
                rel.add("labels_finais", f"{key}: n == pca", status, f"{len(labels):,} labels")
        except Exception as e:
            rel.add("labels_finais", "carregamento", "FALHA", str(e))

    n_ok, n_warn, n_fail = rel.resumo()
    print("\n" + "=" * 78)
    print(f"  RESUMO: {n_ok} OK | {n_warn} AVISOS | {n_fail} FALHAS")
    print("=" * 78)
    rel.salvar_json(RELATORIO_OUT)
    print(f"  Relatório completo salvo em: {RELATORIO_OUT}")

    if n_fail:
        print("\n  >> Existem FALHAS. Não confie nos resultados de clustering até resolver.")
        sys.exit(1)
    elif n_warn:
        print("\n  >> Sem falhas críticas, mas revise os avisos acima.")
    else:
        print("\n  >> Pipeline íntegro em todas as checagens realizadas.")


if __name__ == "__main__":
    main()
