import scanpy as sc
import numpy as np

CAMINHO = (
    r"C:\Users\igorm\Documents\programacao\trabalho-cad-v2"
    r"\dataset\pre-processado\dataset_filtrado.h5ad"
)

print("Carregando dataset...")
adata = sc.read_h5ad(CAMINHO)

n_cells, n_genes = adata.n_obs, adata.n_vars
data       = adata.X.data
indices    = adata.X.indices
actual_nnz = len(data)

# Converte pra int64 e corrige overflow de int32
indptr_i64 = adata.X.indptr.astype(np.int64)
indptr_i64 = np.where(indptr_i64 < 0, indptr_i64 + 2**32, indptr_i64)
indptr     = np.minimum(indptr_i64, actual_nnz)

print(f"  NNZ real    : {actual_nnz:,}")
print(f"  indptr[-1]  : {int(adata.X.indptr[-1]):,}  (overflow detectado e corrigido)\n")

total_entradas = n_cells * n_genes
pct_zeros      = (1 - actual_nnz / total_entradas) * 100

counts_por_celula = np.diff(indptr).astype(np.int32)

cumdata         = np.concatenate([[0.0], np.cumsum(data.astype(np.float64))])
soma_por_celula = cumdata[indptr[1:]] - cumdata[indptr[:-1]]
del cumdata

gene_n_cells = np.bincount(indices, minlength=n_genes).astype(np.int32)
gene_totals  = np.bincount(indices, weights=data.astype(np.float64), minlength=n_genes)
gene_mean    = gene_totals / n_cells

print("=" * 58)
print("  ANÁLISE DO DATASET PRÉ-PROCESSADO")
print("=" * 58)

print(f"\n  Dimensões")
print(f"    Células      : {n_cells:>15,}")
print(f"    Genes        : {n_genes:>15,}")
print(f"    Total slots  : {total_entradas:>15,}")

print(f"\n  Sparsidade")
print(f"    NNZ             : {actual_nnz:>15,}  ({100-pct_zeros:.2f}%)")
print(f"    Zeros           : {total_entradas-actual_nnz:>15,}  ({pct_zeros:.2f}%)")

print(f"\n  Genes expressos por célula")
print(f"    Mínimo   : {counts_por_celula.min():>10,}")
print(f"    Mediana  : {int(np.median(counts_por_celula)):>10,}")
print(f"    Média    : {counts_por_celula.mean():>10.1f}")
print(f"    Máximo   : {counts_por_celula.max():>10,}")

print(f"\n  Counts totais por célula (UMIs)")
print(f"    Mínimo   : {soma_por_celula.min():>10.0f}")
print(f"    Mediana  : {np.median(soma_por_celula):>10.0f}")
print(f"    Média    : {soma_por_celula.mean():>10.1f}")
print(f"    Máximo   : {soma_por_celula.max():>10.0f}")

print(f"\n  Cobertura por gene")
print(f"    Genes em 0 células    : {(gene_n_cells == 0).sum():>8,}")
print(f"    Genes em < 10 células : {(gene_n_cells < 10).sum():>8,}")
print(f"    Genes em > 1k células : {(gene_n_cells > 1000).sum():>8,}")
print(f"    Mediana células/gene  : {int(np.median(gene_n_cells)):>8,}")

print(f"\n  Comparação com o paper")
print(f"    {'Métrica':<25} {'Paper':>12}  {'Atual':>12}")
print(f"    {'-'*50}")
print(f"    {'Células':<25} {'1,232,055':>12}  {n_cells:>12,}")
print(f"    {'Genes':<25} {'11,720':>12}  {n_genes:>12,}")
print("=" * 58)