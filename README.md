# 🧬 Reimplementação heterogênea (GPU + CPU multicore) do pipeline mbkmeans para scRNA-seq

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.x-76B900?logo=nvidia&logoColor=white)
![CuPy](https://img.shields.io/badge/CuPy-14.1-1A1A1A)
![Numba](https://img.shields.io/badge/Numba-0.65-00A3E0)
![Scanpy](https://img.shields.io/badge/Scanpy-1.11-3776AB)
![Speedup](https://img.shields.io/badge/speedup-~232×-C1121F)

> 🌐 *English version below — [jump to it](#-english-version).*

> Reimplementação em **GPU (CuPy/CUDA)** e **CPU multicore (Numba/OpenMP)** do pipeline de análise de
> RNA de célula única (scRNA-seq) de **Hicks et al. (2021)**, originalmente executado em CPU single-thread.
> Mais do que acelerar, o trabalho investiga **onde a GPU realmente compensa** — e o que o *speedup*
> revela sobre a estrutura dos dados.

**Disciplina:** Computação de Alto Desempenho — UFG, Instituto de Informática
**Professor:** Ricardo Augusto Pereira Franco

| Integrante | Matrícula |
|---|---|
| Christian de Souza Ramos | 202403897 |
| Giulio Henrique Borges Basso | 202403904 |
| Igor Garbin Manzan Mazo | 202406401 |
| Wallisson Policarpo Teodoro | 202403930 |

🔗 Repositório: https://github.com/IgorGMMazo/cuda-multi-k-means-scrna

---

## 📑 Sumário

- [Visão geral](#-visão-geral)
- [O dado e o pipeline](#-o-dado-e-o-pipeline)
- [Princípio de arquitetura](#-princípio-de-arquitetura-onde-a-gpu-compensa)
- [Resultados](#-resultados)
- [Estrutura do repositório](#-estrutura-do-repositório)
- [Os códigos, em detalhe](#-os-códigos-em-detalhe)
- [Fluxo de dados](#-fluxo-de-dados-ordem-de-execução)
- [Como executar](#-como-executar)
- [Validação numérica](#-validação-numérica)
- [Achado principal](#-achado-principal-wcss-vs-silhouette)
- [Referências](#-referências)

---

## 🔭 Visão geral

O sequenciamento de RNA de célula única (scRNA-seq) gera matrizes de **milhões de células × dezenas de
milhares de genes**. Identificar subpopulações celulares exige *clustering* não supervisionado — e o
K-means clássico se torna inviável nessa escala por exigir toda a matriz em memória.

**Hicks et al. (2021)** resolveram o problema de memória com o pacote
[`mbkmeans`](https://bioconductor.org/packages/mbkmeans) (R/Bioconductor), um Mini-Batch K-Means *on-disk*.
Porém, todo o pipeline roda em **CPU single-thread** (~104 h), dominado pelo PCA-IRLBA (96 h) e pela
normalização (5,18 h).

Neste trabalho reimplementamos o pipeline em **arquitetura heterogênea**: GPU para os trechos limitados
por cálculo, CPU multicore para os trechos limitados por movimentação de memória. Reduzimos as 5 etapas
de **~102 h para ~26 min (≈ 232×)** e, sobretudo, **medimos etapa a etapa onde a GPU compensa e por quê** —
uma análise de *trade-off* que o paper (CPU-only) não faz.

> **Referência:** Hicks, S. C., Liu, R., Ni, Y., Purdom, E., & Risso, D. (2021). *mbkmeans: Fast clustering
> for single cell data using mini-batch k-means.* PLOS Computational Biology, 17(1), e1008625.
> https://doi.org/10.1371/journal.pcbi.1008625

---

## 🧩 O dado e o pipeline

**Entrada:** matriz esparsa de **~1,3 milhão de células × ~28 mil genes** (~55 GB), armazenada em HDF5 no
formato CSR (~2,4 bilhões de não-zeros) — o dataset *1.3M mouse brain cells* da 10x Genomics.

O pipeline executa cinco etapas encadeadas:

| # | Etapa | Implementação | Recurso |
|---|---|---|---|
| 1 | Pré-processamento (QC / filtragem) | prefix-sums sobre CSR + montagem da matriz | GPU (cálculo) + CPU (cópia) |
| 2 | Clustering inicial (mini-batch K-means) | atribuição/somas/update na VRAM | GPU |
| 3 | Normalização (scran / deconvolution) | deconvolution por bloco | GPU |
| 4 | PCA (50 componentes) | covariância gene×gene + `eigh` | GPU |
| 5 | Clusterização final (Lloyd + busca de K) | GEMM denso por iteração | GPU |

---

## ⚙️ Princípio de arquitetura: onde a GPU compensa

> **A GPU só compensa quando o gargalo é o cálculo.**

O dado viaja `SSD → RAM → VRAM → cálculo`. Se o tempo é gasto **movendo** dado (I/O de disco ou
transferência PCIe), acelerar o cálculo quase não muda o tempo total — a GPU fica ociosa esperando
(*data starvation*). Ela só rende quando o cálculo é **pesado, regular e reutiliza** dados já transferidos.

| Etapa | Gargalo | Natureza | GPU compensa? |
|---|---|---|:---:|
| Pré-proc. (QC) | SSD → RAM (I/O) | I/O-bound | Pouco |
| Clustering inicial | PCIe (repetido) | Transfer-bound + esparso | **Não** |
| Normalização | Cálculo por bloco | Compute-bound | Sim |
| PCA | Algorítmico | Compute-bound, denso | **Sim, muito** |
| Clusterização final | Cálculo denso | Compute-bound, denso | **Sim, decisivo** |

---

## 📊 Resultados

Speedup por etapa (baseline = tempos reportados por Hicks et al., CPU single-thread):

| Etapa | Baseline | Proposta | Speedup |
|---|---|---|---|
| Pré-processamento (QC) | 8 min | 4 min 40 s | ~1,71× |
| Clustering inicial | 8 min 30 s | 7 min 14 s | ~1,17× |
| Normalização (scran) | 5,18 h | 8 min | 38,85× |
| PCA | 96 h | 5 min 20 s | **1080×** |
| Clusterização final (busca de K) | ~30 s | ~70 s | ~0,43× |
| **Total (5 etapas)** | **~102,3 h** | **~26,47 min** | **~231,91×** |

> O maior ganho (PCA) é **majoritariamente algorítmico**: trocar o SVD iterativo *out-of-core* (IRLBA) por
> uma covariância em duas passadas esmagaria o baseline mesmo em CPU. A GPU atua como multiplicador.

**Ambiente experimental:** NVIDIA GeForce RTX 5070 Laptop (Blackwell, 4608 CUDA cores, GDDR7), 8 GB VRAM,
24 CPU-cores, 32 GB RAM, SSD NVMe; `cupy-cuda13x` 14.1.0, `numba` 0.65.1, Python 3.11.

---

## 🗂️ Estrutura do repositório

```
trabalho-cad-v2/
├── README.md
├── requirements.txt
├── gerar_apresentacao.py            # gera o .pptx final (lê imagens de metricas-etapas/)
│
├── entregaveis/                     # PDFs finais entregues na disciplina
│   ├── Artigo_final_CAD.pdf
│   ├── Apresentacao_final_CAD.pdf
│   └── Resumo_final_CAD.pdf
│
├── dataset/                         # artefatos de dados (um por etapa) — fora do git (.gitignore)
│   ├── inicial/                     # entrada: HDF5 bruto 10x (~55 GB)
│   │   └── 1M_neurons_filtered_gene_bc_matrices_h5.h5
│   ├── pre-processado/              # saída da etapa 1
│   │   └── dataset_filtrado.h5ad
│   ├── clusterizado-inicial/        # saída da etapa 2 (labels do clustering inicial)
│   │   └── clustering_all_genes.h5ad
│   ├── normalizado/                 # saída da etapa 3 (counts log-normalizados + size factors)
│   │   └── normalizado_scran.h5ad
│   ├── pca/                         # saída da etapa 4, baseline com TODOS os genes
│   │   └── pca.npy                  # scores (n_células × 50 PCs)
│   └── pca_hvg/                     # saída da etapa 4, sobre top-2000 HVG (ver seção HVG abaixo)
│       ├── pca.npy                  # scores (n_células × 50 PCs)
│       ├── hvg_mask.npy             # bool (n_genes_total,) — quais genes viraram HVG
│       └── hvg_gene_ids.npy         # nomes dos genes selecionados
│
├── src/
│   ├── paralelizacao/               # PIPELINE PRINCIPAL (as 5 etapas)
│   │   ├── pre-processamento.py     # Etapa 1 — QC: GPU (Pass 1) + CPU multicore (Pass 2)
│   │   ├── clusteringAllv2.py       # Etapa 2 — mini-batch K-means (CuPy)
│   │   ├── normalizar.py            # Etapa 3 — scran/deconvolution (CuPy)
│   │   ├── pca-gpu.py               # Etapa 4 — PCA por covariância (CuPy + cuSOLVER), sobre HVG
│   │   └── clustering-finalv2.py    # Etapa 5 — Multi-K Lloyd + busca de K (CuPy)
│   │
│   └── analise/                     # ANÁLISES E VALIDAÇÕES (fora do caminho crítico)
│       ├── analise.py               # estatísticas descritivas do dataset filtrado
│       ├── validar_inetgridade.py   # validação de integridade ponta a ponta (streaming)
│       ├── pca_analise.py           # scatter rápido PC1 × PC2
│       ├── teste.py                 # sanity-check do pca.npy (NaN / variância decrescente)
│       ├── umap_vs_pca.py           # PC1×PC2 vs UMAP, todos os genes (k=15)
│       ├── umap_vs_pca_multi_k.py   # PC1×PC2 vs UMAP, todos os genes, varrendo K de 1 a 20
│       ├── hvg_pca_silhouette.py    # HVG vs todos-os-genes: silhouette + UMAP (exploratório)
│       ├── umap_vs_pca_hvg.py       # PC1×PC2 vs UMAP em hexbin, sobre HVG (estilo Fig.5A do paper)
│       └── investiga_variancia_hvg.py  # diagnóstico do bug de precisão do PCA-HVG (ver nota abaixo)
│
└── metricas-etapas/                 # métricas (JSON), gráficos (PNG) e labels de saída
    ├── metricas_*.json              # tempos/VRAM por etapa (uma por etapa do pipeline)
    ├── resultados_multi_k_todos_genes.json  # WCSS/silhouette por K — clustering final, todos os genes
    ├── grafico_metricas_todos_genes.png     # WCSS/n vs silhouette — todos os genes
    ├── labels_multi_k.npz           # labels finais por K — todos os genes
    ├── clusters_2d_k15.png          # projeção PCA (PC1×PC2) colorida por k=15 — todos os genes
    ├── relatorio_integridade.json   # veredito da validação ponta a ponta
    ├── analise_clusterabilidade.json  # sessão exploratória que motivou usar HVG (ver pca-gpu.py)
    ├── umap_vs_pca.png              # saída de src/analise/umap_vs_pca.py
    ├── hvg_vs_allgenes_umap.png     # saída de src/analise/hvg_pca_silhouette.py
    ├── umap_multi_k/                # saída de src/analise/umap_vs_pca_multi_k.py (K=1..20)
    └── hvg/                         # resultados do pipeline rodado sobre HVG (top-2000 genes)
        ├── resultados_multi_k.json  # WCSS/silhouette por K
        ├── grafico_metricas.png     # WCSS/n vs silhouette
        ├── labels_multi_k.npz       # labels finais por K
        └── umap_vs_pca_hvg_k16_hexbin.png  # UMAP hexbin estilo Fig.5A do paper (k=16)
```

> **Nota sobre o PCA-HVG:** a primeira versão de `pca-gpu.py` calculava a covariância como um único
> matmul sobre 1,15M linhas em float32, sofrendo cancelamento catastrófico na centralização
> (`M = S − n·μμᵀ`) — o que corrompia os componentes principais silenciosamente (PC1+PC2 saía em 7,6%
> da variância em vez dos ~36% esperados, e o silhouette do clustering final ficava artificialmente
> baixo). Corrigido somando a covariância em chunks convertidos pra float64 antes de acumular; ver o
> cabeçalho de `src/paralelizacao/pca-gpu.py` e `src/analise/investiga_variancia_hvg.py` para os detalhes.

> **Caminhos portáveis:** os scripts do pipeline resolvem os caminhos **relativos à raiz do repositório**
> (`BASE = Path(__file__).resolve().parents[2]`), apontando para a estrutura `dataset/` e `metricas-etapas/`
> mostrada acima. Não é preciso editar caminhos absolutos por máquina — basta manter o layout de pastas e
> colocar o HDF5 bruto em `dataset/inicial/`.

---

## 🧠 Os códigos, em detalhe

### Pipeline principal — `src/paralelizacao/`

#### `pre-processamento.py` — Etapa 1: QC (GPU + CPU multicore)
Varre o HDF5 bruto em **dois passes**, por *chunks* de 100 mil células:
- **Pass 1 (GPU/CuPy):** calcula métricas de QC — contagem de genes por célula e fração mitocondrial —
  com *prefix-sums* (`cumsum`) direto sobre os arranjos CSR, **sem densificar** a matriz. Decide quais
  células/genes ficam (`MIN_GENES=500`, `MAX_PCT_MT=5%`, expressão média mínima por gene).
- **Pass 2 (CPU/Numba + OpenMP):** monta a matriz filtrada num buffer pré-alocado. É movimentação de
  memória, paralelizada por célula (`@njit(parallel=True)`), embaraçosamente paralela e sem *race
  condition*. Cuidados de robustez: `gene_mask` em `uint8` e `indptr`/`dst_starts` em `int64` (o NNZ
  ultrapassa o limite de `int32`).
- Inclui um `MetricsMonitor` que amostra RAM em *thread* separada e salva `metricas.json`/`.txt`.
- **Entrada:** `dataset/inicial/*.h5` → **Saída:** `dataset/pre-processado/dataset_filtrado.h5ad`.

#### `clusteringAllv2.py` — Etapa 2: mini-batch K-means (CuPy)
Clustering inicial que serve de **insumo para o scran** (não é o resultado final). Roda na GPU:
- Normas das células via `np.add.reduceat` na **CPU** (O(NNZ), zero PCIe).
- Inicialização **K-Means++** num subconjunto, centroides residentes na VRAM.
- *Mini-batch* com **buffers *pinned*** (page-locked) para DMA eficiente; o batch permanece **esparso** —
  só a matriz `(b, k)` de distâncias é densa. *Cluster-sums* via *one-hot* esparso.
- *Update* agregado de Sculley, convergência por *shift* relativo de Frobenius (EWA).
- `k=15`, `batch_size=5000`. **Saída:** `clustering_all_genes.h5ad` com `obs["cluster_all_genes"]`.

#### `normalizar.py` — Etapa 3: scran / deconvolution (CuPy)
Reimplementação **fiel** do algoritmo de *deconvolution* do scran (Lun et al., 2016) — o paper terceiriza
essa etapa, então a referência aqui é o scran, não o mbkmeans. Por bloco de cluster (≤ `max_block=3000`):
pré-escala por *library size* → ordena por *library* → pseudo-célula de referência → `cumsum` (P) →
pools circulares (tamanhos 21–101) → `mediana(pool/ref)` → resolve `A·β = R` por **LSQR na GPU** (com
*fallback* por equações normais) → *size factors* → pseudo-célula. Depois: reescala entre blocos,
centraliza em média 1 e aplica `log1p(counts/sf)` *in-place* em *chunks*.
- **Saída:** `normalizado_scran.h5ad` com `obs["size_factors"]`.

#### `pca-gpu.py` — Etapa 4: PCA por covariância (CuPy + cuSOLVER)
Substitui o IRLBA *out-of-core* (96 h) por covariância gene×gene + eigendecomposição. Como
`n (1,1 M) ≫ p (13,9 k)`, a matriz `p×p` (~1,5 GB) cabe na VRAM. **Duas passadas:**
- **Passe 1:** acumula `S = XᵀX` (esparso, por batch) e as somas de coluna (média μ);
  `M = S − n·μμᵀ` centra **sem densificar** X.
- `eigh(M)` em **float64** → top-50 autovetores (loadings), com sinal determinístico igual ao sklearn.
- **Passe 2:** projeta `scores = (X − μ)·V` em *streaming*; reordena por variância empírica (rede de
  segurança para espectro plano).
- *Guards* contra NaN/Inf na entrada e nos loadings. **Saída:** `dataset/pca/pca.npy` + `metricas_pca.json`.

#### `clustering-finalv2.py` — Etapa 5: Multi-K Lloyd + busca de K (CuPy)
Após o PCA, a matriz `1,1 M × 50` (densa, ~220 MB) cabe inteira na VRAM → usa **Lloyd completo** (cada
iteração é um único GEMM). Contribuição além do paper (que fixa K=15): **busca sistemática sobre K**.
- **K-Means++ com Gumbel-Max trick** (amostragem ponderada 100% na GPU) e **reuso de prefixo**: pré-computa
  cadeias longas até `K_max` e fatia os primeiros `k` centroides para cada K — evita recomputar a init.
- `n_init=3` *restarts*, convergência por *shift* de centroides.
- **Silhouette amostrado** na GPU (métrica ausente no paper) + **método do cotovelo** (2ª derivada).
- Um *worker* por K em *thread*/*stream* CUDA concorrente, com limpeza explícita de VRAM.
- **Saída:** `labels_multi_k.npz`, `resultados_multi_k*.json`, `grafico_metricas*.png`.

### Análises e validações — `src/analise/`

| Script | O que faz | Saída |
|---|---|---|
| `validar_inetgridade.py` | Percorre `raw → filtrado → cluster → normalizado → PCA` em **streaming** (memória limitada a um *chunk*) e confere consistência entre etapas: contagem de células, conjunto/ordem de genes, *barcodes* duplicados, NaN/Inf, e o alinhamento crítico PCA ↔ normalizado. Emite veredito OK/AVISO/FALHA. | `relatorio_integridade.json` |
| `analise.py` | Estatísticas descritivas do dataset filtrado (esparsidade, genes/célula, UMIs/célula, cobertura por gene) e comparação com os números do paper. | *(stdout)* |
| `pca_analise.py` | *Scatter* rápido PC1 × PC2 a partir do `pca.npy`. | *(janela matplotlib)* |
| `teste.py` | *Sanity-check* do `pca.npy`: zero NaN e variância decrescente por PC. | *(stdout)* |
| `umap_vs_pca.py` | Painel PC1×PC2 vs **UMAP** dos 50 PCs (k=15), evidenciando que a diferença de "borrão vs ilhas" é de **lente**, não de dados. | `metricas-etapas/umap_vs_pca.png` |
| `umap_vs_pca_multi_k.py` | Versão em **loop de K=1 a 20** do anterior. O embedding UMAP **independe de K**, então é calculado uma única vez e reaproveitado; só a coloração (labels do k-means) muda. | `metricas-etapas/umap_multi_k/umap_vs_pca_kNN.png` |
| `hvg_pca_silhouette.py` | Compara o silhouette usando **todos os genes** vs **top-2000 HVG** (Seurat), com PCA por covariância em CPU e UMAP lado a lado — testa se a não-clusterabilidade é artefato de usar todos os genes. | `metricas-etapas/hvg_vs_allgenes_umap.png` |

> ℹ️ **Sobre o UMAP:** os scripts de UMAP são **análises exploratórias de apoio** e *não fazem parte* do
> escopo central do trabalho (que vai até o silhouette/projeção PCA). Estão incluídos por completude.

---

## 🔁 Fluxo de dados (ordem de execução)

```
dataset/inicial/*.h5
        │  (1) pre-processamento.py
        ▼
dataset/pre-processado/dataset_filtrado.h5ad
        │  (2) clusteringAllv2.py
        ▼
dataset/clusterizado-inicial/clustering_all_genes.h5ad   (+ obs["cluster_all_genes"])
        │  (3) normalizar.py
        ▼
dataset/normalizado/normalizado_scran.h5ad               (+ obs["size_factors"])
        │  (4) pca-gpu.py
        ▼
dataset/pca/pca.npy                                       (n_células × 50 PCs)
        │  (5) clustering-finalv2.py
        ▼
metricas-etapas/labels_multi_k.npz + gráficos + JSON
```

A qualquer momento, `src/analise/validar_inetgridade.py` valida o encadeamento ponta a ponta.

---

## ▶️ Como executar

> **Pré-requisitos:** Python 3.11, GPU NVIDIA com CUDA 13.x, e os dados em `dataset/inicial/`.

```bash
# 1. ambiente
python -m venv venv
venv\Scripts\activate            # Windows (PowerShell/CMD)
# source venv/bin/activate       # Linux/macOS
pip install -r requirements.txt

# 2. coloque o HDF5 bruto em dataset/inicial/ (os demais caminhos são relativos à raiz do repo)

# 3. execute as etapas em ordem
python src/paralelizacao/pre-processamento.py
python src/paralelizacao/clusteringAllv2.py
python src/paralelizacao/normalizar.py
python src/paralelizacao/pca-gpu.py
python src/paralelizacao/clustering-finalv2.py

# 4. (opcional) valide a integridade do pipeline
python src/analise/validar_inetgridade.py
```

Dependências principais (ver `requirements.txt` completo): `cupy-cuda13x`, `numba`, `scanpy`, `anndata`,
`h5py`, `scipy`, `scikit-learn`, `numpy`, `matplotlib`, `umap-learn`, `psutil`.

---

## ✅ Validação numérica

- **Normalização (scran):** correlação **0,99** (1 cluster) e **0,997** (multi-cluster) contra fatores de
  tamanho conhecidos; erro ~2,6–6% (vs ~15–23% da normalização simples por *library size*).
- **PCA:** correlação **1,000000** vs `scikit-learn` nos 50 componentes — reformulação matematicamente exata.
- **Integridade ponta a ponta:** **1.148.558 células** consistentes em todas as etapas, **0** *barcodes*
  duplicados, **13.897 genes** consistentes.

---

## 🔬 Achado principal: WCSS vs silhouette

O paper avalia o agrupamento apenas por **WCSS**, que decresce monotonicamente com K e não mede separação.
Ao introduzir o **silhouette** (ausente no paper), observamos um pico isolado em **K=2 (~0,20)** e colapso
para **≈ 0** nos demais K, consistente entre rodadas independentes (sinal real, não ruído). Por convenção,
silhouette < 0,25 indica estrutura de cluster fraca ou ausente.

**Hipótese:** dados de neurônios formam um **contínuo de diferenciação**, não clusters discretos. O K-means
(distância euclidiana) assume clusters globulares — num gradiente, qualquer K produz silhouette baixo. Isso
**não invalida** a implementação; é uma limitação conhecida do método frente ao regime do dado. *Trabalho
futuro:* GMM e métodos baseados em grafo (Louvain/Leiden sobre KNN).

---

## 📚 Referências

1. Hicks, S. C., Liu, R., Ni, Y., Purdom, E., & Risso, D. (2021). *mbkmeans: Fast clustering for single cell
   data using mini-batch k-means.* PLOS Computational Biology, 17(1), e1008625.
2. Lun, A. T. L., Bach, K., & Marioni, J. C. (2016). *Pooling across cells to normalize single-cell RNA
   sequencing data with many zero counts.* Genome Biology, 17:75.
3. Baglama, J., & Reichel, L. (2005). *Augmented implicitly restarted Lanczos bidiagonalization methods.*
   SIAM J. Sci. Comput., 27(1):19–42.
4. Lloyd, S. P. (1982). *Least squares quantization in PCM.* IEEE Trans. Inf. Theory, 28(2):129–137.
5. Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster
   analysis.* J. Comput. Appl. Math., 20:53–65.

---
---

# 🌍 English version

> Heterogeneous (GPU + CPU multicore) reimplementation of the **mbkmeans** scRNA-seq pipeline by
> **Hicks et al. (2021)**, originally run single-thread on CPU. Beyond acceleration, the project studies
> **where the GPU actually pays off** — and what the *speedup* reveals about the data's structure.

**Course:** High Performance Computing — UFG, Institute of Informatics · **Professor:** Ricardo Augusto Pereira Franco

| Member | Student ID |
|---|---|
| Christian de Souza Ramos | 202403897 |
| Giulio Henrique Borges Basso | 202403904 |
| Igor Garbin Manzan Mazo | 202406401 |
| Wallisson Policarpo Teodoro | 202403930 |

🔗 Repository: https://github.com/IgorGMMazo/cuda-multi-k-means-scrna

## Overview

Single-cell RNA-seq (scRNA-seq) produces matrices of **millions of cells × tens of thousands of genes**.
Classic K-means becomes infeasible at this scale because it requires the whole matrix in memory.
Hicks et al. (2021) solved the memory problem with [`mbkmeans`](https://bioconductor.org/packages/mbkmeans)
(an on-disk Mini-Batch K-Means in R/Bioconductor), but their full pipeline runs **CPU single-thread**
(~104 h), dominated by IRLBA-PCA (96 h) and normalization (5.18 h).

We reimplement the pipeline on a **heterogeneous architecture**: GPU (CuPy/CUDA) for compute-bound stages,
CPU multicore (Numba/OpenMP) for memory-movement-bound stages. The five stages drop from **~102 h to
~26 min (≈ 232×)**, and — more importantly — we **measure stage by stage where the GPU pays off and why**.

## Input & pipeline

**Input:** sparse matrix of **~1.3 M cells × ~28 k genes** (~55 GB), stored as CSR in HDF5
(~2.4 B non-zeros) — the 10x Genomics *1.3M mouse brain cells* dataset. Five chained stages:
QC/filtering → initial mini-batch K-means → scran normalization → PCA (50 PCs) → final K-means with K search.

## Architecture principle: where the GPU pays off

> **The GPU only pays off when the bottleneck is computation.**

Data travels `SSD → RAM → VRAM → compute`. If time is spent **moving** data (disk I/O or PCIe transfer),
speeding up the compute barely changes total time (*data starvation*).

| Stage | Bottleneck | Nature | GPU pays off? |
|---|---|---|:---:|
| Pre-proc. (QC) | SSD → RAM (I/O) | I/O-bound | A little |
| Initial clustering | PCIe (repeated) | Transfer-bound + sparse | **No** |
| Normalization | Per-block compute | Compute-bound | Yes |
| PCA | Algorithmic | Compute-bound, dense | **Yes, a lot** |
| Final clustering | Dense compute | Compute-bound, dense | **Yes, decisive** |

## Results

| Stage | Baseline | Ours | Speedup |
|---|---|---|---|
| Pre-processing (QC) | 8 min | 4 min 40 s | ~1.71× |
| Initial clustering | 8 min 30 s | 7 min 14 s | ~1.17× |
| Normalization (scran) | 5.18 h | 8 min | 38.85× |
| PCA | 96 h | 5 min 20 s | **1080×** |
| Final clustering (K search) | ~30 s | ~70 s | ~0.43× |
| **Total (5 stages)** | **~102.3 h** | **~26.47 min** | **~231.91×** |

The largest gain (PCA) is **mostly algorithmic** — replacing iterative out-of-core SVD (IRLBA) with a
two-pass gene×gene covariance + eigendecomposition. The GPU acts as a multiplier.

**Environment:** NVIDIA RTX 5070 Laptop (Blackwell, 8 GB VRAM), 24 CPU-cores, 32 GB RAM, NVMe SSD;
`cupy-cuda13x` 14.1.0, `numba` 0.65.1, Python 3.11.

## Code map

**Main pipeline — `src/paralelizacao/`:** `pre-processamento.py` (Stage 1 — QC: GPU prefix-sums in Pass 1,
Numba/OpenMP matrix assembly in Pass 2); `clusteringAllv2.py` (Stage 2 — CuPy mini-batch K-means with
pinned buffers and sparse one-hot cluster sums); `normalizar.py` (Stage 3 — faithful scran deconvolution on
GPU: circular pools 21–101, median ratios, GPU LSQR); `pca-gpu.py` (Stage 4 — covariance PCA: `XᵀX`
accumulation, float64 `eigh`, two passes; corr 1.000000 vs scikit-learn); `clustering-finalv2.py`
(Stage 5 — full Lloyd K-means with K sweep, Gumbel-Max K-Means++ with prefix reuse, sampled silhouette,
per-K CUDA streams).

**Analysis — `src/analise/`:** `validar_inetgridade.py` (streaming end-to-end integrity validation),
`analise.py` (descriptive stats), `pca_analise.py` / `teste.py` (quick checks), `umap_vs_pca.py` /
`umap_vs_pca_multi_k.py` (PCA vs UMAP, single K=15 or K=1..20 loop), `hvg_pca_silhouette.py`
(HVG vs all-genes silhouette). *UMAP scripts are supporting exploration, outside the core scope.*

Paths in the pipeline scripts are **repo-relative** (`Path(__file__).resolve().parents[2]`) — no per-machine
editing needed; just keep the folder layout and drop the raw HDF5 into `dataset/inicial/`.

## How to run

```bash
python -m venv venv && venv\Scripts\activate     # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
# place the raw HDF5 in dataset/inicial/ then run the stages in order:
python src/paralelizacao/pre-processamento.py
python src/paralelizacao/clusteringAllv2.py
python src/paralelizacao/normalizar.py
python src/paralelizacao/pca-gpu.py
python src/paralelizacao/clustering-finalv2.py
python src/analise/validar_inetgridade.py        # optional: validate the pipeline
```

## Numerical validation & main finding

Normalization: correlation **0.99 / 0.997** vs known size factors; PCA: correlation **1.000000** vs
scikit-learn; integrity: **1,148,558 cells** consistent across all stages, **0** duplicate barcodes,
**13,897** consistent genes.

Adding the **silhouette** (absent from the paper) reveals an isolated peak at **K=2 (~0.20)** that collapses
to **≈ 0** for all other K — consistent across runs. The likely explanation: neuronal data form a
**differentiation continuum**, not discrete clusters, and K-means (Euclidean) assumes globular clusters.
This is a known limitation of the method for this data regime — not a bug. *Future work:* GMM and
graph-based methods (Louvain/Leiden over KNN).
