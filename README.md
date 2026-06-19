# 🧬 CUDA Multi-K Means for Single-Cell Genomics

> Implementação paralela em GPU (CUDA C) do algoritmo Multi-K Means para clusterização de dados de expressão gênica de célula única (scRNA-seq).

---

## Motivação

O sequenciamento de RNA de célula única (scRNA-seq) gera datasets com milhares a **milhões de células**, cada uma com milhares de genes mensurados. Identificar subpopulações celulares nesses dados exige algoritmos de clusterização eficientes — e o K-means, um dos métodos mais populares, se torna inviável em escala: para 1,3 milhão de células e 11.720 genes, a execução sequencial em CPU exige mais de **52 GB de RAM** e pode levar **dezenas de minutos**.

## Baseline

Este projeto parte do trabalho de **Hicks et al. (2021)**, que propôs o pacote [`mbkmeans`](https://bioconductor.org/packages/mbkmeans) — uma implementação do Mini-Batch K-Means para dados de célula única em R/Bioconductor. Apesar de reduzir o consumo de memória ao processar mini-batches, a solução é **exclusivamente baseada em CPU**, sem explorar o paralelismo massivo de GPUs modernas.

> Hicks, S. C., Liu, R., Ni, Y., Purdom, E., & Risso, D. (2021). mbkmeans: Fast clustering for single cell data using mini-batch k-means. *PLOS Computational Biology*, 17(1), e1008625. https://doi.org/10.1371/journal.pcbi.1008625

## Nossa Proposta

Implementar o algoritmo **Multi-K Means em CUDA C**, calculando simultaneamente múltiplos valores de K em GPU — maximizando o paralelismo e reduzindo drasticamente o tempo de análise exploratória de clustering em datasets genômicos de larga escala.

---

## Equipe

Disciplina: **Computação de Alto Desempenho** — UFG, Instituto de Informática  
Prof. Ricardo Augusto Pereira Franco

| Nome | Matrícula |
|---|---|
| Christian de Souza Ramos |
| Giulio Henrique Borges Basso | 
| Igor Garbin Manzan Mazo | 
| Wallisson Policarpo Teodoro |

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 🧬 CUDA Multi-K Means for Single-Cell Genomics

> Parallel GPU implementation (CUDA C) of the Multi-K Means algorithm for clustering single-cell RNA sequencing (scRNA-seq) data.

---

## Motivation

Single-cell RNA sequencing (scRNA-seq) produces datasets ranging from thousands to **millions of cells**, each measured across thousands of genes. Identifying cell subpopulations in this data requires efficient clustering algorithms — and K-means, one of the most widely used methods, becomes computationally infeasible at scale: for 1.3 million cells and 11,720 genes, a sequential CPU execution demands more than **52 GB of RAM** and can take **tens of minutes**.

## Baseline

This project builds upon the work of **Hicks et al. (2021)**, which proposed the [`mbkmeans`](https://bioconductor.org/packages/mbkmeans) package — a Mini-Batch K-Means implementation for single-cell data in R/Bioconductor. Although it reduces memory consumption by processing mini-batches, the solution is **CPU-only**, without leveraging the massive parallelism offered by modern GPUs.

> Hicks, S. C., Liu, R., Ni, Y., Purdom, E., & Risso, D. (2021). mbkmeans: Fast clustering for single cell data using mini-batch k-means. *PLOS Computational Biology*, 17(1), e1008625. https://doi.org/10.1371/journal.pcbi.1008625

## Our Proposal

Implement the **Multi-K Means algorithm in CUDA C**, computing multiple values of K simultaneously on GPU — maximizing parallelism and drastically reducing the runtime of exploratory clustering analysis on large-scale genomic datasets.

---

## Team

Course: **High Performance Computing** — UFG, Institute of Informatics  
Prof. Ricardo Augusto Pereira Franco

| Name | Student ID |
|---|---|
| Christian de Souza Ramos | 
| Giulio Henrique Borges Basso | 
| Igor Garbin Manzan Mazo | 
| Wallisson Policarpo Teodoro | 