# -*- coding: utf-8 -*-
"""
Gera a apresentacao final (PPTX, importavel no Canva) do trabalho de
Computacao de Alto Desempenho: reimplementacao heterogenea do pipeline mbkmeans.
Identidade visual: branco + vermelho. Layout autoral.
"""
import os
from pptx import Presentation
from pptx.util import Inches as In, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

# ----------------------------------------------------------------------------- paleta
RED      = RGBColor(0xC1, 0x12, 0x1F)   # vermelho principal
DARKRED  = RGBColor(0x78, 0x00, 0x00)   # vermelho escuro
REDWASH  = RGBColor(0xFB, 0xE9, 0xEA)   # vermelho lavado (paineis)
INK      = RGBColor(0x20, 0x24, 0x29)   # texto principal
GRAY     = RGBColor(0x70, 0x76, 0x7C)   # texto secundario
LGRAY    = RGBColor(0xF3, 0xF4, 0xF6)   # cinza claro (paineis)
LINEGRAY = RGBColor(0xDD, 0xDF, 0xE3)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)

HF = "Poppins"      # titulos (Canva tem)
BF = "Inter"        # corpo (Canva tem)

L, R, C, T, M, B = PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.CENTER, MSO_ANCHOR.TOP, MSO_ANCHOR.MIDDLE, MSO_ANCHOR.BOTTOM

BASE = os.path.dirname(os.path.abspath(__file__))
IMG  = os.path.join(BASE, "metricas-etapas")

prs = Presentation()
prs.slide_width  = In(13.333)
prs.slide_height = In(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5

# ----------------------------------------------------------------------------- helpers
def slide():
    s = prs.slides.add_slide(BLANK)
    bg = s.background
    bg.fill.solid(); bg.fill.fore_color.rgb = WHITE
    return s

def rect(s, x, y, w, h, fill=None, line=None, lw=1.0, shape=MSO_SHAPE.RECTANGLE):
    sp = s.shapes.add_shape(shape, In(x), In(y), In(w), In(h))
    sp.shadow.inherit = False
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(lw)
    return sp

def txt(s, x, y, w, h, paras, anchor=T, wrap=True):
    tb = s.shapes.add_textbox(In(x), In(y), In(w), In(h))
    tf = tb.text_frame; tf.word_wrap = wrap; tf.vertical_anchor = anchor
    tf.margin_left = 0; tf.margin_right = 0; tf.margin_top = 0; tf.margin_bottom = 0
    for i, p in enumerate(paras):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.alignment = p.get("align", L)
        if "sa" in p: par.space_after = Pt(p["sa"])
        if "sb" in p: par.space_before = Pt(p["sb"])
        par.line_spacing = p.get("ls", 1.05)
        run = par.add_run(); run.text = p["text"]
        f = run.font
        f.size = Pt(p.get("size", 16)); f.bold = p.get("bold", False)
        f.italic = p.get("italic", False)
        f.name = p.get("font", BF); f.color.rgb = p.get("color", INK)
    return tb

def image_fit(s, path, x, y, maxw, maxh, frame=True):
    iw, ih = Image.open(path).size
    ratio = iw / ih
    w = maxw; h = w / ratio
    if h > maxh:
        h = maxh; w = h * ratio
    px = x + (maxw - w) / 2.0
    py = y + (maxh - h) / 2.0
    if frame:
        rect(s, px - 0.06, py - 0.06, w + 0.12, h + 0.12, fill=WHITE, line=LINEGRAY, lw=1.0)
    s.shapes.add_picture(path, In(px), In(py), In(w), In(h))
    return px, py, w, h

def leftbar(s):
    rect(s, 0, 0, 0.22, SH, fill=RED)

def pagenum(s, n):
    txt(s, SW - 1.2, SH - 0.55, 0.9, 0.35,
        [{"text": f"{n:02d} / 15", "size": 10, "color": GRAY, "font": BF, "align": R}])

def header(s, kicker, title, n, title_size=29):
    leftbar(s)
    rect(s, 0.7, 0.62, 0.34, 0.07, fill=RED)
    txt(s, 1.12, 0.5, 11.0, 0.35,
        [{"text": kicker, "size": 12.5, "bold": True, "color": RED, "font": HF}])
    txt(s, 0.68, 0.86, 12.0, 1.0,
        [{"text": title, "size": title_size, "bold": True, "color": INK, "font": HF, "ls": 1.0}])
    pagenum(s, n)

def badge(s, x, y, w, label, big, sub=None, fill=RED, tcolor=WHITE):
    h = 1.05 if sub else 0.95
    rect(s, x, y, w, h, fill=fill)
    txt(s, x + 0.22, y + 0.13, w - 0.44, 0.3,
        [{"text": label, "size": 11, "bold": True, "color": tcolor, "font": HF}])
    txt(s, x + 0.2, y + 0.36, w - 0.4, 0.5,
        [{"text": big, "size": 25, "bold": True, "color": tcolor, "font": HF}])
    if sub:
        txt(s, x + 0.22, y + 0.78, w - 0.44, 0.25,
            [{"text": sub, "size": 10.5, "color": tcolor, "font": BF}])

def bullets(s, x, y, w, items, size=15, gap=9, lead_color=INK):
    paras = []
    for it in items:
        paras.append({"text": it, "size": size, "color": lead_color, "font": BF,
                      "sa": gap, "ls": 1.12})
    txt(s, x, y, w, SH, paras)

def draw_table(s, x, y, colw, header_row, body, row_h=0.52, hh=0.56,
               header_fill=RED, aligns=None, last_bold=False, body_size=13.5,
               header_size=13.5):
    n = len(colw)
    if aligns is None:
        aligns = [L] + [C] * (n - 1)
    # header
    cx = x
    for j in range(n):
        rect(s, cx, y, colw[j], hh, fill=header_fill)
        txt(s, cx + 0.12, y, colw[j] - 0.24, hh,
            [{"text": header_row[j], "size": header_size, "bold": True,
              "color": WHITE, "font": HF, "align": aligns[j]}], anchor=M)
        cx += colw[j]
    # body
    ry = y + hh
    for r, row in enumerate(body):
        is_last = last_bold and r == len(body) - 1
        base = REDWASH if is_last else (LGRAY if r % 2 else WHITE)
        cx = x
        for j in range(n):
            rect(s, cx, ry, colw[j], row_h, fill=base)
            col = DARKRED if (is_last) else (INK if j == 0 else GRAY)
            bold = is_last or (j == n - 1)
            txt(s, cx + 0.12, ry, colw[j] - 0.24, row_h,
                [{"text": row[j], "size": body_size, "bold": bold,
                  "color": (col if not is_last else DARKRED), "font": BF,
                  "align": aligns[j]}], anchor=M)
            cx += colw[j]
        ry += row_h
    # filete vermelho sob o header
    rect(s, x, y + hh, sum(colw), 0.035, fill=DARKRED)

# =============================================================================
# SLIDE 1 - CAPA
# =============================================================================
s = slide()
# numero gigante de fundo (hook do speedup)
txt(s, 5.4, 0.2, 8.6, 7.2,
    [{"text": "232×", "size": 290, "bold": True, "color": REDWASH, "font": HF, "align": R}],
    anchor=M)
# bloco vermelho lateral esquerdo
rect(s, 0, 0, 0.45, SH, fill=RED)
rect(s, 0.45, 0, 0.12, SH, fill=DARKRED)
txt(s, 0.95, 0.85, 11.0, 0.4,
    [{"text": "COMPUTAÇÃO DE ALTO DESEMPENHO  ·  UFG", "size": 13, "bold": True,
      "color": RED, "font": HF}])
rect(s, 0.97, 1.35, 0.5, 0.08, fill=RED)
txt(s, 0.92, 1.7, 9.6, 2.6,
    [{"text": "Reimplementação heterogênea (GPU + CPU multicore) do pipeline mbkmeans",
      "size": 37, "bold": True, "color": INK, "font": HF, "ls": 1.02, "sa": 6},
     {"text": "para scRNA-seq", "size": 37, "bold": True, "color": RED, "font": HF, "ls": 1.02}])
txt(s, 0.95, 4.35, 9.4, 0.8,
    [{"text": "Onde a GPU compensa — e o que o speedup revela sobre os dados",
      "size": 17, "italic": True, "color": GRAY, "font": BF}])
rect(s, 0.97, 5.25, 5.6, 0.025, fill=LINEGRAY)
txt(s, 0.95, 5.5, 11.3, 1.0,
    [{"text": "Christian de Souza Ramos   ·   Giulio Henrique Borges Basso   ·   "
              "Igor Garbin Manzan Mazo   ·   Wallisson Policarpo Teodoro",
      "size": 13.5, "color": INK, "font": BF, "sa": 5, "ls": 1.2}])
txt(s, 0.95, 6.5, 11.0, 0.6,
    [{"text": "Instituto de Informática – Universidade Federal de Goiás   |   "
              "Paper de referência: Hicks et al. (2021), PLOS Comput. Biol.",
      "size": 11.5, "color": GRAY, "font": BF}])

# =============================================================================
# SLIDE 2 - O PROBLEMA
# =============================================================================
s = slide()
header(s, "CONTEXTO", "O problema: clusterizar milhões de células", 2)
bullets(s, 0.7, 2.0, 7.4, [
    "scRNA-seq agrupa células pela expressão gênica para identificar subpopulações "
    "celulares — um passo central é o clustering não supervisionado.",
    "Dado de entrada: matriz esparsa de ~1,3 milhão de células × ~28 mil genes, "
    "~55 GB em HDF5 (formato CSR), com ~2,4 bilhões de não-zeros.",
    "Nessa escala, o K-means clássico (que carrega tudo na memória) torna-se "
    "custoso ou inviável.",
    "Hicks et al. (2021) propõem o mbkmeans (mini-batch K-means on-disk). O "
    "pipeline completo roda em CPU single-thread em ~104 h — dominado pelo "
    "PCA-IRLBA (96 h) e pela normalização (5,18 h).",
], size=15.5, gap=12)
# cartoes de estatistica
cards = [("VOLUME DE DADOS", "55 GB", "matriz esparsa em HDF5"),
         ("ESCALA", "1,3 M", "células × 28 mil genes"),
         ("BASELINE (CPU)", "~104 h", "single-thread, paper")]
cy = 2.0
for lab, big, sub in cards:
    rect(s, 8.55, cy, 4.0, 1.45, fill=WHITE, line=LINEGRAY, lw=1.2)
    rect(s, 8.55, cy, 0.1, 1.45, fill=RED)
    txt(s, 8.85, cy + 0.18, 3.6, 0.3, [{"text": lab, "size": 10.5, "bold": True, "color": RED, "font": HF}])
    txt(s, 8.85, cy + 0.45, 3.6, 0.6, [{"text": big, "size": 32, "bold": True, "color": INK, "font": HF}])
    txt(s, 8.85, cy + 1.08, 3.6, 0.3, [{"text": sub, "size": 11, "color": GRAY, "font": BF}])
    cy += 1.65

# =============================================================================
# SLIDE 3 - O PIPELINE
# =============================================================================
s = slide()
header(s, "O PIPELINE", "Cinco etapas encadeadas", 3)
steps = [
    ("1", "Pré-processamento\n(QC / filtragem)", "Métricas de QC e descarte de células/genes"),
    ("2", "Clustering inicial\n(mini-batch K-means)", "Passo intermediário — insumo do scran"),
    ("3", "Normalização\n(scran / deconvolution)", "Preserva o sinal biológico; usa o clustering"),
    ("4", "PCA\n(50 componentes)", "Redução de dimensionalidade"),
    ("5", "Clusterização final\n(K-means pós-PCA)", "Resultado que o paper valida"),
]
n = len(steps)
bw, gap = 2.18, 0.28
total = n * bw + (n - 1) * gap
x0 = (SW - total) / 2.0
y0 = 2.5
for i, (num, title, desc) in enumerate(steps):
    x = x0 + i * (bw + gap)
    rect(s, x, y0, bw, 2.5, fill=(RED if i in (0, 2, 3, 4) else WHITE),
         line=(None if i in (0, 2, 3, 4) else RED), lw=1.6)
    onred = i in (0, 2, 3, 4)
    tc = WHITE if onred else INK
    nc = WHITE if onred else RED
    txt(s, x, y0 + 0.28, bw, 0.7, [{"text": num, "size": 40, "bold": True, "color": nc, "font": HF, "align": C}])
    txt(s, x + 0.12, y0 + 1.05, bw - 0.24, 0.9,
        [{"text": title, "size": 13.5, "bold": True, "color": tc, "font": HF, "align": C, "ls": 1.0}])
    txt(s, x + 0.12, y0 + 1.9, bw - 0.24, 0.55,
        [{"text": desc, "size": 9.8, "color": (REDWASH if onred else GRAY), "font": BF, "align": C, "ls": 1.05}])
    if i < n - 1:
        txt(s, x + bw - 0.02, y0 + 0.9, gap + 0.04, 0.5,
            [{"text": "›", "size": 30, "bold": True, "color": RED, "font": HF, "align": C}], anchor=M)
rect(s, x0, 5.55, total, 0.9, fill=LGRAY)
txt(s, x0 + 0.3, 5.55, total - 0.6, 0.9,
    [{"text": "Objetivo: reimplementar o pipeline em arquitetura heterogênea — "
              "GPU (CuPy/CUDA) para o cálculo pesado e CPU multicore (Numba/OpenMP) "
              "para a movimentação de dados.", "size": 14, "color": INK, "font": BF, "ls": 1.15}],
    anchor=M)

# =============================================================================
# SLIDE 4 - TESE CENTRAL
# =============================================================================
s = slide()
header(s, "A PERGUNTA QUE GUIA O TRABALHO", "A GPU é sempre mais rápida?", 4)
# caixa-regra vermelha
rect(s, 0.7, 2.05, 11.9, 1.15, fill=RED)
txt(s, 1.1, 2.05, 11.1, 1.15,
    [{"text": "A GPU só compensa quando o gargalo é o cálculo.",
      "size": 25, "bold": True, "color": WHITE, "font": HF}], anchor=M)
bullets(s, 0.7, 3.55, 7.3, [
    "O dado precisa viajar até a GPU: SSD → RAM → VRAM → cálculo.",
    "Se o tempo é gasto movendo dado (I/O de disco ou transferência PCIe), "
    "acelerar o cálculo quase não muda o tempo total — a GPU fica ociosa "
    "esperando o dado (data starvation).",
    "Ela só rende quando o cálculo é pesado, regular e reutiliza dados já transferidos.",
    "Nossa contribuição: medir, etapa a etapa, onde a GPU compensa e onde a CPU "
    "é a escolha certa — análise que o paper (CPU-only) não faz.",
], size=15, gap=11)
# caixa analogia
rect(s, 8.4, 3.65, 4.2, 2.85, fill=REDWASH)
rect(s, 8.4, 3.65, 0.1, 2.85, fill=RED)
txt(s, 8.72, 3.9, 3.7, 0.3, [{"text": "A ANALOGIA", "size": 11, "bold": True, "color": RED, "font": HF}])
txt(s, 8.72, 4.3, 3.65, 2.1,
    [{"text": "“Uma Ferrari (a GPU) presa no engarrafamento (I/O ou PCIe) anda na "
              "velocidade do trânsito. Trocar o motor não adianta — o que limita é a estrada.”",
      "size": 15.5, "italic": True, "color": INK, "font": BF, "ls": 1.25}])

# =============================================================================
# SLIDE 5 - ETAPA 1 (QC)
# =============================================================================
s = slide()
header(s, "ETAPA 1 · PRÉ-PROCESSAMENTO (QC)", "Híbrido: GPU no cálculo, CPU na movimentação", 5, title_size=27)
# duas colunas
def colbox(s, x, tag, title, items, tagfill):
    rect(s, x, 2.1, 5.8, 3.0, fill=WHITE, line=LINEGRAY, lw=1.2)
    rect(s, x, 2.1, 5.8, 0.62, fill=tagfill)
    txt(s, x + 0.25, 2.1, 5.3, 0.62, [{"text": title, "size": 16, "bold": True, "color": WHITE, "font": HF}], anchor=M)
    paras = [{"text": it, "size": 13.5, "color": INK, "font": BF, "sa": 8, "ls": 1.12} for it in items]
    txt(s, x + 0.28, 2.92, 5.25, 2.0, paras)
colbox(s, 0.7, "", "Pass 1  →  GPU", [
    "Métricas de QC (contagem por célula, % mitocondrial) via prefix-sums (cumsum) "
    "direto sobre os arrays CSR — sem nunca densificar a matriz.",
    "Acesso sequencial e regular: o padrão em que a GPU é eficiente. É aqui que está "
    "o ganho legítimo.",
], RED)
colbox(s, 6.83, "", "Pass 2  →  CPU multicore", [
    "Montar a matriz filtrada é mover memória, não calcular (Numba + OpenMP, "
    "paralelizado por célula).",
    "Escrita irregular (cada célula tem nº diferente de não-zeros) e embaraçosamente "
    "paralela, sem race condition: cenário ideal para CPU.",
], DARKRED)
badge(s, 0.7, 5.45, 4.0, "SPEEDUP DA ETAPA", "~1,71×", "I/O-bound · 8 min → 4 min 40 s")
rect(s, 5.05, 5.45, 7.55, 1.05, fill=LGRAY)
txt(s, 5.35, 5.45, 7.0, 1.05,
    [{"text": "Ganho modesto porque o tempo é dominado por ler ~2,4 bilhões de "
              "não-zeros do SSD — o disco não fica mais rápido só porque quem "
              "processa é a GPU.", "size": 13.5, "color": INK, "font": BF, "ls": 1.18}], anchor=M)

# =============================================================================
# SLIDE 6 - ETAPA 2 (CLUSTERING INICIAL)
# =============================================================================
s = slide()
header(s, "ETAPA 2 · CLUSTERING INICIAL", "Roda em GPU — e mede-se que ela NÃO compensa", 6, title_size=26)
txt(s, 0.7, 1.95, 11.9, 0.7,
    [{"text": "Implementado em GPU (CuPy): assignment, somas por cluster e update "
              "dos centróides na VRAM. Ainda assim, o ganho é de apenas ~1,17×. "
              "Isso não é defeito — é um resultado medido.",
      "size": 14.5, "color": INK, "font": BF, "ls": 1.18}])
factors = [
    ("Iterações repetidas", "Cada uma das ~2.000 iterações reenvia um novo batch pela "
     "PCIe. O QC transfere 1×; o clustering, milhares."),
    ("Cálculo barato demais", "A GPU calcula as distâncias em microssegundos, mas a "
     "transferência custa milissegundos → data starvation."),
    ("Esparsidade (CSR)", "A indireção (olhar o índice antes do valor) quebra o acesso "
     "coalescido e desperdiça a largura de banda da GPU."),
]
cw = 3.83
x = 0.7
for tit, desc in factors:
    rect(s, x, 2.85, cw, 1.95, fill=WHITE, line=LINEGRAY, lw=1.2)
    rect(s, x, 2.85, cw, 0.1, fill=RED)
    txt(s, x + 0.22, 3.05, cw - 0.44, 0.45, [{"text": tit, "size": 14.5, "bold": True, "color": RED, "font": HF}])
    txt(s, x + 0.22, 3.55, cw - 0.44, 1.15, [{"text": desc, "size": 12.5, "color": INK, "font": BF, "ls": 1.16}])
    x += cw + 0.2
rect(s, 0.7, 5.1, 8.05, 1.4, fill=RED)
txt(s, 1.05, 5.1, 7.4, 1.4,
    [{"text": "Mostrar onde a GPU NÃO ajuda é parte da contribuição. O gargalo "
              "aqui é a transferência PCIe repetida de dados esparsos com acesso "
              "irregular — não o cálculo.", "size": 15.5, "bold": True, "color": WHITE, "font": HF, "ls": 1.18}],
    anchor=M)
badge(s, 8.95, 5.1, 3.65, "SPEEDUP DA ETAPA", "~1,17×", "transfer-bound + esparso", fill=DARKRED)

# =============================================================================
# SLIDE 7 - ETAPA 3 (NORMALIZACAO scran)
# =============================================================================
s = slide()
header(s, "ETAPA 3 · NORMALIZAÇÃO (scran)", "Reimplementação fiel do scran na GPU", 7, title_size=27)
bullets(s, 0.7, 2.0, 7.3, [
    "O paper não descreve a normalização — terceiriza ao scran (Lun et al., 2016). "
    "Reimplementamos o algoritmo de deconvolution passo a passo: pré-escala por "
    "library size, pools circulares (21–101), mediana de razões, solve linear (LSQR) "
    "e reescala entre clusters.",
    "Por que a GPU compensa (≠ clustering): o cálculo por bloco é pesado e regular "
    "(divisões, medianas em janelas, solve) sobre um bloco denso e limitado "
    "(≤ ~170 MB) que cabe na VRAM — sem as idas-e-vindas da PCIe.",
    "Por que scran e não library-size: a library-size assume mesma composição de RNA "
    "e apaga sinal biológico real; o scran compara células parecidas (pooling) — por "
    "isso depende do clustering inicial.",
], size=14.5, gap=12)
badge(s, 8.4, 2.05, 4.2, "SPEEDUP DA ETAPA", "38,85×", "compute-bound · 5,18 h → 8 min")
rect(s, 8.4, 3.3, 4.2, 3.2, fill=REDWASH)
rect(s, 8.4, 3.3, 0.1, 3.2, fill=RED)
txt(s, 8.72, 3.55, 3.7, 0.3, [{"text": "VALIDAÇÃO", "size": 11, "bold": True, "color": RED, "font": HF}])
txt(s, 8.72, 3.95, 3.65, 2.4,
    [{"text": "Conferido em NumPy contra fatores de tamanho conhecidos:",
      "size": 13, "color": INK, "font": BF, "sa": 8, "ls": 1.15},
     {"text": "corr 0,99 (1 cluster) · 0,997 (multi-cluster)",
      "size": 14, "bold": True, "color": DARKRED, "font": HF, "sa": 8, "ls": 1.1},
     {"text": "Erro ~2,6–6% — contra ~15–23% da normalização simples por library size.",
      "size": 13, "color": INK, "font": BF, "ls": 1.15}])

# =============================================================================
# SLIDE 8 - ETAPA 4 (PCA)
# =============================================================================
s = slide()
header(s, "ETAPA 4 · PCA (50 COMPONENTES)", "Troca de algoritmo: IRLBA → covariância", 8, title_size=27)
# coluna IRLBA
rect(s, 0.7, 2.05, 5.8, 2.35, fill=WHITE, line=LINEGRAY, lw=1.2)
rect(s, 0.7, 2.05, 5.8, 0.58, fill=GRAY)
txt(s, 0.95, 2.05, 5.3, 0.58, [{"text": "IRLBA  (paper)", "size": 15, "bold": True, "color": WHITE, "font": HF}], anchor=M)
txt(s, 0.98, 2.78, 5.25, 1.55,
    [{"text": "SVD truncado e iterativo que relê a matriz do disco a cada iteração. "
              "Genérico, mas o preço é I/O: dezenas de passadas pelo disco = 96 h. "
              "O gargalo é leitura repetida, não cálculo.",
      "size": 13.5, "color": INK, "font": BF, "ls": 1.18}])
# coluna nosso metodo
rect(s, 6.83, 2.05, 5.77, 2.35, fill=WHITE, line=RED, lw=1.6)
rect(s, 6.83, 2.05, 5.77, 0.58, fill=RED)
txt(s, 7.08, 2.05, 5.3, 0.58, [{"text": "Nosso método  (covariância)", "size": 15, "bold": True, "color": WHITE, "font": HF}], anchor=M)
txt(s, 7.11, 2.78, 5.25, 1.55,
    [{"text": "Como n (1,1 M) ≫ p (13,9 k), a covariância gene×gene (~1,5 GB) cabe na "
              "VRAM. Em 2 passadas: S = XᵀX + média → eigh(M) dá os top-50 loadings → "
              "projeta os scores. Exato: corr 1,000000 vs scikit-learn.",
      "size": 13.5, "color": INK, "font": BF, "ls": 1.18}])
# caixa insight
rect(s, 0.7, 4.65, 8.05, 1.85, fill=RED)
txt(s, 1.05, 4.85, 7.4, 0.35, [{"text": "INSIGHT CENTRAL", "size": 11.5, "bold": True, "color": WHITE, "font": HF}])
txt(s, 1.05, 5.25, 7.4, 1.15,
    [{"text": "O maior ganho é ALGORÍTMICO, não da GPU. Reconhecer que n ≫ p e trocar "
              "o SVD iterativo por covariância em 2 passadas esmagaria o IRLBA mesmo em "
              "CPU. A GPU atua como multiplicador (acelera o eigh e os matmuls).",
      "size": 15, "bold": True, "color": WHITE, "font": HF, "ls": 1.18}])
badge(s, 8.95, 4.65, 3.65, "SPEEDUP DA ETAPA", "1080×", "96 h → 5 min 20 s", fill=DARKRED)

# =============================================================================
# SLIDE 9 - ETAPA 5 (CLUSTERING FINAL)
# =============================================================================
s = slide()
header(s, "ETAPA 5 · CLUSTERIZAÇÃO FINAL", "Lloyd completo, pós-PCA, com busca de K", 9, title_size=27)
bullets(s, 0.7, 2.0, 7.4, [
    "Após o PCA, a matriz é 1,1 M × 50 (densa, ~220 MB) e cabe inteira na VRAM. "
    "Sem motivo de memória para aproximar → usamos Lloyd completo, em que cada "
    "iteração é um único GEMM (n×d)·(d×k): o padrão em que a GPU é imbatível.",
    "O próprio paper mostra que mini-batch ≈ K-means exato (batch ≥ 500), então a "
    "troca remove a aproximação sem alterar as conclusões biológicas.",
    "Além do paper: o paper fixa K=15. Fazemos busca sistemática sobre K, com "
    "n_init=3 reinícios, selecionando por WCSS/n (cotovelo) + silhouette — "
    "operacionalizando uma sugestão que o paper faz mas nunca executa.",
], size=14.5, gap=13)
rect(s, 8.55, 2.05, 4.05, 4.45, fill=REDWASH)
rect(s, 8.55, 2.05, 0.1, 4.45, fill=RED)
txt(s, 8.87, 2.3, 3.6, 0.3, [{"text": "INICIAL  vs  FINAL", "size": 11, "bold": True, "color": RED, "font": HF}])
comp = [("Clustering inicial", "1,1 M × 13,9 k, esparso — não cabe na VRAM. Mini-batch obrigatório. GPU NÃO compensa.", GRAY),
        ("Clusterização final", "1,1 M × 50, densa (220 MB) — cabe na VRAM. Lloyd completo. GPU decisiva.", DARKRED)]
yy = 2.75
for tit, desc, c in comp:
    txt(s, 8.87, yy, 3.55, 0.35, [{"text": tit, "size": 14, "bold": True, "color": c, "font": HF}])
    txt(s, 8.87, yy + 0.4, 3.55, 1.3, [{"text": desc, "size": 12.5, "color": INK, "font": BF, "ls": 1.16}])
    yy += 1.85

# =============================================================================
# SLIDE 10 - RESULTADOS (TABELA SPEEDUP)
# =============================================================================
s = slide()
header(s, "RESULTADOS · DESEMPENHO", "~102 h → ~26 min: speedup global ~232×", 10, title_size=27)
draw_table(s, 0.7, 2.05,
           [4.7, 2.45, 2.45, 2.3],
           ["Etapa", "Baseline", "Proposta", "Speedup"],
           [["Pré-processamento (QC)", "8 min", "4 min 40 s", "~1,71×"],
            ["Clustering inicial", "8 min 30 s", "7 min 14 s", "~1,17×"],
            ["Normalização (scran)", "5,18 h", "8 min", "38,85×"],
            ["PCA", "96 h", "5 min 20 s", "1080×"],
            ["Clusterização final (busca de K)", "~30 s", "~70 s", "~0,43×"],
            ["Total (5 etapas)", "~102,3 h", "~26,47 min", "~231,91×"]],
           row_h=0.5, hh=0.55, last_bold=True, body_size=14, header_size=14)
txt(s, 0.7, 6.05, 11.9, 1.2,
    [{"text": "Ambiente: NVIDIA RTX 5070 Laptop (Blackwell, 8 GB VRAM), 24 CPU-cores, "
              "32 GB RAM, SSD NVMe. O baseline são os tempos reportados por Hicks et al. "
              "(CPU single-thread); a comparação é entre pipelines, não entre hardwares "
              "equivalentes.", "size": 12, "italic": True, "color": GRAY, "font": BF, "ls": 1.2}])

# =============================================================================
# SLIDE 11 - ANALISE DE GARGALOS
# =============================================================================
s = slide()
header(s, "RESULTADOS · ANÁLISE DE GARGALOS", "O número agregado não é o ponto — o porquê é", 11, title_size=26)
draw_table(s, 0.7, 2.05,
           [3.4, 3.4, 3.2, 1.9],
           ["Etapa", "Gargalo", "Natureza", "GPU compensa?"],
           [["Pré-proc. (QC)", "SSD → RAM (I/O)", "I/O-bound", "Pouco"],
            ["Clustering inicial", "PCIe (repetido)", "Transfer-bound + esparso", "Não"],
            ["Normalização", "Cálculo por bloco", "Compute-bound", "Sim"],
            ["PCA", "Algorítmico", "Compute-bound, denso", "Sim, muito"],
            ["Clusterização final", "Cálculo denso", "Compute-bound, denso", "Sim, decisivo"]],
           row_h=0.6, hh=0.55, aligns=[L, L, L, C], body_size=13.5, header_size=13.5)
rect(s, 0.7, 6.0, 11.9, 1.05, fill=RED)
txt(s, 1.05, 6.0, 11.2, 1.05,
    [{"text": "A mesma GPU dá ~1,7× num caso e ordens de magnitude no outro. Distinguir "
              "I/O-bound × transfer-bound × compute-bound — e alocar GPU ou CPU de acordo — "
              "é o cerne do trabalho.", "size": 14.5, "bold": True, "color": WHITE, "font": HF, "ls": 1.18}],
    anchor=M)

# =============================================================================
# SLIDE 12 - VALIDACAO NUMERICA
# =============================================================================
s = slide()
header(s, "RESULTADOS · VALIDAÇÃO NUMÉRICA", "A aceleração não comprometeu a corretude", 12, title_size=27)
val = [("NORMALIZAÇÃO (scran)", "corr 0,99 / 0,997",
        "1 cluster e multi-cluster vs. fatores de tamanho conhecidos. Erro ~2,6–6% "
        "(contra ~15–23% da library-size simples)."),
       ("PCA", "corr 1,000000",
        "Validado contra o PCA do scikit-learn nos 50 componentes — reformulação "
        "matematicamente exata."),
       ("INTEGRIDADE PONTA A PONTA", "1.148.558 células",
        "Validador em streaming (bruto → filtrado → clusterizado → normalizado → PCA): "
        "0 barcodes duplicados, 13.897 genes consistentes.")]
x = 0.7
cw = 3.83
for lab, big, desc in val:
    rect(s, x, 2.25, cw, 3.6, fill=WHITE, line=LINEGRAY, lw=1.2)
    rect(s, x, 2.25, cw, 0.85, fill=RED)
    txt(s, x + 0.22, 2.25, cw - 0.44, 0.85, [{"text": lab, "size": 12.5, "bold": True, "color": WHITE, "font": HF, "ls": 1.0}], anchor=M)
    txt(s, x + 0.22, 3.3, cw - 0.44, 0.7, [{"text": big, "size": 24, "bold": True, "color": DARKRED, "font": HF}])
    txt(s, x + 0.22, 4.15, cw - 0.44, 1.55, [{"text": desc, "size": 13, "color": INK, "font": BF, "ls": 1.2}])
    x += cw + 0.2
txt(s, 0.7, 6.1, 11.9, 0.6,
    [{"text": "A consistência ponta a ponta reforça que o silhouette baixo (próximo slide) "
              "é estrutura real do dado — não desalinhamento entre etapas do pipeline.",
      "size": 13, "italic": True, "color": GRAY, "font": BF, "ls": 1.18}])

# =============================================================================
# SLIDE 13 - WCSS vs SILHOUETTE
# =============================================================================
s = slide()
header(s, "RESULTADOS · QUALIDADE DO AGRUPAMENTO", "WCSS vs. Silhouette — métrica ausente no paper", 13, title_size=24)
bullets(s, 0.7, 2.15, 4.7, [
    "O paper avalia o agrupamento só por WCSS — que decresce monotonicamente com K "
    "e não mede a separação dos clusters.",
    "Adicionamos o silhouette: pico isolado em K=2 (~0,20) e colapso para ≈ 0 nos "
    "demais valores de K.",
    "O padrão se manteve entre rodadas independentes → é sinal real, não ruído.",
    "Convenção: silhouette < 0,25 indica estrutura de cluster fraca ou ausente.",
], size=14.5, gap=14)
image_fit(s, os.path.join(IMG, "grafico_metricas_todos_genes.png"), 5.6, 2.0, 7.3, 4.6)
txt(s, 5.6, 6.75, 7.3, 0.4,
    [{"text": "Análise multi-K: WCSS/n e silhouette para K de 1 a 20.",
      "size": 11, "italic": True, "color": GRAY, "font": BF, "align": C}])

# =============================================================================
# SLIDE 14 - NAO-CLUSTERABILIDADE
# =============================================================================
s = slide()
header(s, "RESULTADOS · O QUE OS DADOS REVELAM", "Aumentar K não cria fronteiras", 14, title_size=27)
image_fit(s, os.path.join(IMG, "clusters_2d_k3.png"), 0.7, 1.95, 5.7, 3.0)
image_fit(s, os.path.join(IMG, "clusters_2d_k8.png"), 6.55, 1.95, 5.7, 3.0)
txt(s, 0.7, 5.05, 11.6, 0.35,
    [{"text": "Projeção dos clusters no espaço PCA (PC1 × PC2): K=3 (esq.) e K=8 (dir.) — "
              "uma nuvem contínua, sem fronteiras nítidas.",
      "size": 11, "italic": True, "color": GRAY, "font": BF, "align": C}])
rect(s, 0.7, 5.55, 11.9, 1.4, fill=REDWASH)
rect(s, 0.7, 5.55, 0.1, 1.4, fill=RED)
txt(s, 1.05, 5.55, 11.3, 1.4,
    [{"text": "Hipótese: neurônios formam um contínuo de diferenciação, não clusters "
              "discretos. O K-means (distância euclidiana) assume clusters globulares — num "
              "gradiente, qualquer K dá silhouette baixo. Não invalida a implementação; é "
              "limitação do método frente ao dado. Trabalho futuro: GMM e métodos de grafo "
              "(Louvain/Leiden sobre KNN).",
      "size": 13.5, "color": INK, "font": BF, "ls": 1.2}], anchor=M)

# =============================================================================
# SLIDE 15 - CONCLUSAO
# =============================================================================
s = slide()
rect(s, 0, 0, SW, SH, fill=RED)
rect(s, 0, 0, SW, 0.18, fill=DARKRED)
txt(s, 0.9, 0.7, 11.0, 0.4, [{"text": "CONCLUSÃO", "size": 14, "bold": True, "color": REDWASH, "font": HF}])
txt(s, 0.88, 1.05, 11.5, 0.8, [{"text": "O que levamos deste trabalho", "size": 30, "bold": True, "color": WHITE, "font": HF}])
takeaways = [
    ("232×", "Speedup global reimplementando as 5 etapas em arquitetura heterogênea "
     "(GPU + CPU multicore): ~102 h → ~26 min."),
    ("Onde", "A GPU só compensa quando o gargalo é cálculo pesado e regular que cabe na "
     "VRAM — documentamos até um caso onde ela não compensa (clustering inicial)."),
    ("Algoritmo", "O maior ganho (PCA, 1080×) é majoritariamente algorítmico; a GPU "
     "atua como multiplicador, não como fonte primária do ganho."),
    ("Silhouette", "A métrica ausente no paper revela que os dados não são bem separados "
     "por K-means — estrutura mais contínua que discreta."),
]
yy = 2.15
for tag, desc in takeaways:
    rect(s, 0.9, yy, 1.95, 0.92, fill=WHITE)
    txt(s, 0.9, yy, 1.95, 0.92, [{"text": tag, "size": 19, "bold": True, "color": RED, "font": HF, "align": C}], anchor=M)
    txt(s, 3.05, yy, 9.4, 0.92, [{"text": desc, "size": 14.5, "color": WHITE, "font": BF, "ls": 1.16}], anchor=M)
    yy += 1.05
rect(s, 0.9, 6.55, 11.5, 0.025, fill=REDWASH)
txt(s, 0.9, 6.7, 11.5, 0.5,
    [{"text": "Código: github.com/IgorGMMazo/cuda-multi-k-means-scrna      |      Obrigado!",
      "size": 13.5, "bold": True, "color": WHITE, "font": HF}])

# -----------------------------------------------------------------------------
out = os.path.join(os.path.expanduser("~"), "Downloads", "Apresentacao_CAD_mbkmeans.pptx")
prs.save(out)
print("OK ->", out)
