"""Gera docs/diagrama_arquitetura.png: as duas familias neurais do projeto.

Uso: python docs/gerar_diagrama.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

AZUL = "#4C78A8"
AZUL_CLARO = "#9ECAE9"
LARANJA = "#F58518"
VERDE = "#54A24B"
CINZA = "#555555"
VERMELHO = "#E45756"


def caixa(eixo, x, y, w, h, texto, cor, cor_texto="white", tamanho=9, negrito=False):
    eixo.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                  facecolor=cor, edgecolor="none"))
    eixo.text(x, y, texto, ha="center", va="center", color=cor_texto, fontsize=tamanho,
              fontweight="bold" if negrito else "normal")


def seta(eixo, x1, y1, x2, y2, cor=CINZA, estilo="-|>", tracejada=False, curva=0.0):
    eixo.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=estilo, mutation_scale=14, color=cor,
                                   linewidth=1.6, linestyle="--" if tracejada else "-",
                                   connectionstyle=f"arc3,rad={curva}"))


def painel_autoencoder(eixo):
    eixo.set_title("(a) Aprendizado de representação: AUTOENCODER CONVOLUCIONAL",
                   fontsize=13, fontweight="bold", loc="left")
    y = 2.2
    # Encoder: alturas proporcionais (visualmente) ao tamanho espacial.
    camadas_enc = [("Entrada\n3×96×96", 2.6, "#888888"),
                   ("Conv 4×4 s2\nBN + ReLU\n32×48×48", 2.2, AZUL),
                   ("Conv 4×4 s2\nBN + ReLU\n64×24×24", 1.8, AZUL),
                   ("Conv 4×4 s2\nBN + ReLU\n128×12×12", 1.45, AZUL),
                   ("Conv 4×4 s2\nBN + ReLU\n256×6×6", 1.15, AZUL),
                   ("Achatar\n9216\n→ Linear", 0.95, AZUL)]
    xs = [0.7, 2.0, 3.2, 4.4, 5.6, 6.75]
    for (texto, h, cor), x in zip(camadas_enc, xs):
        caixa(eixo, x, y, 1.0, h, texto, cor, tamanho=8)
    for a, b in zip(xs, xs[1:]):
        seta(eixo, a + 0.52, y, b - 0.52, y)

    caixa(eixo, 8.0, y, 1.0, 0.75, "z\n32 dims\n(sem ativ.)", LARANJA, tamanho=9, negrito=True)
    seta(eixo, 7.27, y, 7.48, y)
    eixo.text(8.0, y + 0.65, "ESPAÇO LATENTE", ha="center", fontsize=9, color=LARANJA, fontweight="bold")

    camadas_dec = [("Linear → 9216\nReLU\n256×6×6", 0.95, AZUL_CLARO),
                   ("ConvT 4×4 s2\nBN + ReLU\n128×12×12", 1.45, AZUL_CLARO),
                   ("ConvT 4×4 s2\nBN + ReLU\n64×24×24", 1.8, AZUL_CLARO),
                   ("ConvT 4×4 s2\nBN + ReLU\n32×48×48", 2.2, AZUL_CLARO),
                   ("ConvT 4×4 s2\nSigmoide\n3×96×96", 2.6, "#888888")]
    xs_dec = [9.25, 10.4, 11.6, 12.8, 14.0]
    for (texto, h, cor), x in zip(camadas_dec, xs_dec):
        caixa(eixo, x, y, 1.0, h, texto, cor, cor_texto="black" if cor == AZUL_CLARO else "white", tamanho=8)
    seta(eixo, 8.52, y, 8.73, y)
    for a, b in zip(xs_dec, xs_dec[1:]):
        seta(eixo, a + 0.52, y, b - 0.52, y)

    eixo.text(3.7, 3.85, "ENCODER  (comprime)", ha="center", fontsize=11, color=AZUL, fontweight="bold")
    eixo.text(11.6, 3.85, "DECODER  (reconstrói)", ha="center", fontsize=11, color="#2b6a9e", fontweight="bold")

    # Perda: compara saida com a propria entrada.
    seta(eixo, 0.7, y - 1.35, 14.0, y - 1.35, cor=VERMELHO, estilo="<|-|>", tracejada=True, curva=0.0)
    eixo.text(7.35, 0.4, "Perda = MSE(entrada, reconstrução)  →  o alvo é a PRÓPRIA FOTO: nenhum rótulo",
              ha="center", fontsize=10, color=VERMELHO, fontweight="bold")
    eixo.text(7.35, 0.0, "Opcional (--ruido): denoising autoencoder, entrada com ruído e alvo limpo",
              ha="center", fontsize=8.5, color=CINZA, style="italic")
    eixo.set_xlim(0, 14.7)
    eixo.set_ylim(-0.3, 4.2)
    eixo.axis("off")


def painel_dec(eixo):
    eixo.set_title("(b) Agrupamento profundo: DEEP EMBEDDED CLUSTERING (DEC / IDEC)",
                   fontsize=13, fontweight="bold", loc="left")
    y = 3.0
    caixa(eixo, 0.8, y, 1.2, 1.4, "Foto\n3×96×96", "#888888", tamanho=9)
    caixa(eixo, 2.7, y, 1.9, 1.4, "ENCODER\n(o mesmo da\netapa a)", AZUL, tamanho=9, negrito=True)
    caixa(eixo, 4.7, y, 1.0, 0.8, "z\n32 dims", LARANJA, tamanho=9, negrito=True)
    seta(eixo, 1.42, y, 1.73, y)
    seta(eixo, 3.67, y, 4.18, y)

    caixa(eixo, 7.0, y, 2.6, 1.6,
          "CAMADA DE AGRUPAMENTO\nk centroides treináveis μⱼ\n\nqᵢⱼ ∝ (1 + ‖zᵢ − μⱼ‖²)⁻¹",
          VERDE, tamanho=8.5)
    seta(eixo, 5.22, y, 5.68, y)

    caixa(eixo, 9.7, y, 1.5, 1.0, "Q\natribuição\nsuave", VERDE, tamanho=9)
    seta(eixo, 8.32, y, 8.93, y)

    caixa(eixo, 12.4, 4.15, 2.6, 1.0, "Alvo P (\"afiado\")\npᵢⱼ ∝ qᵢⱼ² / Σᵢ qᵢⱼ", "#2E7D32", tamanho=8.5)
    seta(eixo, 10.48, y + 0.3, 11.08, 4.0, curva=-0.2)

    caixa(eixo, 12.4, 2.3, 2.6, 0.9, "Perda KL( P ‖ Q )", VERMELHO, tamanho=10, negrito=True)
    seta(eixo, 10.48, y - 0.3, 11.08, 2.45, curva=0.2)
    seta(eixo, 12.4, 3.63, 12.4, 2.77, cor="#2E7D32")

    # Decoder mantido (IDEC)
    caixa(eixo, 4.7, 0.75, 2.2, 0.75, "DECODER → reconstrução", AZUL_CLARO, cor_texto="black", tamanho=8.5)
    seta(eixo, 4.7, y - 0.42, 4.7, 1.15)
    caixa(eixo, 8.4, 0.75, 3.4, 0.75, "+ γ · MSE (mantém o latente fiel às fotos)", VERMELHO, tamanho=8.5)
    seta(eixo, 5.82, 0.75, 6.68, 0.75)

    # Gradiente volta para o encoder e centroides
    seta(eixo, 11.08, 2.0, 2.7, 2.28, cor=VERMELHO, tracejada=True)
    eixo.text(8.6, 1.78, "gradiente ajusta ENCODER + CENTROIDES", ha="center", fontsize=8.5,
              color=VERMELHO, style="italic")

    eixo.text(7.35, -0.2,
              "Sinal de supervisão = a própria confiança da rede (P vem de Q). Início dos centroides: k-means uma vez (Xie et al., 2016)",
              ha="center", fontsize=9, color=CINZA)
    eixo.set_xlim(0, 14.7)
    eixo.set_ylim(-0.5, 4.8)
    eixo.axis("off")


def main():
    figura, (a, b) = plt.subplots(2, 1, figsize=(16, 10.5))
    painel_autoencoder(a)
    painel_dec(b)
    figura.text(0.99, 0.005, "Diagrama elaborado pelo grupo. Baseado em Hinton e Salakhutdinov (2006), "
                "Xie et al. (2016) e Guo et al. (2017).", ha="right", fontsize=8, color=CINZA)
    figura.tight_layout()
    destino = Path(__file__).resolve().parent / "diagrama_arquitetura.png"
    figura.savefig(destino, dpi=150, facecolor="white")
    print(destino)


if __name__ == "__main__":
    main()
