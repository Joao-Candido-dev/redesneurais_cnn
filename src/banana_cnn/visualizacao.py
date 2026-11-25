"""Graficos: curvas de aprendizado, matriz de confusao e mapas de recursos."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera arquivos PNG sem precisar de janela grafica
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from .dados import carregar_imagem, transformacoes_avaliacao  # noqa: E402


def plotar_curvas(arquivo_historico: Path, destino: Path) -> Path:
    """Perda e acuracia ao longo de TODAS as rodadas de treino.

    As linhas verticais tracejadas marcam cada vez que voce rodou `treinar`
    com fotos novas: da para ver o modelo evoluindo com o tempo.
    """
    with arquivo_historico.open(encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    if not linhas:
        raise ValueError("Historico vazio.")

    epocas = [int(l["epoca_global"]) for l in linhas]
    inicios_rodada = [
        int(l["epoca_global"]) for i, l in enumerate(linhas)
        if i > 0 and l["execucao"] != linhas[i - 1]["execucao"]
    ]

    figura, (eixo_perda, eixo_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    for eixo, chave, titulo in (
        (eixo_perda, "perda", "Perda (menor e melhor)"),
        (eixo_acc, "acuracia", "Acuracia (maior e melhor)"),
    ):
        eixo.plot(epocas, [float(l[f"{chave}_treino"]) for l in linhas], label="treino")
        eixo.plot(epocas, [float(l[f"{chave}_validacao"]) for l in linhas], label="validacao")
        for inicio in inicios_rodada:
            eixo.axvline(inicio - 0.5, color="gray", linestyle="--", linewidth=0.8)
        eixo.set_title(titulo)
        eixo.set_xlabel("Epoca (acumulada)")
        eixo.grid(alpha=0.3)
        eixo.legend()

    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=120)
    plt.close(figura)
    return destino


def plotar_matriz_confusao(matriz: np.ndarray, classes: list[str], destino: Path) -> Path:
    figura, eixo = plt.subplots(figsize=(4.5, 4))
    imagem = eixo.imshow(matriz, cmap="Blues")
    eixo.set_xticks(range(len(classes)), classes)
    eixo.set_yticks(range(len(classes)), classes)
    eixo.set_xlabel("Prevista")
    eixo.set_ylabel("Real")
    limite = matriz.max() / 2 if matriz.size else 0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            cor = "white" if matriz[i, j] > limite else "black"
            eixo.text(j, i, str(matriz[i, j]), ha="center", va="center", color=cor)
    figura.colorbar(imagem)
    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=120)
    plt.close(figura)
    return destino


@torch.no_grad()
def plotar_mapas_recursos(
    modelo: nn.Module,
    caminho_imagem: Path,
    tamanho_imagem: int,
    dispositivo: torch.device,
    destino: Path,
    quantidade: int = 16,
) -> Path:
    """Mostra o "Mapa de Recursos" gerado pela primeira camada de convolucao.

    E o mesmo conceito da aula 04: cada kernel aprendido percorre a imagem
    (stride = 1) e gera um mapa que destaca um padrao diferente, como bordas
    da banana ou manchas escuras na casca.
    """
    primeira_conv = next(m for m in modelo.modules() if isinstance(m, nn.Conv2d))
    capturado: dict[str, torch.Tensor] = {}

    # Hook: "escuta" a saida da camada durante a propagacao para frente.
    gancho = primeira_conv.register_forward_hook(
        lambda _m, _e, saida: capturado.__setitem__("mapas", saida.detach().cpu())
    )
    modelo.eval()
    imagem = carregar_imagem(caminho_imagem)
    entrada = transformacoes_avaliacao(tamanho_imagem)(imagem).unsqueeze(0).to(dispositivo)
    modelo(entrada)
    gancho.remove()

    mapas = capturado["mapas"][0]
    quantidade = min(quantidade, mapas.shape[0])
    colunas = 4
    linhas = int(np.ceil((quantidade + 1) / colunas))
    figura, eixos = plt.subplots(linhas, colunas, figsize=(colunas * 2.4, linhas * 2.4))
    eixos = np.atleast_1d(eixos).ravel()

    eixos[0].imshow(imagem.resize((tamanho_imagem, tamanho_imagem)))
    eixos[0].set_title("Entrada")
    for i in range(quantidade):
        eixos[i + 1].imshow(mapas[i].numpy(), cmap="gray")
        eixos[i + 1].set_title(f"Filtro {i}")
    for eixo in eixos:
        eixo.axis("off")

    figura.suptitle("Mapas de recursos da 1a camada convolucional")
    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=110)
    plt.close(figura)
    return destino
