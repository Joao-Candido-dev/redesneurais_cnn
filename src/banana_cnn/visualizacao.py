"""Graficos para o relatorio e o video."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera arquivos PNG sem precisar de janela grafica
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from .modelo import AutoencoderConv  # noqa: E402


def _salvar(figura: plt.Figure, destino: Path, dpi: int = 120) -> Path:
    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=dpi)
    plt.close(figura)
    return destino


def _mostrar(eixo: plt.Axes, imagem: torch.Tensor, titulo: str = "") -> None:
    eixo.imshow(imagem.permute(1, 2, 0).clamp(0, 1).numpy())
    eixo.set_title(titulo, fontsize=8)
    eixo.axis("off")


def _ler_csv(arquivo: Path) -> list[dict]:
    if not arquivo.exists():
        raise FileNotFoundError(f"Historico nao encontrado: {arquivo}")
    with arquivo.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plotar_curvas_autoencoder(arquivo: Path, destino: Path) -> Path:
    """Perda de reconstrucao ao longo de todas as rodadas de treino."""
    linhas = _ler_csv(arquivo)
    epocas = [int(l["epoca_global"]) for l in linhas]
    figura, eixo = plt.subplots(figsize=(7, 4))
    eixo.plot(epocas, [float(l["perda_treino"]) for l in linhas], label="treino")
    eixo.plot(epocas, [float(l["perda_validacao"]) for l in linhas], label="validacao")
    for i in range(1, len(linhas)):
        if linhas[i]["execucao"] != linhas[i - 1]["execucao"]:
            eixo.axvline(epocas[i] - 0.5, color="gray", linestyle="--", linewidth=0.8)
    eixo.set_title("Autoencoder: erro de reconstrucao (MSE, menor e melhor)")
    eixo.set_xlabel("Epoca (acumulada)")
    eixo.grid(alpha=0.3)
    eixo.legend()
    return _salvar(figura, destino)


def plotar_curvas_dec(arquivo: Path, destino: Path) -> Path:
    """Ultima rodada do DEC: perda KL, fotos que trocaram de grupo e acuracia de referencia."""
    linhas = _ler_csv(arquivo)
    ultima = linhas[-1]["execucao"]
    linhas = [l for l in linhas if l["execucao"] == ultima]
    epocas = [int(l["epoca"]) for l in linhas]
    figura, eixos = plt.subplots(1, 3, figsize=(13, 3.8))
    series = [("perda_kl", "Perda KL(P || Q)"),
              ("fracao_trocou_grupo", "Fracao de fotos que trocaram de grupo"),
              ("acuracia_referencia", "Acuracia de referencia (so medida)")]
    for eixo, (chave, titulo) in zip(eixos, series):
        eixo.plot(epocas, [float(l[chave]) for l in linhas], marker="o", markersize=3)
        eixo.set_title(titulo, fontsize=10)
        eixo.set_xlabel("Epoca do DEC")
        eixo.grid(alpha=0.3)
    return _salvar(figura, destino)


@torch.no_grad()
def plotar_reconstrucoes(autoencoder: AutoencoderConv, imagens: torch.Tensor,
                         dispositivo: torch.device, destino: Path) -> Path:
    """Linha de cima: fotos originais. Linha de baixo: o que a rede redesenhou."""
    autoencoder.eval()
    reconstruidas = autoencoder(imagens.to(dispositivo))[0].cpu()
    n = len(imagens)
    figura, eixos = plt.subplots(2, n, figsize=(1.6 * n, 3.6))
    eixos = np.atleast_2d(eixos).reshape(2, n)
    for i in range(n):
        _mostrar(eixos[0, i], imagens[i], "original" if i == 0 else "")
        _mostrar(eixos[1, i], reconstruidas[i], "reconstruida" if i == 0 else "")
    figura.suptitle(f"Autoencoder: entrada -> espaco latente de {autoencoder.dimensao_latente} numeros -> saida")
    return _salvar(figura, destino)


def plotar_espaco_latente(z: np.ndarray, grupos: np.ndarray, rotulos_ref: np.ndarray,
                          nomes_rotulos: list[str], destino: Path, centroides: np.ndarray | None = None) -> Path:
    """Projeta o espaco latente em 2D (PCA, apenas para DESENHAR o grafico).

    Esquerda: cores = grupos descobertos pela rede. Direita: cores = pastas
    reais. Se as duas figuras se parecem, a rede achou a divisao sozinha.
    """
    projetor = PCA(n_components=2).fit(z)
    pontos = projetor.transform(z)
    figura, (esq, dir_) = plt.subplots(1, 2, figsize=(12, 5))
    for g in np.unique(grupos):
        m = grupos == g
        esq.scatter(pontos[m, 0], pontos[m, 1], s=10, alpha=0.6, label=f"grupo {g}")
    if centroides is not None:
        c = projetor.transform(centroides)
        esq.scatter(c[:, 0], c[:, 1], marker="X", s=200, c="black", label="centroides")
    esq.set_title("Grupos descobertos (sem rotulos)")
    esq.legend()
    for r in np.unique(rotulos_ref):
        m = rotulos_ref == r
        nome = nomes_rotulos[r] if r >= 0 else "sem_rotulo"
        dir_.scatter(pontos[m, 0], pontos[m, 1], s=10, alpha=0.6, label=nome)
    dir_.set_title("Pastas reais (usadas so para conferir)")
    dir_.legend()
    for eixo in (esq, dir_):
        eixo.set_xlabel("direcao 1")
        eixo.set_ylabel("direcao 2")
        eixo.grid(alpha=0.3)
    figura.suptitle("Espaco latente do autoencoder projetado em 2D")
    return _salvar(figura, destino)


def plotar_fotos_por_grupo(imagens: torch.Tensor, q: np.ndarray, destino: Path, por_grupo: int = 8) -> Path:
    """As fotos mais "tipicas" (maior q) de cada grupo."""
    grupos = q.argmax(axis=1)
    k = q.shape[1]
    figura, eixos = plt.subplots(k, por_grupo, figsize=(1.6 * por_grupo, 1.8 * k))
    eixos = np.atleast_2d(eixos).reshape(k, por_grupo)
    for g in range(k):
        membros = np.where(grupos == g)[0]
        ordem = membros[np.argsort(-q[membros, g])][:por_grupo]
        for j in range(por_grupo):
            eixo = eixos[g, j]
            if j < len(ordem):
                _mostrar(eixo, imagens[ordem[j]].float() / 255.0, f"q={q[ordem[j], g]:.2f}")
            else:
                eixo.axis("off")
        eixos[g, 0].text(-0.15, 0.5, f"grupo {g}\n({len(membros)})", transform=eixos[g, 0].transAxes,
                         ha="right", va="center", fontsize=10)
    figura.suptitle("Fotos mais tipicas de cada grupo descoberto")
    return _salvar(figura, destino)


def plotar_tabela_grupos(matriz: np.ndarray, nomes_rotulos: list[str], destino: Path) -> Path:
    figura, eixo = plt.subplots(figsize=(1.6 + 1.3 * matriz.shape[1], 1.2 + 0.9 * matriz.shape[0]))
    imagem = eixo.imshow(matriz, cmap="Blues")
    eixo.set_xticks(range(matriz.shape[1]), [f"grupo {g}" for g in range(matriz.shape[1])])
    eixo.set_yticks(range(len(nomes_rotulos)), nomes_rotulos)
    eixo.set_xlabel("Grupo descoberto pela rede")
    eixo.set_ylabel("Pasta real")
    limite = matriz.max() / 2 if matriz.size else 0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            eixo.text(j, i, str(matriz[i, j]), ha="center", va="center",
                      color="white" if matriz[i, j] > limite else "black")
    figura.colorbar(imagem)
    return _salvar(figura, destino)


@torch.no_grad()
def plotar_interpolacao(autoencoder: AutoencoderConv, imagem_a: torch.Tensor, imagem_b: torch.Tensor,
                        dispositivo: torch.device, destino: Path, passos: int = 8) -> Path:
    """Caminha em linha reta no espaco latente da foto A ate a foto B.

    O decoder desenha cada ponto intermediario. Mostra que o espaco latente
    e "continuo": pontos vizinhos viram bananas parecidas.
    """
    autoencoder.eval()
    z_a = autoencoder.codificar(imagem_a.unsqueeze(0).to(dispositivo))
    z_b = autoencoder.codificar(imagem_b.unsqueeze(0).to(dispositivo))
    pesos = torch.linspace(0, 1, passos, device=dispositivo).unsqueeze(1)
    geradas = autoencoder.decodificar((1 - pesos) * z_a + pesos * z_b).cpu()

    figura, eixos = plt.subplots(1, passos + 2, figsize=(1.6 * (passos + 2), 2.2))
    _mostrar(eixos[0], imagem_a, "foto A")
    for i in range(passos):
        _mostrar(eixos[i + 1], geradas[i], f"{pesos[i].item():.2f}")
    _mostrar(eixos[-1], imagem_b, "foto B")
    figura.suptitle("Interpolacao no espaco latente (A -> B)")
    return _salvar(figura, destino)
