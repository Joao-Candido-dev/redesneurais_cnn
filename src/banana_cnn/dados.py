"""Camada de dados: descoberta das fotos, divisao treino/validacao e cache.

IMPORTANTE (aprendizado nao supervisionado): a rede recebe SOMENTE os pixels.
O nome da pasta onde a foto esta (ex.: boa/, impropria/) nunca entra no
treinamento. Ele e guardado a parte apenas para, no final, medir se os
grupos que a rede descobriu sozinha coincidem com o que um humano diria.

    dados/imagens/
        boa/         -> usado so na avaliacao final
        impropria/   -> usado so na avaliacao final
        sem_rotulo/  -> fotos sem nenhuma etiqueta (tambem entram no treino)
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from .config import PASTA_SEM_ROTULO, ConfigCaminhos, ConfigTreino

logger = logging.getLogger(__name__)

EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEM_ROTULO = -1


@dataclass(frozen=True)
class Amostra:
    caminho: Path
    rotulo_referencia: int  # -1 = sem rotulo. NUNCA e usado no treino.


class DadosInsuficientesError(RuntimeError):
    """Lancada quando nao ha fotos suficientes para treinar com seguranca."""


# ---------------------------------------------------------------------------
# Leitura de imagens
# ---------------------------------------------------------------------------
def carregar_imagem(caminho: Path) -> Image.Image:
    """Abre a imagem em RGB respeitando a rotacao EXIF (fotos de celular)."""
    with Image.open(caminho) as imagem:
        imagem = ImageOps.exif_transpose(imagem)
        return imagem.convert("RGB")


def recortar_centro(imagem: Image.Image, fracao: float) -> Image.Image:
    """Mantem so a parte central da foto (fracao da largura e da altura)."""
    if not 0 < fracao <= 1:
        raise ValueError("recorte_central precisa estar entre 0 (exclusivo) e 1.")
    if fracao == 1:
        return imagem
    largura, altura = imagem.size
    margem_x, margem_y = largura * (1 - fracao) / 2, altura * (1 - fracao) / 2
    return imagem.crop((round(margem_x), round(margem_y),
                        round(largura - margem_x), round(altura - margem_y)))


def imagem_para_tensor(imagem: Image.Image, tamanho: int, recorte: float = 1.0) -> torch.Tensor:
    """PIL -> recorte central -> tensor uint8 (3, tamanho, tamanho)."""
    redimensionada = recortar_centro(imagem, recorte).resize((tamanho, tamanho), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(redimensionada, dtype=np.uint8).copy()).permute(2, 0, 1)


def carregar_tensor(caminho: Path, tamanho: int, recorte: float = 1.0) -> torch.Tensor:
    """Foto do disco -> tensor float (3, tamanho, tamanho) com valores em [0, 1].

    Use o MESMO recorte do treino (fica salvo no checkpoint).
    """
    return imagem_para_tensor(carregar_imagem(caminho), tamanho, recorte).float() / 255.0


def eh_imagem(caminho: Path) -> bool:
    return caminho.is_file() and caminho.suffix.lower() in EXTENSOES_VALIDAS


def hash_arquivo(caminho: Path) -> str:
    return hashlib.md5(caminho.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Descoberta das fotos
# ---------------------------------------------------------------------------
def descobrir_rotulos_referencia(pasta_dados: Path) -> list[str]:
    """Nomes das subpastas (exceto sem_rotulo). Usados SO na avaliacao."""
    if not pasta_dados.exists():
        return []
    return sorted(
        p.name for p in pasta_dados.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name != PASTA_SEM_ROTULO
    )


def listar_amostras(pasta_dados: Path, rotulos: list[str]) -> list[Amostra]:
    """Todas as fotos dentro de dados/imagens (inclusive subpastas)."""
    indice = {nome: i for i, nome in enumerate(rotulos)}
    amostras = []
    for arquivo in sorted(pasta_dados.rglob("*")):
        if not eh_imagem(arquivo):
            continue
        partes = arquivo.relative_to(pasta_dados).parts
        rotulo = indice.get(partes[0], SEM_ROTULO) if len(partes) > 1 else SEM_ROTULO
        amostras.append(Amostra(arquivo, rotulo))
    return amostras


def pertence_validacao(caminho: Path, fracao_validacao: float) -> bool:
    """Decide de forma DETERMINISTICA se uma foto vai para a validacao.

    O sorteio e feito pelo hash do conteudo do arquivo: ao adicionar fotos
    novas, as antigas continuam no mesmo grupo e a validacao segue honesta.
    """
    valor = int(hash_arquivo(caminho)[:8], 16) / 0xFFFFFFFF
    return valor < fracao_validacao


def dividir_treino_validacao(
    amostras: list[Amostra], fracao_validacao: float
) -> tuple[list[Amostra], list[Amostra]]:
    treino, validacao = [], []
    for amostra in amostras:
        destino = validacao if pertence_validacao(amostra.caminho, fracao_validacao) else treino
        destino.append(amostra)
    return treino, validacao


def contar_por_rotulo(amostras: list[Amostra], rotulos: list[str]) -> dict[str, int]:
    contagem = Counter(a.rotulo_referencia for a in amostras)
    resumo = {nome: contagem.get(i, 0) for i, nome in enumerate(rotulos)}
    if contagem.get(SEM_ROTULO):
        resumo[PASTA_SEM_ROTULO] = contagem[SEM_ROTULO]
    return resumo


# ---------------------------------------------------------------------------
# Cache: decodificar milhares de JPGs a cada comando e lento. As fotos ja
# recortadas e redimensionadas ficam num unico arquivo (uint8, ~27 KB por foto).
# ---------------------------------------------------------------------------
def _chave_cache(amostras: list[Amostra], tamanho: int, recorte: float) -> str:
    h = hashlib.md5(f"{tamanho}|{recorte}".encode())
    for a in amostras:
        estado = a.caminho.stat()
        h.update(f"{a.caminho}|{estado.st_size}|{estado.st_mtime_ns}".encode())
    return h.hexdigest()


def carregar_tensores(amostras: list[Amostra], tamanho: int, recorte: float,
                      arquivo_cache: Path) -> torch.Tensor:
    """Tensor uint8 (N, 3, tamanho, tamanho) com todas as fotos."""
    chave = _chave_cache(amostras, tamanho, recorte)
    if arquivo_cache.exists():
        conteudo = torch.load(arquivo_cache, weights_only=True)
        if conteudo.get("chave") == chave:
            return conteudo["imagens"]

    logger.info("Lendo e redimensionando %d fotos (so na primeira vez)...", len(amostras))
    imagens = torch.empty((len(amostras), 3, tamanho, tamanho), dtype=torch.uint8)
    for i, amostra in enumerate(amostras):
        imagens[i] = imagem_para_tensor(carregar_imagem(amostra.caminho), tamanho, recorte)
    arquivo_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"chave": chave, "imagens": imagens}, arquivo_cache)
    return imagens


# ---------------------------------------------------------------------------
# Aumento de dados geometrico (aplicado ao lote inteiro, na CPU ou na GPU)
# ---------------------------------------------------------------------------
def aumentar_geometria(x: torch.Tensor, intensidade: float = 1.0) -> torch.Tensor:
    """Gira, aproxima, desloca e espelha cada foto do lote ao acaso.

    Com intensidade 1: rotacao de ate 180 graus, zoom de ate 30% e
    deslocamento de ate 15%, mais espelhamento horizontal com chance de 50%.
    A COR NAO E ALTERADA: ela e o principal sinal de banana estragada.

    Por que ajuda no agrupamento: sem aumento, o que mais varia entre as fotos
    e a posicao, o angulo e o fundo, e o espaco latente se organiza por isso.
    Mostrando a mesma banana em varias posicoes, a rede aprende que essas
    diferencas nao importam e sobra a aparencia da casca.

    As bordas que ficam vazias apos girar sao preenchidas por reflexao, para
    nao criar cantos pretos artificiais.
    """
    if intensidade <= 0:
        return x
    n, dispositivo = x.size(0), x.device

    def sorteio(*forma: int) -> torch.Tensor:  # uniforme em [-1, 1]
        return torch.rand(*forma, device=dispositivo) * 2 - 1

    angulo = sorteio(n) * math.pi * intensidade
    # Escala da grade < 1 = zoom de aproximacao (nunca "afasta" a banana).
    escala = 1 - torch.rand(n, device=dispositivo) * 0.3 * intensidade
    espelho = torch.where(torch.rand(n, device=dispositivo) < 0.5, -1.0, 1.0)
    deslocamento = sorteio(n, 2) * 0.15 * intensidade

    cos, sen = torch.cos(angulo) * escala, torch.sin(angulo) * escala
    # Matriz afim 2x3 de cada foto: [rotacao * escala * espelho | deslocamento].
    theta = torch.stack([
        torch.stack([cos * espelho, -sen, deslocamento[:, 0]], dim=1),
        torch.stack([sen * espelho, cos, deslocamento[:, 1]], dim=1),
    ], dim=1)
    grade = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grade, padding_mode="reflection", align_corners=False)


# ---------------------------------------------------------------------------
# Dataset e DataLoaders
# ---------------------------------------------------------------------------
class DatasetImagens(Dataset):
    """Devolve (imagem, indice). Nao devolve rotulo: a rede nao ve rotulos.

    O aumento de dados e feito depois, no lote inteiro (aumentar_geometria).
    """

    def __init__(self, imagens: torch.Tensor) -> None:
        self.imagens = imagens

    def __len__(self) -> int:
        return len(self.imagens)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, int]:
        return self.imagens[indice].float() / 255.0, indice


@dataclass
class PacoteDados:
    """Tudo que o treinamento e a avaliacao precisam sobre os dados."""

    amostras: list[Amostra]
    imagens: torch.Tensor            # uint8 (N, 3, H, W), todas as fotos
    indices_treino: list[int]
    indices_validacao: list[int]
    rotulos_referencia: list[str]    # nomes das pastas (so para avaliacao)

    @property
    def rotulos_array(self) -> np.ndarray:
        return np.array([a.rotulo_referencia for a in self.amostras])

    def loader(self, indices: list[int] | None, cfg: ConfigTreino,
               embaralhar: bool = False) -> DataLoader:
        imagens = self.imagens if indices is None else self.imagens[indices]
        return DataLoader(
            DatasetImagens(imagens),
            batch_size=cfg.tamanho_lote,
            shuffle=embaralhar,
            num_workers=cfg.num_workers,
        )


def preparar_dados(caminhos: ConfigCaminhos, cfg: ConfigTreino) -> PacoteDados:
    if cfg.tamanho_imagem % 16:
        raise ValueError("tamanho_imagem precisa ser multiplo de 16.")
    rotulos = descobrir_rotulos_referencia(caminhos.dados)
    amostras = listar_amostras(caminhos.dados, rotulos)
    if len(amostras) < cfg.min_imagens:
        raise DadosInsuficientesError(
            f"Sao necessarias pelo menos {cfg.min_imagens} fotos em {caminhos.dados} "
            f"(encontradas: {len(amostras)})."
        )

    treino, validacao = dividir_treino_validacao(amostras, cfg.fracao_validacao)
    if not validacao or not treino:
        raise DadosInsuficientesError("Fotos insuficientes para separar treino e validacao.")
    em_validacao = {a.caminho for a in validacao}
    indices_treino = [i for i, a in enumerate(amostras) if a.caminho not in em_validacao]
    indices_validacao = [i for i, a in enumerate(amostras) if a.caminho in em_validacao]

    logger.info("Fotos: %d no treino, %d na validacao", len(treino), len(validacao))
    imagens = carregar_tensores(amostras, cfg.tamanho_imagem, cfg.recorte_central, caminhos.cache)
    return PacoteDados(amostras, imagens, indices_treino, indices_validacao, rotulos)
