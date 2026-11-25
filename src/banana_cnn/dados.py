"""Camada de dados: descoberta das fotos, divisao treino/validacao e transformacoes.

Organizacao esperada das fotos (o nome da pasta e o rotulo da classe):

    dados/imagens/
        boa/        -> bananas boas para consumo
        impropria/  -> bananas estragadas, podres ou mofadas
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import ConfigCaminhos, ConfigTreino

logger = logging.getLogger(__name__)

EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Media e desvio padrao do ImageNet: padrao para normalizar imagens RGB
# e obrigatorio quando usamos pesos pre-treinados (MobileNet).
MEDIA_RGB = (0.485, 0.456, 0.406)
DESVIO_RGB = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Amostra:
    caminho: Path
    rotulo: int


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


def eh_imagem(caminho: Path) -> bool:
    return caminho.is_file() and caminho.suffix.lower() in EXTENSOES_VALIDAS


def hash_arquivo(caminho: Path) -> str:
    return hashlib.md5(caminho.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Descoberta das classes e divisao treino/validacao
# ---------------------------------------------------------------------------
def descobrir_classes(pasta_dados: Path) -> list[str]:
    """Cada subpasta de dados/imagens vira uma classe (ordem alfabetica)."""
    classes = sorted(
        p.name for p in pasta_dados.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if len(classes) < 2:
        raise DadosInsuficientesError(
            f"Sao necessarias pelo menos 2 pastas de classe em {pasta_dados}."
        )
    return classes


def listar_amostras(pasta_dados: Path, classes: list[str]) -> list[Amostra]:
    amostras: list[Amostra] = []
    for indice, classe in enumerate(classes):
        for arquivo in sorted((pasta_dados / classe).iterdir()):
            if eh_imagem(arquivo):
                amostras.append(Amostra(arquivo, indice))
    return amostras


def pertence_validacao(caminho: Path, fracao_validacao: float) -> bool:
    """Decide de forma DETERMINISTICA se uma foto vai para a validacao.

    O sorteio e feito pelo hash do conteudo do arquivo. Assim, quando voce
    adicionar fotos novas e treinar de novo, as fotos antigas continuam no
    mesmo grupo. Isso impede que uma foto usada no treino de ontem "vaze"
    para a validacao de hoje, o que deixaria a acuracia falsamente alta.
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


def contar_por_classe(amostras: list[Amostra], classes: list[str]) -> dict[str, int]:
    contagem = Counter(a.rotulo for a in amostras)
    return {classe: contagem.get(i, 0) for i, classe in enumerate(classes)}


def validar_quantidade(
    amostras: list[Amostra], classes: list[str], minimo_por_classe: int
) -> None:
    faltando = {
        classe: qtd
        for classe, qtd in contar_por_classe(amostras, classes).items()
        if qtd < minimo_por_classe
    }
    if faltando:
        detalhes = ", ".join(f"{c}: {q}" for c, q in faltando.items())
        raise DadosInsuficientesError(
            f"Cada classe precisa de pelo menos {minimo_por_classe} fotos. "
            f"Atualmente: {detalhes}."
        )


# ---------------------------------------------------------------------------
# Transformacoes (pre-processamento e aumento de dados)
# ---------------------------------------------------------------------------
def transformacoes_treino(tamanho: int) -> transforms.Compose:
    """Aumento de dados: gera variacoes de cada foto a cada epoca.

    Com poucas fotos a rede tende a decorar o conjunto (overfitting). Girar,
    espelhar e recortar levemente a imagem ensina a rede que a posicao da
    banana nao importa. O brilho e o contraste variam pouco e a MATIZ (hue)
    nao e alterada de proposito: a cor (amarelo, verde, marrom, preto) e o
    principal sinal de que a banana esta boa ou nao.
    """
    return transforms.Compose(
        [
            transforms.Resize((tamanho, tamanho)),
            transforms.RandomResizedCrop(tamanho, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEDIA_RGB, DESVIO_RGB),
        ]
    )


def transformacoes_avaliacao(tamanho: int) -> transforms.Compose:
    """Sem aleatoriedade: a mesma foto sempre gera a mesma entrada."""
    return transforms.Compose(
        [
            transforms.Resize((tamanho, tamanho)),
            transforms.ToTensor(),
            transforms.Normalize(MEDIA_RGB, DESVIO_RGB),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset e DataLoaders
# ---------------------------------------------------------------------------
class DatasetBananas(Dataset):
    def __init__(self, amostras: list[Amostra], transformacao: transforms.Compose) -> None:
        self.amostras = amostras
        self.transformacao = transformacao

    def __len__(self) -> int:
        return len(self.amostras)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, int]:
        amostra = self.amostras[indice]
        imagem = carregar_imagem(amostra.caminho)
        return self.transformacao(imagem), amostra.rotulo


@dataclass
class PacoteDados:
    """Tudo que o treinamento precisa sobre os dados."""

    loader_treino: DataLoader
    loader_validacao: DataLoader
    classes: list[str]
    pesos_classes: torch.Tensor
    qtd_treino: int
    qtd_validacao: int


def calcular_pesos_classes(amostras: list[Amostra], num_classes: int) -> torch.Tensor:
    """Peso maior para a classe com menos fotos (compensa desbalanceamento).

    Formula: total / (num_classes * quantidade_da_classe).
    """
    contagem = Counter(a.rotulo for a in amostras)
    total = len(amostras)
    pesos = [total / (num_classes * max(contagem.get(i, 0), 1)) for i in range(num_classes)]
    return torch.tensor(pesos, dtype=torch.float32)


def preparar_dados(caminhos: ConfigCaminhos, cfg: ConfigTreino) -> PacoteDados:
    classes = descobrir_classes(caminhos.dados)
    amostras = listar_amostras(caminhos.dados, classes)
    validar_quantidade(amostras, classes, cfg.min_imagens_por_classe)

    treino, validacao = dividir_treino_validacao(amostras, cfg.fracao_validacao)
    if not validacao:
        raise DadosInsuficientesError("Nenhuma foto caiu na validacao. Adicione mais fotos.")

    logger.info("Classes: %s", classes)
    logger.info("Treino: %s", contar_por_classe(treino, classes))
    logger.info("Validacao: %s", contar_por_classe(validacao, classes))
    for classe, qtd in contar_por_classe(validacao, classes).items():
        if qtd == 0:
            logger.warning("A classe '%s' nao tem fotos na validacao ainda.", classe)

    loader_treino = DataLoader(
        DatasetBananas(treino, transformacoes_treino(cfg.tamanho_imagem)),
        batch_size=cfg.tamanho_lote,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    loader_validacao = DataLoader(
        DatasetBananas(validacao, transformacoes_avaliacao(cfg.tamanho_imagem)),
        batch_size=cfg.tamanho_lote,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return PacoteDados(
        loader_treino=loader_treino,
        loader_validacao=loader_validacao,
        classes=classes,
        pesos_classes=calcular_pesos_classes(treino, len(classes)),
        qtd_treino=len(treino),
        qtd_validacao=len(validacao),
    )
