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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
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


def imagem_para_tensor(imagem: Image.Image, tamanho: int) -> torch.Tensor:
    """PIL -> tensor uint8 (3, tamanho, tamanho)."""
    redimensionada = imagem.resize((tamanho, tamanho), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(redimensionada, dtype=np.uint8).copy()).permute(2, 0, 1)


def carregar_tensor(caminho: Path, tamanho: int) -> torch.Tensor:
    """Foto do disco -> tensor float (3, tamanho, tamanho) com valores em [0, 1]."""
    return imagem_para_tensor(carregar_imagem(caminho), tamanho).float() / 255.0


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
# redimensionadas ficam guardadas em um unico arquivo (uint8, ~27 KB por foto).
# ---------------------------------------------------------------------------
def _chave_cache(amostras: list[Amostra], tamanho: int) -> str:
    h = hashlib.md5(str(tamanho).encode())
    for a in amostras:
        estado = a.caminho.stat()
        h.update(f"{a.caminho}|{estado.st_size}|{estado.st_mtime_ns}".encode())
    return h.hexdigest()


def carregar_tensores(amostras: list[Amostra], tamanho: int, arquivo_cache: Path) -> torch.Tensor:
    """Tensor uint8 (N, 3, tamanho, tamanho) com todas as fotos."""
    chave = _chave_cache(amostras, tamanho)
    if arquivo_cache.exists():
        conteudo = torch.load(arquivo_cache, weights_only=True)
        if conteudo.get("chave") == chave:
            return conteudo["imagens"]

    logger.info("Lendo e redimensionando %d fotos (so na primeira vez)...", len(amostras))
    imagens = torch.empty((len(amostras), 3, tamanho, tamanho), dtype=torch.uint8)
    for i, amostra in enumerate(amostras):
        imagens[i] = imagem_para_tensor(carregar_imagem(amostra.caminho), tamanho)
    arquivo_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"chave": chave, "imagens": imagens}, arquivo_cache)
    return imagens


# ---------------------------------------------------------------------------
# Dataset e DataLoaders
# ---------------------------------------------------------------------------
class DatasetImagens(Dataset):
    """Devolve (imagem, indice). Nao devolve rotulo: a rede nao ve rotulos.

    O aumento de dados e so espelhamento. A cor nao e alterada de proposito:
    ela e o principal sinal visual de banana madura ou estragada.
    """

    def __init__(self, imagens: torch.Tensor, aumentar: bool = False) -> None:
        self.imagens = imagens
        self.aumentar = aumentar

    def __len__(self) -> int:
        return len(self.imagens)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, int]:
        x = self.imagens[indice].float() / 255.0
        if self.aumentar:
            if torch.rand(1).item() < 0.5:
                x = x.flip(2)
            if torch.rand(1).item() < 0.5:
                x = x.flip(1)
        return x, indice


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
               embaralhar: bool = False, aumentar: bool = False) -> DataLoader:
        imagens = self.imagens if indices is None else self.imagens[indices]
        return DataLoader(
            DatasetImagens(imagens, aumentar=aumentar),
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
    imagens = carregar_tensores(amostras, cfg.tamanho_imagem, caminhos.cache)
    return PacoteDados(amostras, imagens, indices_treino, indices_validacao, rotulos)
