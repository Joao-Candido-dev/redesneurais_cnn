"""Persistencia do modelo: salvar e carregar o "cerebro" treinado."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .modelo import criar_modelo


@dataclass
class ModeloCarregado:
    modelo: nn.Module
    classes: list[str]
    arquitetura: str
    tamanho_imagem: int
    epocas_totais: int
    metricas: dict[str, float]


def salvar_checkpoint(
    caminho: Path,
    modelo: nn.Module,
    classes: list[str],
    arquitetura: str,
    tamanho_imagem: int,
    epocas_totais: int,
    metricas: dict[str, float],
) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    conteudo: dict[str, Any] = {
        "estado_modelo": modelo.state_dict(),
        "classes": classes,
        "arquitetura": arquitetura,
        "tamanho_imagem": tamanho_imagem,
        "epocas_totais": epocas_totais,
        "metricas": metricas,
    }
    # Grava em arquivo temporario e depois renomeia: se o programa for
    # interrompido no meio, o modelo anterior nao fica corrompido.
    temporario = caminho.with_suffix(".tmp")
    torch.save(conteudo, temporario)
    temporario.replace(caminho)


def carregar_checkpoint(caminho: Path, dispositivo: torch.device) -> ModeloCarregado:
    if not caminho.exists():
        raise FileNotFoundError(
            f"Nenhum modelo treinado em {caminho}. Rode primeiro: python main.py treinar"
        )
    conteudo = torch.load(caminho, map_location=dispositivo, weights_only=True)
    modelo = criar_modelo(
        conteudo["arquitetura"], len(conteudo["classes"]), pretreinado=False
    )
    modelo.load_state_dict(conteudo["estado_modelo"])
    modelo.to(dispositivo)
    return ModeloCarregado(
        modelo=modelo,
        classes=list(conteudo["classes"]),
        arquitetura=conteudo["arquitetura"],
        tamanho_imagem=int(conteudo["tamanho_imagem"]),
        epocas_totais=int(conteudo["epocas_totais"]),
        metricas=dict(conteudo["metricas"]),
    )
