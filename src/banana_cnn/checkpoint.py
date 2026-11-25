"""Persistencia: salvar e carregar o autoencoder e o modelo DEC."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .modelo import AutoencoderConv, ModeloDEC


def _gravar(caminho: Path, conteudo: dict[str, Any]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    # Grava em arquivo temporario e depois renomeia: se o programa for
    # interrompido no meio, o modelo anterior nao fica corrompido.
    temporario = caminho.with_suffix(".tmp")
    torch.save(conteudo, temporario)
    temporario.replace(caminho)


def _ler(caminho: Path, dispositivo: torch.device, comando: str) -> dict[str, Any]:
    if not caminho.exists():
        raise FileNotFoundError(f"Modelo nao encontrado em {caminho}. Rode antes: python main.py {comando}")
    return torch.load(caminho, map_location=dispositivo, weights_only=True)


def salvar_autoencoder(caminho: Path, modelo: AutoencoderConv, epocas_totais: int,
                       metricas: dict[str, float]) -> None:
    _gravar(caminho, {
        "estado": modelo.state_dict(),
        "hiperparametros": modelo.hiperparametros(),
        "epocas_totais": epocas_totais,
        "metricas": metricas,
    })


def carregar_autoencoder(caminho: Path, dispositivo: torch.device) -> tuple[AutoencoderConv, dict]:
    conteudo = _ler(caminho, dispositivo, "treinar")
    hp = conteudo["hiperparametros"]
    modelo = AutoencoderConv(hp["tamanho_imagem"], hp["dimensao_latente"], tuple(hp["canais"]))
    modelo.load_state_dict(conteudo["estado"])
    return modelo.to(dispositivo), conteudo


def salvar_dec(caminho: Path, modelo: ModeloDEC, metricas: dict[str, Any]) -> None:
    _gravar(caminho, {
        "estado": modelo.state_dict(),
        "hiperparametros": modelo.autoencoder.hiperparametros(),
        "num_grupos": modelo.num_grupos,
        "alfa": modelo.agrupamento.alfa,
        "metricas": metricas,
    })


def carregar_dec(caminho: Path, dispositivo: torch.device) -> tuple[ModeloDEC, dict]:
    conteudo = _ler(caminho, dispositivo, "agrupar")
    hp = conteudo["hiperparametros"]
    autoencoder = AutoencoderConv(hp["tamanho_imagem"], hp["dimensao_latente"], tuple(hp["canais"]))
    modelo = ModeloDEC(autoencoder, int(conteudo["num_grupos"]), float(conteudo["alfa"]))
    modelo.load_state_dict(conteudo["estado"])
    return modelo.to(dispositivo), conteudo
