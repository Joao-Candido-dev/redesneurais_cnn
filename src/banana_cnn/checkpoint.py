"""Persistencia: salvar e carregar o autoencoder e o modelo DEC.

Alem dos pesos, cada arquivo guarda o PRE-PROCESSAMENTO usado no treino
(tamanho da imagem e recorte central). Quem carrega o modelo deve preparar as
fotos do mesmo jeito; use `aplicar_preprocessamento` para isso.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import ConfigTreino
from .modelo import AutoencoderConv, ModeloDEC

# Modelos salvos antes do recorte existir usavam a foto inteira.
RECORTE_PADRAO_ANTIGO = 1.0


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


def _criar_autoencoder(hp: dict[str, Any]) -> AutoencoderConv:
    return AutoencoderConv(hp["tamanho_imagem"], hp["dimensao_latente"], tuple(hp["canais"]))


def aplicar_preprocessamento(cfg: ConfigTreino, conteudo: dict[str, Any]) -> None:
    """Ajusta cfg para preparar as fotos igual ao treino do modelo carregado."""
    cfg.tamanho_imagem = int(conteudo["hiperparametros"]["tamanho_imagem"])
    cfg.recorte_central = float(conteudo.get("recorte_central", RECORTE_PADRAO_ANTIGO))


def salvar_autoencoder(caminho: Path, modelo: AutoencoderConv, recorte_central: float,
                       epocas_totais: int, metricas: dict[str, float]) -> None:
    _gravar(caminho, {
        "estado": modelo.state_dict(),
        "hiperparametros": modelo.hiperparametros(),
        "recorte_central": recorte_central,
        "epocas_totais": epocas_totais,
        "metricas": metricas,
    })


def carregar_autoencoder(caminho: Path, dispositivo: torch.device) -> tuple[AutoencoderConv, dict]:
    conteudo = _ler(caminho, dispositivo, "treinar")
    modelo = _criar_autoencoder(conteudo["hiperparametros"])
    modelo.load_state_dict(conteudo["estado"])
    return modelo.to(dispositivo), conteudo


def salvar_dec(caminho: Path, modelo: ModeloDEC, recorte_central: float, metricas: dict[str, Any]) -> None:
    _gravar(caminho, {
        "estado": modelo.state_dict(),
        "hiperparametros": modelo.autoencoder.hiperparametros(),
        "recorte_central": recorte_central,
        "num_grupos": modelo.num_grupos,
        "alfa": modelo.agrupamento.alfa,
        "metricas": metricas,
    })


def carregar_dec(caminho: Path, dispositivo: torch.device) -> tuple[ModeloDEC, dict]:
    conteudo = _ler(caminho, dispositivo, "agrupar")
    modelo = ModeloDEC(_criar_autoencoder(conteudo["hiperparametros"]),
                       int(conteudo["num_grupos"]), float(conteudo["alfa"]))
    modelo.load_state_dict(conteudo["estado"])
    return modelo.to(dispositivo), conteudo
