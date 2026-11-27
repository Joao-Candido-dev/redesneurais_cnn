"""Funcoes utilitarias compartilhadas: reprodutibilidade, dispositivo, log e historico."""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch


def configurar_log(nivel: int = logging.INFO) -> None:
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def definir_semente(semente: int) -> None:
    """Fixa as sementes aleatorias para que os experimentos sejam reprodutiveis."""
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


def escolher_dispositivo() -> torch.device:
    """Usa GPU NVIDIA (CUDA) ou Apple (MPS) quando disponivel, senao CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def registrar_historico(arquivo: Path, campos: list[str], registros: list[dict]) -> None:
    """Acrescenta as linhas de uma rodada de treino ao CSV de historico."""
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    novo = not arquivo.exists()
    with arquivo.open("a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        if novo:
            escritor.writeheader()
        escritor.writerows(registros)
