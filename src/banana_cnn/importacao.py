"""Importa as fotos de banana de um dataset baixado (ex.: Kaggle).

Procura, dentro da pasta informada, subpastas cujo nome contem "banana" e
"fresh"/"rotten" (padrao do dataset "Fruits fresh and rotten for
classification") e copia as fotos para dados/imagens/boa e impropria.

Lembrete: essas pastas so servem para a AVALIACAO no final. A rede treina
com as fotos misturadas, sem saber de qual pasta cada uma veio.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .dados import eh_imagem, hash_arquivo

logger = logging.getLogger(__name__)

PALAVRAS_BOA = ("fresh", "fresca", "boa", "good")
PALAVRAS_IMPROPRIA = ("rotten", "podre", "ruim", "impropria", "bad")


def classificar_pasta(nome: str) -> str | None:
    nome = nome.lower()
    if "banana" not in nome:
        return None
    if any(p in nome for p in PALAVRAS_IMPROPRIA):
        return "impropria"
    if any(p in nome for p in PALAVRAS_BOA):
        return "boa"
    return None


def importar_bananas(origem: Path, pasta_dados: Path) -> dict[str, int]:
    if not origem.is_dir():
        raise FileNotFoundError(f"Pasta nao encontrada: {origem}. Extraia o .zip antes.")
    copiadas = {"boa": 0, "impropria": 0}
    for pasta in sorted(p for p in origem.rglob("*") if p.is_dir()):
        destino_classe = classificar_pasta(pasta.name)
        if destino_classe is None:
            continue
        destino = pasta_dados / destino_classe
        destino.mkdir(parents=True, exist_ok=True)
        logger.info("Importando %s -> %s", pasta, destino_classe)
        for arquivo in sorted(pasta.iterdir()):
            if not eh_imagem(arquivo):
                continue
            # Nome pelo hash do conteudo: evita colisao (train/ e test/ tem
            # nomes repetidos) e ignora fotos duplicadas.
            alvo = destino / f"{hash_arquivo(arquivo)[:16]}{arquivo.suffix.lower()}"
            if not alvo.exists():
                shutil.copy2(arquivo, alvo)
                copiadas[destino_classe] += 1
    if not any(copiadas.values()):
        raise ValueError(f"Nenhuma pasta de banana encontrada em {origem}.")
    return copiadas
