"""Ciclo de feedback: cada correcao sua vira uma nova foto de treino.

Quando o modelo erra (ou acerta) uma foto, voce informa a classe correta e a
foto e copiada para dados/imagens/<classe>/. No proximo `treinar`, o modelo
aprende com ela. E assim que ele "vai acertando com o tempo".
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .dados import hash_arquivo


def registrar_exemplo(caminho_imagem: Path, classe_correta: str, pasta_dados: Path) -> Path | None:
    """Copia a foto para a pasta da classe correta.

    O nome do arquivo e o hash do conteudo, entao a mesma foto nunca e
    adicionada duas vezes. Retorna None se ela ja estava no conjunto.
    """
    pasta_classe = pasta_dados / classe_correta
    if not pasta_classe.is_dir():
        raise ValueError(f"Classe '{classe_correta}' nao existe em {pasta_dados}.")

    nome = f"{hash_arquivo(caminho_imagem)[:16]}{caminho_imagem.suffix.lower()}"
    destino = pasta_classe / nome
    if destino.exists():
        return None

    # Se a mesma foto ja tinha sido registrada em outra classe, o rotulo
    # antigo estava errado: movemos a copia em vez de duplicar com rotulos
    # conflitantes (o que confundiria a rede).
    for outra_pasta in pasta_dados.iterdir():
        anterior = outra_pasta / nome
        if outra_pasta.is_dir() and anterior.exists():
            shutil.move(str(anterior), destino)
            return destino

    shutil.copy2(caminho_imagem, destino)
    return destino
