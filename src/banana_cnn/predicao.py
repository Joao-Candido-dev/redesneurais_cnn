"""Inferencia: dada uma foto, diz se a banana esta boa para consumo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import ModeloCarregado, carregar_checkpoint
from .config import VEREDITOS
from .dados import carregar_imagem, eh_imagem, transformacoes_avaliacao
from .utils import escolher_dispositivo


@dataclass
class Previsao:
    caminho: Path
    classe: str
    confianca: float
    probabilidades: dict[str, float]
    incerta: bool

    @property
    def veredito(self) -> str:
        texto = VEREDITOS.get(self.classe, self.classe)
        return f"{texto} (incerto, confira manualmente)" if self.incerta else texto


class PreditorBanana:
    def __init__(self, carregado: ModeloCarregado, dispositivo: torch.device,
                 limiar_confianca: float) -> None:
        self.modelo = carregado.modelo.eval()
        self.classes = carregado.classes
        self.dispositivo = dispositivo
        self.limiar_confianca = limiar_confianca
        self.transformacao = transformacoes_avaliacao(carregado.tamanho_imagem)

    @classmethod
    def do_checkpoint(cls, caminho_checkpoint: Path, limiar_confianca: float) -> "PreditorBanana":
        dispositivo = escolher_dispositivo()
        return cls(carregar_checkpoint(caminho_checkpoint, dispositivo), dispositivo,
                   limiar_confianca)

    @torch.no_grad()
    def prever(self, caminho: Path) -> Previsao:
        entrada = self.transformacao(carregar_imagem(caminho)).unsqueeze(0)  # lote de 1
        logits = self.modelo(entrada.to(self.dispositivo))

        # Softmax converte os logits em probabilidades que somam 100%,
        # como a saida "20% mosquito / 10% besouro / 70% abelha" das aulas.
        probabilidades = torch.softmax(logits, dim=1).squeeze(0).cpu()
        indice = int(probabilidades.argmax())
        confianca = float(probabilidades[indice])
        return Previsao(
            caminho=caminho,
            classe=self.classes[indice],
            confianca=confianca,
            probabilidades={c: float(p) for c, p in zip(self.classes, probabilidades)},
            incerta=confianca < self.limiar_confianca,
        )

    def prever_varias(self, alvo: Path) -> list[Previsao]:
        """Aceita um arquivo ou uma pasta inteira de fotos."""
        if alvo.is_dir():
            arquivos = sorted(p for p in alvo.iterdir() if eh_imagem(p))
        else:
            arquivos = [alvo]
        return [self.prever(arquivo) for arquivo in arquivos]
