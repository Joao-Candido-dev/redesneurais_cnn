"""Metricas de avaliacao: acuracia, matriz de confusao, precisao, revocacao e F1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class ResultadoAvaliacao:
    acuracia: float
    matriz_confusao: np.ndarray  # linhas = classe real, colunas = classe prevista
    por_classe: dict[str, dict[str, float]]

    def como_texto(self, classes: list[str]) -> str:
        linhas = [f"Acuracia geral: {self.acuracia:.2%}", "", "Por classe:"]
        for classe in classes:
            m = self.por_classe[classe]
            linhas.append(
                f"  {classe:<12} precisao={m['precisao']:.2%}  "
                f"revocacao={m['revocacao']:.2%}  f1={m['f1']:.2%}  (n={int(m['suporte'])})"
            )
        linhas += ["", "Matriz de confusao (linha = real, coluna = prevista):"]
        cabecalho = " " * 14 + "".join(f"{c[:10]:>12}" for c in classes)
        linhas.append(cabecalho)
        for i, classe in enumerate(classes):
            valores = "".join(f"{v:>12d}" for v in self.matriz_confusao[i])
            linhas.append(f"  {classe[:10]:<12}{valores}")
        return "\n".join(linhas)


@torch.no_grad()
def coletar_previsoes(
    modelo: nn.Module, loader: DataLoader, dispositivo: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    modelo.eval()
    reais, previstos = [], []
    for imagens, rotulos in loader:
        logits = modelo(imagens.to(dispositivo))
        previstos.append(logits.argmax(dim=1).cpu().numpy())
        reais.append(rotulos.numpy())
    return np.concatenate(reais), np.concatenate(previstos)


def calcular_metricas(
    reais: np.ndarray, previstos: np.ndarray, classes: list[str]
) -> ResultadoAvaliacao:
    n = len(classes)
    matriz = np.zeros((n, n), dtype=int)
    for r, p in zip(reais, previstos):
        matriz[r, p] += 1

    por_classe = {}
    for i, classe in enumerate(classes):
        verdadeiro_positivo = matriz[i, i]
        previstos_como_classe = matriz[:, i].sum()
        reais_da_classe = matriz[i, :].sum()
        precisao = verdadeiro_positivo / previstos_como_classe if previstos_como_classe else 0.0
        revocacao = verdadeiro_positivo / reais_da_classe if reais_da_classe else 0.0
        f1 = (2 * precisao * revocacao / (precisao + revocacao)) if (precisao + revocacao) else 0.0
        por_classe[classe] = {
            "precisao": float(precisao),
            "revocacao": float(revocacao),
            "f1": float(f1),
            "suporte": float(reais_da_classe),
        }

    acuracia = float(np.trace(matriz) / matriz.sum()) if matriz.sum() else 0.0
    return ResultadoAvaliacao(acuracia, matriz, por_classe)


def avaliar_modelo(
    modelo: nn.Module, loader: DataLoader, dispositivo: torch.device, classes: list[str]
) -> ResultadoAvaliacao:
    reais, previstos = coletar_previsoes(modelo, loader, dispositivo)
    return calcular_metricas(reais, previstos, classes)
