"""Avaliacao dos grupos encontrados SEM rotulos.

Os rotulos (nomes das pastas) entram SOMENTE aqui, depois do treino, para
responder: "os grupos que a rede descobriu sozinha batem com o que um humano
chamaria de banana boa e banana impropria?"

Metricas:
- Acuracia de agrupamento: os numeros dos grupos sao arbitrarios (o grupo 0
  pode ser "impropria"), entao testamos a melhor correspondencia grupo->rotulo
  (algoritmo hungaro) antes de contar acertos.
- NMI (informacao mutua normalizada): 0 = grupos sem relacao com os rotulos,
  1 = grupos identicos aos rotulos.
- ARI (indice de Rand ajustado): 0 = equivalente ao acaso, 1 = perfeito.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch import nn
from torch.utils.data import DataLoader

from .modelo import AutoencoderConv, ModeloDEC


@torch.no_grad()
def extrair_latentes(autoencoder: AutoencoderConv, loader: DataLoader,
                     dispositivo: torch.device) -> np.ndarray:
    """Vetor z (espaco latente) de cada foto, na ordem do loader."""
    autoencoder.eval()
    return np.concatenate([autoencoder.codificar(x.to(dispositivo)).cpu().numpy() for x, _ in loader])


@torch.no_grad()
def extrair_atribuicoes(modelo: ModeloDEC, loader: DataLoader, dispositivo: torch.device) -> np.ndarray:
    """Matriz Q (N x k): probabilidade de cada foto pertencer a cada grupo."""
    modelo.eval()
    return np.concatenate([modelo(x.to(dispositivo))[0].cpu().numpy() for x, _ in loader])


@torch.no_grad()
def perda_reconstrucao(autoencoder: AutoencoderConv, loader: DataLoader,
                       dispositivo: torch.device) -> float:
    autoencoder.eval()
    criterio = nn.MSELoss(reduction="sum")
    soma, total = 0.0, 0
    for x, _ in loader:
        x = x.to(dispositivo)
        soma += criterio(autoencoder(x)[0], x).item()
        total += x.numel()
    return soma / total


@dataclass
class ResultadoAgrupamento:
    acuracia: float
    nmi: float
    ari: float
    matriz: np.ndarray            # linhas = rotulo de referencia, colunas = grupo
    grupo_para_rotulo: dict[int, int]
    tamanho_grupos: list[int]
    avaliadas: int                # fotos com rotulo de referencia

    def como_texto(self, rotulos: list[str]) -> str:
        linhas = [
            f"Fotos avaliadas (com pasta de referencia): {self.avaliadas}",
            f"Tamanho de cada grupo: {self.tamanho_grupos}",
            f"Acuracia de agrupamento: {self.acuracia:.2%}",
            f"NMI: {self.nmi:.3f}   ARI: {self.ari:.3f}",
            "",
            "Correspondencia encontrada (grupo -> pasta mais parecida):",
        ]
        for g, r in sorted(self.grupo_para_rotulo.items()):
            linhas.append(f"  grupo {g} -> {rotulos[r] if r >= 0 else '?'}")
        linhas += ["", "Tabela (linha = pasta real, coluna = grupo descoberto):"]
        linhas.append(" " * 14 + "".join(f"{'grupo ' + str(g):>10}" for g in range(self.matriz.shape[1])))
        for i, nome in enumerate(rotulos):
            linhas.append(f"  {nome[:10]:<12}" + "".join(f"{v:>10d}" for v in self.matriz[i]))
        return "\n".join(linhas)


def avaliar_agrupamento(rotulos_ref: np.ndarray, grupos: np.ndarray, num_rotulos: int,
                        num_grupos: int) -> ResultadoAgrupamento:
    """Compara os grupos descobertos com as pastas (ignora fotos sem_rotulo)."""
    tamanhos = np.bincount(grupos, minlength=num_grupos).tolist()
    mascara = rotulos_ref >= 0
    reais, previstos = rotulos_ref[mascara], grupos[mascara]
    matriz = np.zeros((num_rotulos, num_grupos), dtype=int)
    for r, g in zip(reais, previstos):
        matriz[r, g] += 1

    if mascara.sum() == 0 or num_rotulos == 0:
        return ResultadoAgrupamento(float("nan"), float("nan"), float("nan"), matriz,
                                    {g: -1 for g in range(num_grupos)}, tamanhos, 0)

    # Algoritmo hungaro: escolhe a correspondencia grupo<->rotulo com mais acertos.
    linhas, colunas = linear_sum_assignment(-matriz)
    acertos = matriz[linhas, colunas].sum()
    mapa = {int(c): int(r) for r, c in zip(linhas, colunas)}
    # Grupos que sobraram (k > numero de pastas) recebem o rotulo majoritario.
    for g in range(num_grupos):
        if g not in mapa:
            mapa[g] = int(matriz[:, g].argmax()) if matriz[:, g].sum() else -1
            acertos += matriz[mapa[g], g] if mapa[g] >= 0 else 0

    return ResultadoAgrupamento(
        acuracia=float(acertos / mascara.sum()),
        nmi=float(normalized_mutual_info_score(reais, previstos)),
        ari=float(adjusted_rand_score(reais, previstos)),
        matriz=matriz,
        grupo_para_rotulo=mapa,
        tamanho_grupos=tamanhos,
        avaliadas=int(mascara.sum()),
    )
