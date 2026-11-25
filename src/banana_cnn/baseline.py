"""Baseline NAO neural: PCA + k-means.

Pelas regras da atividade, tecnicas como k-means e PCA NAO podem ser o
centro do trabalho; elas aparecem aqui SOMENTE como comparacao (Secao 5).
A ideia e responder: "a CNN e realmente melhor que um metodo classico?"

Funcionamento:
1. Cada foto e reduzida para 64x64 e transformada em um vetor de pixels.
2. PCA comprime esse vetor para poucas componentes principais.
3. k-means agrupa as fotos em k grupos (k = numero de classes) sem usar rotulos.
4. Cada grupo recebe o rotulo mais comum entre as fotos de treino dele.
5. Medimos a acuracia na mesma validacao usada pela CNN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .config import ConfigCaminhos, ConfigTreino
from .dados import (
    Amostra, carregar_imagem, descobrir_classes, dividir_treino_validacao, listar_amostras,
)

TAMANHO_BASELINE = 64


@dataclass
class ResultadoBaseline:
    acuracia_validacao: float
    componentes_pca: int
    variancia_explicada: float


def _vetorizar(amostras: list[Amostra]) -> tuple[np.ndarray, np.ndarray]:
    vetores = [
        np.asarray(
            carregar_imagem(a.caminho).resize((TAMANHO_BASELINE, TAMANHO_BASELINE)),
            dtype=np.float32,
        ).ravel() / 255.0
        for a in amostras
    ]
    return np.stack(vetores), np.array([a.rotulo for a in amostras])


def executar_baseline(caminhos: ConfigCaminhos, cfg: ConfigTreino) -> ResultadoBaseline:
    classes = descobrir_classes(caminhos.dados)
    treino, validacao = dividir_treino_validacao(
        listar_amostras(caminhos.dados, classes), cfg.fracao_validacao
    )
    if not treino or not validacao:
        raise ValueError("Fotos insuficientes para o baseline.")

    x_treino, y_treino = _vetorizar(treino)
    x_val, y_val = _vetorizar(validacao)

    componentes = int(min(50, len(x_treino) - 1, x_treino.shape[1]))
    pca = PCA(n_components=componentes, random_state=cfg.semente)
    z_treino = pca.fit_transform(x_treino)
    z_val = pca.transform(x_val)

    kmeans = KMeans(n_clusters=len(classes), n_init=10, random_state=cfg.semente)
    grupos_treino = kmeans.fit_predict(z_treino)

    # Associa cada grupo ao rotulo majoritario (o k-means em si nao conhece rotulos).
    rotulo_do_grupo = {}
    for grupo in range(len(classes)):
        membros = y_treino[grupos_treino == grupo]
        rotulo_do_grupo[grupo] = int(np.bincount(membros).argmax()) if membros.size else 0

    previstos = np.array([rotulo_do_grupo[g] for g in kmeans.predict(z_val)])
    return ResultadoBaseline(
        acuracia_validacao=float((previstos == y_val).mean()),
        componentes_pca=componentes,
        variancia_explicada=float(pca.explained_variance_ratio_.sum()),
    )
