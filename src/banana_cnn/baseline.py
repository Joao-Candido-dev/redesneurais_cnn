"""Baseline NAO neural: PCA + k-means direto nos pixels.

Pelas regras da atividade, PCA e k-means NAO podem ser o centro do trabalho;
aparecem aqui SOMENTE como comparacao. A pergunta e: "a rede neural encontra
grupos melhores que o metodo classico?"

Diferenca principal:
- PCA e uma projecao LINEAR: so consegue combinar pixels com pesos fixos.
- O autoencoder usa varias camadas com ReLU (transformacao NAO linear) e
  convolucoes, que reconhecem o mesmo padrao (ex.: uma mancha) em qualquer
  posicao da foto.
- k-means so agrupa; o DEC ajusta a propria representacao para que os grupos
  fiquem mais separados.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .avaliacao import ResultadoAgrupamento, avaliar_agrupamento
from .config import ConfigCaminhos, ConfigTreino
from .dados import preparar_dados

TAMANHO_BASELINE = 32


@dataclass
class ResultadoBaseline:
    agrupamento: ResultadoAgrupamento
    componentes_pca: int
    variancia_explicada: float


def executar_baseline(caminhos: ConfigCaminhos, cfg: ConfigTreino) -> ResultadoBaseline:
    dados = preparar_dados(caminhos, cfg)
    # Reduz as fotos para 32x32 e transforma cada uma num vetor de pixels.
    imagens = dados.imagens.float() / 255.0
    passo = max(1, imagens.shape[-1] // TAMANHO_BASELINE)
    vetores = imagens[:, :, ::passo, ::passo].reshape(len(imagens), -1).numpy()

    componentes = int(min(cfg.dimensao_latente, len(vetores) - 1, vetores.shape[1]))
    pca = PCA(n_components=componentes, random_state=cfg.semente)
    z = pca.fit_transform(vetores)
    grupos = KMeans(n_clusters=cfg.num_grupos, n_init=20, random_state=cfg.semente).fit_predict(z)

    return ResultadoBaseline(
        agrupamento=avaliar_agrupamento(dados.rotulos_array, grupos,
                                        len(dados.rotulos_referencia), cfg.num_grupos),
        componentes_pca=componentes,
        variancia_explicada=float(np.sum(pca.explained_variance_ratio_)),
    )
