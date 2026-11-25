"""Etapa 2: Deep Embedded Clustering (familia b) - SEM rotulos.

Baseado em Xie, Girshick e Farhadi (2016) e na variante IDEC de Guo et al.
(2017), que mantem a perda de reconstrucao para o espaco latente nao se
deformar.

Passo a passo:
1. Parte do encoder ja treinado na etapa 1.
2. Inicializa os k centroides no espaco latente. Como no artigo original,
   usamos k-means UMA vez apenas para dar um ponto de partida aos centroides.
   Daqui em diante tudo e treinado por gradiente, na rede neural.
3. Repete:
   a. calcula Q (atribuicao suave de cada foto a cada grupo);
   b. calcula a distribuicao alvo P a partir de Q (reforca o que ja esta confiante);
   c. ajusta encoder E centroides minimizando
          L = KL(P || Q) + gama * L_reconstrucao
   O sinal de supervisao e a propria confianca da rede: nenhum rotulo humano.
4. Para quando quase nenhuma foto troca de grupo entre duas epocas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .avaliacao import avaliar_agrupamento, extrair_atribuicoes, extrair_latentes
from .checkpoint import carregar_autoencoder, salvar_dec
from .config import ConfigCaminhos, ConfigTreino
from .dados import preparar_dados
from .modelo import ModeloDEC, distribuicao_alvo
from .treino_autoencoder import registrar_historico
from .utils import definir_semente, escolher_dispositivo

logger = logging.getLogger(__name__)

CAMPOS_HISTORICO = ["execucao", "epoca", "perda_kl", "perda_reconstrucao",
                    "fracao_trocou_grupo", "acuracia_referencia"]


@dataclass
class ResultadoDEC:
    epocas_executadas: int
    acuracia_inicial: float   # k-means no espaco latente, antes do DEC
    acuracia_final: float


def treinar_dec(caminhos: ConfigCaminhos, cfg: ConfigTreino) -> ResultadoDEC:
    definir_semente(cfg.semente)
    dispositivo = escolher_dispositivo()
    autoencoder, _ = carregar_autoencoder(caminhos.autoencoder, dispositivo)
    cfg.tamanho_imagem = autoencoder.tamanho_imagem
    dados = preparar_dados(caminhos, cfg)
    rotulos_ref = dados.rotulos_array
    num_rotulos = len(dados.rotulos_referencia)

    # O DEC usa todas as fotos: como nao ha rotulos, nao ha o que "vazar".
    loader_ordenado = dados.loader(None, cfg)
    loader_treino = dados.loader(None, cfg, embaralhar=True)

    # --- Inicializacao dos centroides ---
    z = extrair_latentes(autoencoder, loader_ordenado, dispositivo)
    kmeans = KMeans(n_clusters=cfg.num_grupos, n_init=20, random_state=cfg.semente)
    grupos_anteriores = kmeans.fit_predict(z)
    inicial = avaliar_agrupamento(rotulos_ref, grupos_anteriores, num_rotulos, cfg.num_grupos)
    logger.info("Ponto de partida (k-means no espaco latente): acuracia de referencia %.2f%%",
                inicial.acuracia * 100)

    modelo = ModeloDEC(autoencoder, cfg.num_grupos, cfg.alfa_dec).to(dispositivo)
    with torch.no_grad():
        modelo.agrupamento.centroides.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))

    otimizador = torch.optim.Adam(modelo.parameters(), lr=cfg.taxa_aprendizado_dec)
    execucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    registros = []
    atual = inicial

    for epoca in range(1, cfg.epocas_dec + 1):
        # (b) Distribuicao alvo P, recalculada a cada epoca e mantida fixa nela.
        q_todas = torch.tensor(extrair_atribuicoes(modelo, loader_ordenado, dispositivo))
        p_todas = distribuicao_alvo(q_todas).to(dispositivo)

        # (c) Ajusta encoder e centroides.
        modelo.train()
        # BatchNorm fica congelado: as estatisticas do autoencoder ja sao boas
        # e lotes pequenos as deixariam instaveis.
        for m in modelo.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
        soma_kl, soma_rec, total = 0.0, 0.0, 0
        for x, indices in loader_treino:
            x = x.to(dispositivo)
            q, reconstrucao, _ = modelo(x)
            perda_kl = F.kl_div(q.log(), p_todas[indices], reduction="batchmean")
            perda_rec = F.mse_loss(reconstrucao, x)
            perda = perda_kl + cfg.peso_reconstrucao_dec * perda_rec
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            soma_kl += perda_kl.item() * x.size(0)
            soma_rec += perda_rec.item() * x.size(0)
            total += x.size(0)

        grupos = extrair_atribuicoes(modelo, loader_ordenado, dispositivo).argmax(axis=1)
        trocou = float((grupos != grupos_anteriores).mean())
        grupos_anteriores = grupos
        # Acuracia de referencia: so MEDIDA para acompanhar; nao influencia
        # o gradiente nem o criterio de parada.
        atual = avaliar_agrupamento(rotulos_ref, grupos, num_rotulos, cfg.num_grupos)
        logger.info("Epoca DEC %3d/%d | KL %.5f | reconstrucao %.5f | trocaram de grupo %.2f%% | "
                    "acuracia de referencia %.2f%%", epoca, cfg.epocas_dec, soma_kl / total,
                    soma_rec / total, trocou * 100, atual.acuracia * 100)
        registros.append({"execucao": execucao, "epoca": epoca,
                          "perda_kl": round(soma_kl / total, 6),
                          "perda_reconstrucao": round(soma_rec / total, 6),
                          "fracao_trocou_grupo": round(trocou, 5),
                          "acuracia_referencia": round(atual.acuracia, 5)})
        if epoca > 1 and trocou < cfg.tolerancia_dec:
            logger.info("Convergiu: menos de %.1f%% das fotos trocaram de grupo.", cfg.tolerancia_dec * 100)
            break

    # A correspondencia grupo -> pasta e guardada so para exibir nomes amigaveis
    # no comando `grupo` (ex.: "grupo 1, parecido com: impropria").
    salvar_dec(caminhos.dec, modelo, {
        "acuracia_referencia": atual.acuracia,
        "nmi": atual.nmi,
        "grupo_para_rotulo": [atual.grupo_para_rotulo[g] for g in range(cfg.num_grupos)],
        "rotulos_referencia": list(dados.rotulos_referencia),
    })
    registrar_historico(caminhos.historico_dec, CAMPOS_HISTORICO, registros)
    return ResultadoDEC(len(registros), inicial.acuracia, atual.acuracia)


def prever_grupos(modelo: ModeloDEC, imagens: torch.Tensor, dispositivo: torch.device) -> np.ndarray:
    """Atribuicao suave Q para um lote de imagens float (N, 3, H, W)."""
    modelo.eval()
    with torch.no_grad():
        return modelo(imagens.to(dispositivo))[0].cpu().numpy()
