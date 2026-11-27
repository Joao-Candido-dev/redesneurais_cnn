"""Etapa 2: Deep Embedded Clustering (familia b) - SEM rotulos.

Baseado em Xie, Girshick e Farhadi (2016), com duas variantes publicadas:
- IDEC (Guo et al., 2017): mantem a perda de reconstrucao para o espaco
  latente nao se deformar;
- DEC-DA (Guo et al., 2018): a rede ve fotos giradas/deslocadas ao acaso,
  mas o alvo P vem da foto original. Assim o grupo de uma banana nao pode
  depender da posicao ou do angulo, so da aparencia da casca.

Passo a passo:
1. Parte do encoder ja treinado na etapa 1.
2. Inicializa os k centroides no espaco latente. Como no artigo original,
   usamos k-means UMA vez apenas para dar um ponto de partida aos centroides.
   Daqui em diante tudo e treinado por gradiente, na rede neural.
3. Repete, a cada epoca:
   a. calcula Q (atribuicao suave de cada foto a cada grupo) nas fotos originais;
   b. calcula a distribuicao alvo P a partir de Q (reforca o que ja esta confiante);
   c. ajusta encoder E centroides minimizando, nas fotos aumentadas,
          L = KL(P || Q) + gama * L_reconstrucao
   O sinal de supervisao e a propria confianca da rede: nenhum rotulo humano.
4. Para quando quase nenhuma foto troca de grupo entre duas epocas
   (depois de um numero minimo de epocas).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from .avaliacao import avaliar_agrupamento, extrair_atribuicoes, extrair_latentes
from .checkpoint import aplicar_preprocessamento, carregar_autoencoder, salvar_dec
from .config import ConfigCaminhos, ConfigTreino
from .dados import aumentar_geometria, preparar_dados
from .modelo import ModeloDEC, distribuicao_alvo
from .utils import definir_semente, escolher_dispositivo, registrar_historico

logger = logging.getLogger(__name__)

CAMPOS_HISTORICO = ["execucao", "epoca", "perda_kl", "perda_reconstrucao",
                    "fracao_trocou_grupo", "acuracia_referencia"]

# Se um grupo ficar com mais que esta fracao das fotos, o agrupamento
# provavelmente "colapsou" (quase tudo num grupo so) e vale avisar.
FRACAO_GRUPO_DOMINANTE = 0.9


@dataclass
class ResultadoDEC:
    epocas_executadas: int
    acuracia_inicial: float   # k-means no espaco latente, antes do DEC
    acuracia_final: float
    tamanho_grupos: list[int]


def _congelar_batchnorm(modelo: ModeloDEC) -> None:
    """BatchNorm fica congelado: as estatisticas do autoencoder ja sao boas e
    lotes pequenos as deixariam instaveis."""
    for m in modelo.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


def _treinar_uma_epoca(modelo: ModeloDEC, loader: DataLoader, p_todas: torch.Tensor,
                       otimizador: torch.optim.Optimizer, cfg: ConfigTreino,
                       dispositivo: torch.device) -> tuple[float, float]:
    """Ajusta encoder e centroides. Devolve as perdas medias (KL, reconstrucao)."""
    modelo.train()
    _congelar_batchnorm(modelo)
    aumento = cfg.aumento if cfg.aumento_no_dec else 0.0
    soma_kl, soma_rec, total = 0.0, 0.0, 0
    for x, indices in loader:  # indices: posicao da foto, para achar sua linha em P
        x = aumentar_geometria(x.to(dispositivo), aumento)
        q, reconstrucao, _ = modelo(x)
        # clamp evita log(0) caso alguma atribuicao fique numericamente nula.
        perda_kl = F.kl_div(q.clamp_min(1e-12).log(), p_todas[indices], reduction="batchmean")
        perda_rec = F.mse_loss(reconstrucao, x)
        perda = perda_kl + cfg.peso_reconstrucao_dec * perda_rec
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
        soma_kl += perda_kl.item() * x.size(0)
        soma_rec += perda_rec.item() * x.size(0)
        total += x.size(0)
    return soma_kl / total, soma_rec / total


def treinar_dec(caminhos: ConfigCaminhos, cfg: ConfigTreino) -> ResultadoDEC:
    definir_semente(cfg.semente)
    dispositivo = escolher_dispositivo()
    autoencoder, salvo = carregar_autoencoder(caminhos.autoencoder, dispositivo)
    aplicar_preprocessamento(cfg, salvo)
    dados = preparar_dados(caminhos, cfg)
    rotulos_ref = dados.rotulos_array
    num_rotulos = len(dados.rotulos_referencia)

    # O DEC usa todas as fotos: como nao ha rotulos, nao ha o que "vazar".
    loader_ordenado = dados.loader(None, cfg)
    loader_treino = dados.loader(None, cfg, embaralhar=True)

    # --- 2. Inicializacao dos centroides ---
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

    # --- 3. Laco do DEC ---
    for epoca in range(1, cfg.epocas_dec + 1):
        # (a, b) Distribuicao alvo P nas fotos originais, fixa durante a epoca.
        q_todas = torch.tensor(extrair_atribuicoes(modelo, loader_ordenado, dispositivo))
        p_todas = distribuicao_alvo(q_todas).to(dispositivo)

        # (c) Ajusta encoder e centroides.
        perda_kl, perda_rec = _treinar_uma_epoca(modelo, loader_treino, p_todas, otimizador, cfg, dispositivo)

        grupos = extrair_atribuicoes(modelo, loader_ordenado, dispositivo).argmax(axis=1)
        trocou = float((grupos != grupos_anteriores).mean())
        grupos_anteriores = grupos
        # Acuracia de referencia: so MEDIDA para acompanhar; nao influencia
        # o gradiente nem o criterio de parada.
        atual = avaliar_agrupamento(rotulos_ref, grupos, num_rotulos, cfg.num_grupos)
        logger.info("Epoca DEC %3d/%d | KL %.5f | reconstrucao %.5f | trocaram de grupo %.2f%% | "
                    "acuracia de referencia %.2f%%", epoca, cfg.epocas_dec, perda_kl,
                    perda_rec, trocou * 100, atual.acuracia * 100)
        registros.append({"execucao": execucao, "epoca": epoca,
                          "perda_kl": round(perda_kl, 6),
                          "perda_reconstrucao": round(perda_rec, 6),
                          "fracao_trocou_grupo": round(trocou, 5),
                          "acuracia_referencia": round(atual.acuracia, 5)})
        # --- 4. Parada ---
        if epoca >= cfg.epocas_minimas_dec and trocou < cfg.tolerancia_dec:
            logger.info("Convergiu: menos de %.1f%% das fotos trocaram de grupo.", cfg.tolerancia_dec * 100)
            break

    if max(atual.tamanho_grupos) > FRACAO_GRUPO_DOMINANTE * len(grupos_anteriores):
        logger.warning("Um grupo ficou com quase todas as fotos %s. Tente outra --semente, "
                       "mais fotos ou treinar o autoencoder de novo.", atual.tamanho_grupos)

    # A correspondencia grupo -> pasta e guardada so para exibir nomes amigaveis
    # no comando `grupo` (ex.: "grupo 1 -> impropria").
    salvar_dec(caminhos.dec, modelo, cfg.recorte_central, {
        "acuracia_referencia": atual.acuracia,
        "nmi": atual.nmi,
        "grupo_para_rotulo": [atual.grupo_para_rotulo[g] for g in range(cfg.num_grupos)],
        "rotulos_referencia": list(dados.rotulos_referencia),
    })
    registrar_historico(caminhos.historico_dec, CAMPOS_HISTORICO, registros)
    return ResultadoDEC(len(registros), inicial.acuracia, atual.acuracia, atual.tamanho_grupos)


def prever_grupos(modelo: ModeloDEC, imagens: torch.Tensor, dispositivo: torch.device) -> np.ndarray:
    """Atribuicao suave Q para um lote de imagens float (N, 3, H, W)."""
    modelo.eval()
    with torch.no_grad():
        return modelo(imagens.to(dispositivo))[0].cpu().numpy()
