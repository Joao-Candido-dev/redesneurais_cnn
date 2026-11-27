"""Etapa 1: treinamento do autoencoder (familia a) - SEM rotulos.

Sinal de supervisao: a propria foto. A rede recebe x e tenta devolver x.
Perda: erro quadratico medio entre a foto original e a reconstruida,

    L_rec = (1/N) * soma (x - x_reconstruido)^2

Como o gargalo e pequeno, reconstruir bem exige aprender as caracteristicas
que mais variam entre as fotos. Para que isso seja a casca (cor, manchas) e
nao a posicao ou o fundo, as fotos sao recortadas no centro e, no treino,
giradas/deslocadas ao acaso (dados.aumentar_geometria).

Se ja existe um autoencoder salvo, o treino continua dele (taxa menor) e o
novo so substitui o antigo se reconstruir melhor as fotos de validacao.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import aplicar_preprocessamento, carregar_autoencoder, salvar_autoencoder
from .config import ConfigCaminhos, ConfigTreino
from .dados import PacoteDados, aumentar_geometria, preparar_dados
from .modelo import AutoencoderConv, contar_parametros
from .utils import definir_semente, escolher_dispositivo, registrar_historico

logger = logging.getLogger(__name__)

CAMPOS_HISTORICO = ["execucao", "epoca_global", "perda_treino", "perda_validacao",
                    "taxa_aprendizado", "fotos_treino"]


@dataclass
class ResultadoAutoencoder:
    melhor_perda_validacao: float
    epocas_executadas: int
    modelo_atualizado: bool


def _executar_epoca(modelo: AutoencoderConv, loader: DataLoader, criterio: nn.Module,
                    dispositivo: torch.device, otimizador: torch.optim.Optimizer | None = None,
                    aumento: float = 0.0, ruido: float = 0.0) -> float:
    """Uma passada pelos dados. Sem otimizador = so mede (validacao, sem aumento)."""
    treinando = otimizador is not None
    modelo.train(treinando)
    soma, total = 0.0, 0
    with torch.set_grad_enabled(treinando):
        for x, _ in loader:  # o segundo item e so o indice da foto, nunca um rotulo
            x = x.to(dispositivo)
            if treinando:
                # Foto girada/deslocada: e ela que a rede deve reconstruir.
                x = aumentar_geometria(x, aumento)
            entrada = x
            if treinando and ruido > 0:
                # Denoising autoencoder: entrada suja, alvo limpo.
                entrada = (x + ruido * torch.randn_like(x)).clamp(0, 1)
            reconstrucao, _ = modelo(entrada)
            perda = criterio(reconstrucao, x)
            if treinando:
                otimizador.zero_grad()
                perda.backward()
                otimizador.step()
            soma += perda.item() * x.size(0)
            total += x.size(0)
    return soma / total


def treinar_autoencoder(caminhos: ConfigCaminhos, cfg: ConfigTreino,
                        do_zero: bool = False) -> ResultadoAutoencoder:
    definir_semente(cfg.semente)
    caminhos.criar_pastas()
    dispositivo = escolher_dispositivo()
    logger.info("Dispositivo: %s", dispositivo)

    continuar = caminhos.autoencoder.exists() and not do_zero
    if continuar:
        modelo, salvo = carregar_autoencoder(caminhos.autoencoder, dispositivo)
        aplicar_preprocessamento(cfg, salvo)  # mesmo tamanho e recorte do treino anterior
        epoca_inicial = int(salvo["epocas_totais"])
        taxa = cfg.taxa_aprendizado_continuacao
        logger.info("Continuando o autoencoder salvo (%d epocas anteriores).", epoca_inicial)
    else:
        modelo = AutoencoderConv(cfg.tamanho_imagem, cfg.dimensao_latente, cfg.canais).to(dispositivo)
        epoca_inicial = 0
        taxa = cfg.taxa_aprendizado
        logger.info("Criando autoencoder novo.")
    logger.info("Parametros: %s | espaco latente: %d dimensoes | recorte central %.0f%% | aumento %.1f",
                f"{contar_parametros(modelo):,}", modelo.dimensao_latente,
                cfg.recorte_central * 100, cfg.aumento)

    dados: PacoteDados = preparar_dados(caminhos, cfg)
    loader_treino = dados.loader(dados.indices_treino, cfg, embaralhar=True)
    loader_val = dados.loader(dados.indices_validacao, cfg)

    criterio = nn.MSELoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa, weight_decay=cfg.decaimento_pesos)
    agendador = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode="min", factor=0.5, patience=2)

    melhor = math.inf
    if continuar:
        melhor = _executar_epoca(modelo, loader_val, criterio, dispositivo)
        logger.info("Perda do modelo salvo na validacao atual: %.5f", melhor)

    execucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    registros, sem_melhora, atualizou = [], 0, False
    for epoca in range(1, cfg.epocas + 1):
        perda_treino = _executar_epoca(modelo, loader_treino, criterio, dispositivo, otimizador,
                                       aumento=cfg.aumento, ruido=cfg.ruido)
        perda_val = _executar_epoca(modelo, loader_val, criterio, dispositivo)
        agendador.step(perda_val)
        lr = otimizador.param_groups[0]["lr"]
        logger.info("Epoca %3d/%d | reconstrucao: treino %.5f | validacao %.5f",
                    epoca, cfg.epocas, perda_treino, perda_val)
        registros.append({"execucao": execucao, "epoca_global": epoca_inicial + epoca,
                          "perda_treino": round(perda_treino, 6), "perda_validacao": round(perda_val, 6),
                          "taxa_aprendizado": lr, "fotos_treino": len(dados.indices_treino)})

        if perda_val < melhor:
            melhor, sem_melhora, atualizou = perda_val, 0, True
            salvar_autoencoder(caminhos.autoencoder, modelo, cfg.recorte_central,
                               epoca_inicial + epoca, {"perda_validacao": perda_val})
            logger.info("  -> reconstrucao melhorou, modelo salvo.")
        else:
            sem_melhora += 1
            if sem_melhora >= cfg.paciencia_early_stopping:
                logger.info("Parando cedo: %d epocas sem melhora.", sem_melhora)
                break

    registrar_historico(caminhos.historico_autoencoder, CAMPOS_HISTORICO, registros)
    return ResultadoAutoencoder(melhor, len(registros), atualizou)
