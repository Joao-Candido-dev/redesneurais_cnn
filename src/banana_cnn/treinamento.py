"""Treinamento incremental da CNN.

Fluxo para o modelo "ir acertando com o tempo":

1. Voce coloca fotos novas em dados/imagens/<classe>/.
2. Roda `python main.py treinar`.
3. Se ja existe um modelo salvo, ele e carregado e continua aprendendo
   (ajuste fino com taxa de aprendizado menor) em vez de comecar do zero.
4. O modelo novo so substitui o antigo se for MELHOR na validacao atual.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import carregar_checkpoint, salvar_checkpoint
from .config import ConfigCaminhos, ConfigTreino
from .dados import PacoteDados, preparar_dados
from .modelo import contar_parametros, criar_modelo
from .utils import definir_semente, escolher_dispositivo

logger = logging.getLogger(__name__)

CAMPOS_HISTORICO = [
    "execucao", "epoca_global", "perda_treino", "acuracia_treino",
    "perda_validacao", "acuracia_validacao", "taxa_aprendizado", "fotos_treino",
]


@dataclass
class MetricasEpoca:
    perda: float
    acuracia: float


@dataclass
class ResultadoTreino:
    melhor_perda_validacao: float
    melhor_acuracia_validacao: float
    epocas_executadas: int
    modelo_atualizado: bool
    registros: list[dict] = field(default_factory=list)


class Treinador:
    """Encapsula o laco de treino, validacao, early stopping e checkpoint."""

    def __init__(
        self,
        modelo: nn.Module,
        dispositivo: torch.device,
        cfg: ConfigTreino,
        caminhos: ConfigCaminhos,
        pesos_classes: torch.Tensor,
        taxa_aprendizado: float,
    ) -> None:
        self.modelo = modelo.to(dispositivo)
        self.dispositivo = dispositivo
        self.cfg = cfg
        self.caminhos = caminhos

        # Funcao de perda para classificacao: entropia cruzada, com pesos por
        # classe para nao favorecer a classe que tem mais fotos.
        self.criterio = nn.CrossEntropyLoss(weight=pesos_classes.to(dispositivo))

        # Adam (Kingma e Ba, 2015) ajusta os pesos (inclusive os valores dos
        # kernels de convolucao) a partir do gradiente calculado na retropropagacao.
        parametros = [p for p in self.modelo.parameters() if p.requires_grad]
        self.otimizador = torch.optim.Adam(
            parametros, lr=taxa_aprendizado, weight_decay=cfg.decaimento_pesos
        )
        # Reduz a taxa de aprendizado quando a validacao para de melhorar.
        self.agendador = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.otimizador, mode="min", factor=0.5, patience=3
        )

    def _executar_epoca(self, loader: DataLoader, treinando: bool) -> MetricasEpoca:
        self.modelo.train(treinando)
        perda_total, acertos, total = 0.0, 0, 0

        with torch.set_grad_enabled(treinando):
            for imagens, rotulos in loader:
                imagens = imagens.to(self.dispositivo)
                rotulos = rotulos.to(self.dispositivo)

                logits = self.modelo(imagens)           # propagacao para frente
                perda = self.criterio(logits, rotulos)

                if treinando:
                    self.otimizador.zero_grad()
                    perda.backward()                    # retropropagacao
                    self.otimizador.step()              # atualiza os pesos

                perda_total += perda.item() * imagens.size(0)
                acertos += (logits.argmax(dim=1) == rotulos).sum().item()
                total += imagens.size(0)

        return MetricasEpoca(perda_total / total, acertos / total)

    def treinar(
        self,
        dados: PacoteDados,
        arquitetura: str,
        epoca_inicial: int,
        referencia_perda: float,
    ) -> ResultadoTreino:
        """Treina por ate cfg.epocas, salvando apenas quando houver melhora.

        `referencia_perda` e a perda do modelo anterior na validacao atual
        (infinito quando o treino comeca do zero).
        """
        execucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        melhor_perda = referencia_perda
        melhor_acuracia = 0.0
        epocas_sem_melhora = 0
        atualizou = False
        registros: list[dict] = []
        epoca_global = epoca_inicial

        for epoca in range(1, self.cfg.epocas + 1):
            epoca_global = epoca_inicial + epoca
            treino = self._executar_epoca(dados.loader_treino, treinando=True)
            validacao = self._executar_epoca(dados.loader_validacao, treinando=False)
            self.agendador.step(validacao.perda)
            taxa_atual = self.otimizador.param_groups[0]["lr"]

            logger.info(
                "Epoca %3d/%d | treino: perda %.4f acc %.2f%% | validacao: perda %.4f acc %.2f%%",
                epoca, self.cfg.epocas, treino.perda, treino.acuracia * 100,
                validacao.perda, validacao.acuracia * 100,
            )
            registros.append({
                "execucao": execucao,
                "epoca_global": epoca_global,
                "perda_treino": round(treino.perda, 5),
                "acuracia_treino": round(treino.acuracia, 5),
                "perda_validacao": round(validacao.perda, 5),
                "acuracia_validacao": round(validacao.acuracia, 5),
                "taxa_aprendizado": taxa_atual,
                "fotos_treino": dados.qtd_treino,
            })

            if validacao.perda < melhor_perda:
                melhor_perda = validacao.perda
                melhor_acuracia = validacao.acuracia
                epocas_sem_melhora = 0
                atualizou = True
                salvar_checkpoint(
                    self.caminhos.checkpoint, self.modelo, dados.classes, arquitetura,
                    self.cfg.tamanho_imagem, epoca_global,
                    {"perda_validacao": melhor_perda, "acuracia_validacao": melhor_acuracia},
                )
                logger.info("  -> modelo melhorou e foi salvo.")
            else:
                epocas_sem_melhora += 1
                # Early stopping: para quando a validacao nao melhora ha
                # varias epocas, evitando que a rede decore as fotos de treino.
                if epocas_sem_melhora >= self.cfg.paciencia_early_stopping:
                    logger.info("Parando cedo: %d epocas sem melhora.", epocas_sem_melhora)
                    break

        self._registrar_historico(registros)
        return ResultadoTreino(melhor_perda, melhor_acuracia, len(registros), atualizou, registros)

    def _registrar_historico(self, registros: list[dict]) -> None:
        arquivo = self.caminhos.historico
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        novo = not arquivo.exists()
        with arquivo.open("a", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=CAMPOS_HISTORICO)
            if novo:
                escritor.writeheader()
            escritor.writerows(registros)


@torch.no_grad()
def _perda_atual(modelo: nn.Module, loader: DataLoader, criterio: nn.Module,
                 dispositivo: torch.device) -> float:
    modelo.eval()
    soma, total = 0.0, 0
    for imagens, rotulos in loader:
        logits = modelo(imagens.to(dispositivo))
        soma += criterio(logits, rotulos.to(dispositivo)).item() * imagens.size(0)
        total += imagens.size(0)
    return soma / total


def treinar_modelo(
    caminhos: ConfigCaminhos,
    cfg: ConfigTreino,
    do_zero: bool = False,
    arquitetura_solicitada: str | None = None,
) -> ResultadoTreino:
    """Ponto de entrada do treinamento usado pela linha de comando."""
    definir_semente(cfg.semente)
    caminhos.criar_pastas()
    dispositivo = escolher_dispositivo()
    logger.info("Dispositivo: %s", dispositivo)

    continuar = caminhos.checkpoint.exists() and not do_zero
    anterior = carregar_checkpoint(caminhos.checkpoint, dispositivo) if continuar else None
    if anterior is not None:
        # O pre-processamento precisa ser identico ao do treino anterior.
        cfg.tamanho_imagem = anterior.tamanho_imagem

    dados = preparar_dados(caminhos, cfg)

    if anterior is not None:
        if anterior.classes != dados.classes:
            raise ValueError(
                f"As classes mudaram ({anterior.classes} -> {dados.classes}). "
                "Treine do zero com: python main.py treinar --do-zero"
            )
        if arquitetura_solicitada and arquitetura_solicitada != anterior.arquitetura:
            raise ValueError(
                f"O modelo salvo usa '{anterior.arquitetura}'. Para trocar de "
                "arquitetura use --do-zero."
            )
        modelo = anterior.modelo
        arquitetura = anterior.arquitetura
        epoca_inicial = anterior.epocas_totais
        taxa = cfg.taxa_aprendizado_continuacao
        logger.info("Continuando o treino do modelo salvo (%d epocas anteriores).", epoca_inicial)
    else:
        arquitetura = arquitetura_solicitada or cfg.arquitetura
        modelo = criar_modelo(arquitetura, len(dados.classes))
        epoca_inicial = 0
        taxa = cfg.taxa_aprendizado
        logger.info("Criando modelo novo: %s", arquitetura)

    total, treinaveis = contar_parametros(modelo)
    logger.info("Parametros: %s no total, %s treinaveis", f"{total:,}", f"{treinaveis:,}")

    treinador = Treinador(modelo, dispositivo, cfg, caminhos, dados.pesos_classes, taxa)

    # Referencia: desempenho do modelo antigo nas fotos de validacao ATUAIS.
    # O novo so sera salvo se superar esse valor.
    referencia = math.inf
    if continuar:
        referencia = _perda_atual(treinador.modelo, dados.loader_validacao,
                                  treinador.criterio, dispositivo)
        logger.info("Perda do modelo atual na validacao: %.4f", referencia)

    resultado = treinador.treinar(dados, arquitetura, epoca_inicial, referencia)
    if not resultado.modelo_atualizado:
        logger.info("Nenhuma melhora nesta rodada: o modelo anterior foi mantido.")
    return resultado
