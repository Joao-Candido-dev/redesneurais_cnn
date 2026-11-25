"""Definicao das redes neurais convolucionais.

A arquitetura propria segue exatamente o fluxo visto nas aulas:

    Entrada -> [Convolucao -> Funcao de ativacao -> Agregacao] x N
            -> Camada totalmente conectada -> Funcao de ativacao (softmax)
            -> Saida (% de cada classe)

Camadas iniciais aprendem bordas (horizontais, verticais, diagonais); camadas
mais profundas combinam essas bordas em formas mais complexas, como manchas
escuras, textura da casca e regioes moles ou mofadas.
"""

from __future__ import annotations

from torch import Tensor, nn
from torchvision import models

from .config import ARQUITETURA_MOBILENET, ARQUITETURA_PROPRIA, ARQUITETURAS_SUPORTADAS


class BlocoConvolucional(nn.Module):
    """Convolucao + normalizacao + ReLU + agregacao (max pooling).

    - Convolucao 3x3 com stride 1 e padding 1: o kernel desliza sobre a
      imagem e gera um mapa de recursos do mesmo tamanho. Os valores do
      kernel comecam aleatorios e sao ajustados no treinamento.
    - BatchNorm: estabiliza e acelera o treino (Ioffe e Szegedy, 2015).
    - ReLU(x) = max(0, x): funcao de ativacao nao linear.
    - MaxPool 2x2: reduz a dimensionalidade pela metade, mantendo os
      valores mais fortes de cada regiao.
    """

    def __init__(self, canais_entrada: int, canais_saida: int) -> None:
        super().__init__()
        self.camadas = nn.Sequential(
            nn.Conv2d(canais_entrada, canais_saida, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(canais_saida),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.camadas(x)


class BananaCNN(nn.Module):
    """CNN propria, treinada do zero.

    Com entrada 160x160 os mapas de recursos ficam:
    160 -> 80 -> 40 -> 20 -> 10 (quatro agregacoes) e o numero de filtros
    cresce de 32 ate 256 para capturar padroes cada vez mais complexos.
    """

    def __init__(
        self,
        num_classes: int,
        canais: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        blocos = []
        canais_entrada = 3  # imagem RGB
        for canais_saida in canais:
            blocos.append(BlocoConvolucional(canais_entrada, canais_saida))
            canais_entrada = canais_saida
        self.extrator_caracteristicas = nn.Sequential(*blocos)

        # Agregacao global pela media: transforma cada mapa de recursos em um
        # unico numero. Reduz muito o numero de parametros em relacao a um
        # Flatten direto, o que ajuda quando ha poucas fotos.
        self.agregacao_global = nn.AdaptiveAvgPool2d(1)

        # Camada totalmente conectada. Dropout desliga neuronios ao acaso
        # durante o treino para reduzir overfitting (Srivastava et al., 2014).
        self.classificador = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(canais_entrada, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Retorna logits (valores brutos). O softmax e aplicado apenas na
        # previsao, pois a CrossEntropyLoss ja inclui o log-softmax internamente.
        x = self.extrator_caracteristicas(x)
        x = self.agregacao_global(x)
        return self.classificador(x)


def _criar_mobilenet(num_classes: int, pretreinado: bool) -> nn.Module:
    """MobileNetV2 com transferencia de aprendizado (opcional).

    As camadas convolucionais ja aprenderam bordas e texturas em milhoes de
    imagens (ImageNet). Congelamos essas camadas e treinamos so a camada
    final, o que funciona bem mesmo com poucas dezenas de fotos.
    """
    pesos = models.MobileNet_V2_Weights.DEFAULT if pretreinado else None
    rede = models.mobilenet_v2(weights=pesos)
    for parametro in rede.features.parameters():
        parametro.requires_grad = False
    entrada_final = rede.classifier[1].in_features
    rede.classifier[1] = nn.Linear(entrada_final, num_classes)
    return rede


def criar_modelo(arquitetura: str, num_classes: int, pretreinado: bool = True) -> nn.Module:
    """Fabrica de modelos: centraliza a criacao para o resto do codigo."""
    if arquitetura == ARQUITETURA_PROPRIA:
        return BananaCNN(num_classes)
    if arquitetura == ARQUITETURA_MOBILENET:
        return _criar_mobilenet(num_classes, pretreinado)
    raise ValueError(
        f"Arquitetura '{arquitetura}' desconhecida. Use uma de: {ARQUITETURAS_SUPORTADAS}"
    )


def contar_parametros(modelo: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in modelo.parameters())
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return total, treinaveis
