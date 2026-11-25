"""Arquiteturas neurais do projeto.

Familia (a) - Aprendizado de representacao: AUTOENCODER CONVOLUCIONAL
    foto 3x96x96 -> [encoder] -> z (32 numeros) -> [decoder] -> foto reconstruida
    Sinal de supervisao: a propria foto. Perda: erro de reconstrucao (MSE).

Familia (b) - Agrupamento profundo: DEEP EMBEDDED CLUSTERING (DEC)
    Reaproveita o encoder e adiciona uma camada de agrupamento com k
    centroides treinaveis no espaco latente. Sinal de supervisao: as proprias
    atribuicoes da rede, "afiadas" (distribuicao alvo P). Perda: KL(P || Q).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Familia (a): autoencoder convolucional
# ---------------------------------------------------------------------------
class BlocoEncoder(nn.Module):
    """Convolucao 4x4 com stride 2 -> BatchNorm -> ReLU.

    O stride 2 faz o kernel "pular" de 2 em 2 pixels, entao o mapa de
    recursos sai com metade da altura e largura (faz o papel do pooling,
    mas com pesos aprendidos).
    """

    def __init__(self, entrada: int, saida: int) -> None:
        super().__init__()
        self.camadas = nn.Sequential(
            nn.Conv2d(entrada, saida, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(saida),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.camadas(x)


class BlocoDecoder(nn.Module):
    """Convolucao transposta 4x4 com stride 2: dobra a altura e a largura."""

    def __init__(self, entrada: int, saida: int, ultima: bool = False) -> None:
        super().__init__()
        camadas: list[nn.Module] = [
            nn.ConvTranspose2d(entrada, saida, kernel_size=4, stride=2, padding=1, bias=ultima)
        ]
        if ultima:
            # Sigmoide: cada pixel reconstruido fica entre 0 e 1, como a entrada.
            camadas.append(nn.Sigmoid())
        else:
            camadas += [nn.BatchNorm2d(saida), nn.ReLU(inplace=True)]
        self.camadas = nn.Sequential(*camadas)

    def forward(self, x: Tensor) -> Tensor:
        return self.camadas(x)


class AutoencoderConv(nn.Module):
    """Autoencoder convolucional.

    Com entrada 96x96 e canais (32, 64, 128, 256):
        encoder: 3x96x96 -> 32x48x48 -> 64x24x24 -> 128x12x12 -> 256x6x6
                 -> achata (9216) -> Linear -> z (dimensao_latente)
        decoder: o caminho inverso, terminando em 3x96x96 com sigmoide.

    O gargalo z e muito menor que a entrada (32 contra 27.648 valores),
    entao a rede e obrigada a guardar so o que importa para redesenhar a
    foto: cor, formato, manchas. Esse "resumo" e o espaco latente.
    """

    def __init__(self, tamanho_imagem: int = 96, dimensao_latente: int = 32,
                 canais: tuple[int, ...] = (32, 64, 128, 256)) -> None:
        super().__init__()
        reducao = 2 ** len(canais)
        if tamanho_imagem % reducao:
            raise ValueError(f"tamanho_imagem precisa ser multiplo de {reducao}.")
        self.tamanho_imagem = tamanho_imagem
        self.dimensao_latente = dimensao_latente
        self.canais = tuple(canais)
        lado = tamanho_imagem // reducao
        self._forma_gargalo = (canais[-1], lado, lado)
        achatado = canais[-1] * lado * lado

        blocos_enc, entrada = [], 3
        for saida in canais:
            blocos_enc.append(BlocoEncoder(entrada, saida))
            entrada = saida
        # Camada latente sem ativacao: z pode assumir qualquer valor real.
        self.encoder = nn.Sequential(*blocos_enc, nn.Flatten(), nn.Linear(achatado, dimensao_latente))

        self.decoder_linear = nn.Sequential(nn.Linear(dimensao_latente, achatado), nn.ReLU(inplace=True))
        invertidos = list(reversed(canais))
        blocos_dec = [BlocoDecoder(invertidos[i], invertidos[i + 1]) for i in range(len(invertidos) - 1)]
        blocos_dec.append(BlocoDecoder(invertidos[-1], 3, ultima=True))
        self.decoder_conv = nn.Sequential(*blocos_dec)

    def codificar(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decodificar(self, z: Tensor) -> Tensor:
        h = self.decoder_linear(z).view(-1, *self._forma_gargalo)
        return self.decoder_conv(h)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.codificar(x)
        return self.decodificar(z), z

    def hiperparametros(self) -> dict:
        return {"tamanho_imagem": self.tamanho_imagem,
                "dimensao_latente": self.dimensao_latente,
                "canais": list(self.canais)}


# ---------------------------------------------------------------------------
# Familia (b): Deep Embedded Clustering (Xie, Girshick e Farhadi, 2016)
# ---------------------------------------------------------------------------
class CamadaAgrupamento(nn.Module):
    """Camada com k centroides treinaveis no espaco latente.

    Atribuicao suave (distribuicao t de Student):
        q_ij = (1 + ||z_i - mu_j||^2 / alfa) ^ -((alfa + 1) / 2)   normalizado em j
    q_ij e a "probabilidade" de a foto i pertencer ao grupo j.
    """

    def __init__(self, num_grupos: int, dimensao_latente: int, alfa: float = 1.0) -> None:
        super().__init__()
        self.alfa = alfa
        self.centroides = nn.Parameter(torch.zeros(num_grupos, dimensao_latente))

    def forward(self, z: Tensor) -> Tensor:
        distancia2 = torch.cdist(z, self.centroides).pow(2)
        q = (1.0 + distancia2 / self.alfa).pow(-(self.alfa + 1.0) / 2.0)
        return q / q.sum(dim=1, keepdim=True)


def distribuicao_alvo(q: Tensor) -> Tensor:
    """Distribuicao alvo P do DEC: o "professor" que a rede cria para si mesma.

        p_ij = (q_ij^2 / f_j) normalizado em j,  com f_j = soma_i q_ij

    Elevar ao quadrado reforca as atribuicoes confiantes; dividir por f_j
    impede que um grupo grande "engula" todos os outros.
    """
    peso = q.pow(2) / q.sum(dim=0)
    return (peso.t() / peso.sum(dim=1)).t()


class ModeloDEC(nn.Module):
    """Encoder (e decoder) do autoencoder + camada de agrupamento."""

    def __init__(self, autoencoder: AutoencoderConv, num_grupos: int, alfa: float = 1.0) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.agrupamento = CamadaAgrupamento(num_grupos, autoencoder.dimensao_latente, alfa)

    @property
    def num_grupos(self) -> int:
        return self.agrupamento.centroides.shape[0]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        reconstrucao, z = self.autoencoder(x)
        return self.agrupamento(z), reconstrucao, z


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters())
