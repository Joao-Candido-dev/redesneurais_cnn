"""Configuracoes centrais do projeto.

Todos os hiperparametros e caminhos ficam concentrados aqui para que
nenhum outro modulo tenha "numeros magicos" espalhados pelo codigo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Raiz do projeto: src/banana_cnn/config.py -> sobe dois niveis.
RAIZ_PROJETO = Path(__file__).resolve().parents[2]

# Pasta para fotos sem nenhuma etiqueta. Fotos em qualquer outra subpasta de
# dados/imagens tambem sao usadas no treino SEM o nome da pasta; o nome so
# serve, no final, para medir se os grupos encontrados fazem sentido.
PASTA_SEM_ROTULO = "sem_rotulo"


@dataclass(frozen=True)
class ConfigCaminhos:
    """Locais de leitura e escrita de arquivos."""

    dados: Path = RAIZ_PROJETO / "dados" / "imagens"
    modelos: Path = RAIZ_PROJETO / "modelos"
    relatorios: Path = RAIZ_PROJETO / "relatorios"

    @property
    def autoencoder(self) -> Path:
        return self.modelos / "autoencoder.pt"

    @property
    def dec(self) -> Path:
        return self.modelos / "dec.pt"

    @property
    def cache(self) -> Path:
        return self.modelos / "cache_imagens.pt"

    @property
    def historico_autoencoder(self) -> Path:
        return self.relatorios / "historico_autoencoder.csv"

    @property
    def historico_dec(self) -> Path:
        return self.relatorios / "historico_dec.csv"

    def criar_pastas(self) -> None:
        for pasta in (self.dados, self.modelos, self.relatorios):
            pasta.mkdir(parents=True, exist_ok=True)


@dataclass
class ConfigTreino:
    """Hiperparametros do autoencoder e do agrupamento profundo (DEC)."""

    # --- Dados ---
    tamanho_imagem: int = 96          # precisa ser multiplo de 16 (4 reducoes pela metade)
    fracao_validacao: float = 0.2
    min_imagens: int = 20
    # Fracao central da foto que e mantida antes de redimensionar (1.0 = foto
    # inteira). Cortar as bordas tira fundo e os cantos pretos das fotos
    # giradas do dataset: sobra a casca, que e o que diferencia as bananas.
    recorte_central: float = 0.6
    # Intensidade do aumento geometrico (0 = desligado, 1 = completo):
    # rotacao de ate 180 graus, zoom de ate 30%, deslocamento de ate 15% e
    # espelhamento. A COR nunca e alterada: ela e o sinal de banana estragada.
    # Ver dados.aumentar_geometria.
    aumento: float = 1.0

    # --- Autoencoder (familia a) ---
    dimensao_latente: int = 32        # tamanho do "resumo" de cada foto
    canais: tuple[int, ...] = (32, 64, 128, 256)
    tamanho_lote: int = 64
    epocas: int = 40
    taxa_aprendizado: float = 1e-3
    taxa_aprendizado_continuacao: float = 3e-4
    decaimento_pesos: float = 1e-5
    paciencia_early_stopping: int = 6
    # Desvio do ruido gaussiano somado a entrada (0 = autoencoder comum,
    # > 0 = denoising autoencoder: a rede recebe a foto suja e reconstroi a limpa).
    ruido: float = 0.0

    # --- Deep Embedded Clustering (familia b) ---
    num_grupos: int = 2
    epocas_dec: int = 30
    taxa_aprendizado_dec: float = 1e-4
    # Peso da perda de reconstrucao durante o DEC (IDEC, Guo et al. 2017).
    # Mantem o espaco latente fiel as fotos enquanto os grupos se formam.
    peso_reconstrucao_dec: float = 1.0
    # Para quando menos dessa fracao de fotos troca de grupo entre epocas...
    tolerancia_dec: float = 0.001
    # ...mas so depois deste numero de epocas. Nas primeiras epocas os
    # centroides ainda andam devagar e a parada disparava antes da hora.
    epocas_minimas_dec: int = 10
    # DEC-DA (Guo et al., 2018): no DEC a rede ve a foto girada/deslocada,
    # mas o alvo P vem da foto original. Assim o encoder aprende que posicao
    # e angulo nao mudam o grupo, e os grupos passam a depender da casca.
    aumento_no_dec: bool = True
    # Grau de liberdade da distribuicao t de Student usada no DEC.
    alfa_dec: float = 1.0

    semente: int = 42
    # 0 evita problemas de multiprocessamento no Windows.
    num_workers: int = 0
