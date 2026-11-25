"""Configuracoes centrais do projeto.

Todos os hiperparametros e caminhos ficam concentrados aqui para que
nenhum outro modulo tenha "numeros magicos" espalhados pelo codigo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Raiz do projeto: src/banana_cnn/config.py -> sobe dois niveis.
RAIZ_PROJETO = Path(__file__).resolve().parents[2]

# Arquiteturas suportadas pela fabrica de modelos (ver modelo.py).
ARQUITETURA_PROPRIA = "cnn_propria"
ARQUITETURA_MOBILENET = "mobilenet"
ARQUITETURAS_SUPORTADAS = (ARQUITETURA_PROPRIA, ARQUITETURA_MOBILENET)

# Nome da pasta que representa a classe "boa para consumo".
CLASSE_BOA = "boa"

# Texto exibido ao usuario para cada classe conhecida.
VEREDITOS = {
    "boa": "Boa para consumo",
    "impropria": "Nao recomendada para consumo",
}


@dataclass(frozen=True)
class ConfigCaminhos:
    """Locais de leitura e escrita de arquivos."""

    dados: Path = RAIZ_PROJETO / "dados" / "imagens"
    modelos: Path = RAIZ_PROJETO / "modelos"
    relatorios: Path = RAIZ_PROJETO / "relatorios"

    @property
    def checkpoint(self) -> Path:
        return self.modelos / "melhor_modelo.pt"

    @property
    def historico(self) -> Path:
        return self.relatorios / "historico_treino.csv"

    def criar_pastas(self) -> None:
        for pasta in (self.dados, self.modelos, self.relatorios):
            pasta.mkdir(parents=True, exist_ok=True)


@dataclass
class ConfigTreino:
    """Hiperparametros de treinamento e inferencia."""

    arquitetura: str = ARQUITETURA_PROPRIA
    tamanho_imagem: int = 160
    tamanho_lote: int = 16
    epocas: int = 30

    # Taxa de aprendizado do primeiro treino e taxa menor usada quando o
    # modelo ja treinado recebe fotos novas (ajuste fino sem "esquecer" tudo).
    taxa_aprendizado: float = 1e-3
    taxa_aprendizado_continuacao: float = 3e-4
    decaimento_pesos: float = 1e-4

    fracao_validacao: float = 0.2
    paciencia_early_stopping: int = 7
    min_imagens_por_classe: int = 10

    # Abaixo dessa confianca a previsao e marcada como "incerta".
    limiar_confianca: float = 0.6

    semente: int = 42
    # 0 evita problemas de multiprocessamento no Windows.
    num_workers: int = 0
