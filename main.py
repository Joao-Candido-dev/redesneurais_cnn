"""Ponto de entrada: python main.py <comando>

Exemplos:
    python main.py info
    python main.py treinar          # etapa 1: autoencoder
    python main.py agrupar          # etapa 2: Deep Embedded Clustering
    python main.py avaliar
    python main.py grupo minha_banana.jpg
"""

import sys
from pathlib import Path

# Permite rodar sem instalar o pacote: adiciona src/ ao caminho de importacao.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from banana_cnn.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
