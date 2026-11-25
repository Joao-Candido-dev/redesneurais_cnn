"""Ponto de entrada: python main.py <comando>

Exemplos:
    python main.py info
    python main.py treinar
    python main.py prever minha_banana.jpg
    python main.py prever pasta_de_fotos --feedback
"""

import sys
from pathlib import Path

# Permite rodar sem instalar o pacote: adiciona src/ ao caminho de importacao.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from banana_cnn.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
