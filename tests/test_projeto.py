"""Testes basicos: rode com `python -m pytest`."""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from banana_cnn.avaliacao import calcular_metricas  # noqa: E402
from banana_cnn.dados import pertence_validacao  # noqa: E402
from banana_cnn.feedback import registrar_exemplo  # noqa: E402
from banana_cnn.modelo import BananaCNN, BlocoConvolucional  # noqa: E402


def test_bloco_reduz_resolucao_pela_metade():
    bloco = BlocoConvolucional(3, 8)
    saida = bloco(torch.randn(2, 3, 64, 64))
    assert saida.shape == (2, 8, 32, 32)


def test_cnn_retorna_um_valor_por_classe():
    modelo = BananaCNN(num_classes=2)
    saida = modelo(torch.randn(4, 3, 160, 160))
    assert saida.shape == (4, 2)


def test_divisao_validacao_e_estavel(tmp_path):
    foto = tmp_path / "a.png"
    Image.new("RGB", (10, 10), (200, 180, 0)).save(foto)
    assert pertence_validacao(foto, 0.2) == pertence_validacao(foto, 0.2)


def test_metricas_com_acerto_total():
    reais = np.array([0, 0, 1, 1])
    resultado = calcular_metricas(reais, reais.copy(), ["boa", "impropria"])
    assert resultado.acuracia == 1.0
    assert resultado.por_classe["boa"]["f1"] == 1.0


def test_feedback_nao_duplica_e_corrige_rotulo(tmp_path):
    dados = tmp_path / "imagens"
    (dados / "boa").mkdir(parents=True)
    (dados / "impropria").mkdir()
    foto = tmp_path / "foto.jpg"
    Image.new("RGB", (10, 10), (90, 60, 20)).save(foto)

    primeiro = registrar_exemplo(foto, "boa", dados)
    assert primeiro is not None
    assert registrar_exemplo(foto, "boa", dados) is None  # duplicata ignorada

    corrigido = registrar_exemplo(foto, "impropria", dados)
    assert corrigido.parent.name == "impropria"
    assert not primeiro.exists()
