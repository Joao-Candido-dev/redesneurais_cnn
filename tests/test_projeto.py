"""Testes: rode com `python -m pytest`."""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from banana_cnn.avaliacao import avaliar_agrupamento  # noqa: E402
from banana_cnn.checkpoint import aplicar_preprocessamento, carregar_autoencoder, salvar_autoencoder  # noqa: E402
from banana_cnn.cli import nome_do_grupo  # noqa: E402
from banana_cnn.config import ConfigCaminhos, ConfigTreino  # noqa: E402
from banana_cnn.dados import (  # noqa: E402
    SEM_ROTULO, aumentar_geometria, imagem_para_tensor, listar_amostras, pertence_validacao,
)
from banana_cnn.importacao import classificar_pasta  # noqa: E402
from banana_cnn.modelo import AutoencoderConv, CamadaAgrupamento, distribuicao_alvo  # noqa: E402


def test_recorte_central_descarta_as_bordas():
    imagem = Image.new("RGB", (100, 80), (0, 0, 0))           # borda preta
    imagem.paste((230, 200, 40), (25, 20, 75, 60))             # centro amarelo (50% x 50%)
    tensor = imagem_para_tensor(imagem, 16, recorte=0.5)
    assert tensor.shape == (3, 16, 16)
    assert (tensor[0] == 230).all() and (tensor[2] == 40).all()  # nada da borda sobrou


def test_aumento_mexe_na_geometria_mas_nao_na_cor():
    torch.manual_seed(0)
    lisa = torch.tensor([0.9, 0.8, 0.1]).view(1, 3, 1, 1).expand(4, 3, 32, 32).contiguous()
    assert torch.allclose(aumentar_geometria(lisa), lisa, atol=1e-5)  # foto de uma cor so continua igual
    listrada = torch.zeros(4, 3, 32, 32)
    listrada[:, :, :, :16] = 1.0
    girada = aumentar_geometria(listrada)
    assert girada.shape == listrada.shape and not torch.allclose(girada, listrada)
    assert torch.equal(aumentar_geometria(listrada, intensidade=0), listrada)


def test_checkpoint_guarda_o_recorte_do_treino(tmp_path):
    modelo = AutoencoderConv(tamanho_imagem=32, dimensao_latente=4, canais=(8, 8, 8, 8))
    salvar_autoencoder(tmp_path / "ae.pt", modelo, 0.6, epocas_totais=1, metricas={})
    _, salvo = carregar_autoencoder(tmp_path / "ae.pt", torch.device("cpu"))
    cfg = ConfigTreino(tamanho_imagem=96, recorte_central=1.0)
    aplicar_preprocessamento(cfg, salvo)
    assert cfg.tamanho_imagem == 32 and cfg.recorte_central == 0.6
    del salvo["recorte_central"]                     # modelo salvo antes desta versao
    aplicar_preprocessamento(cfg, salvo)
    assert cfg.recorte_central == 1.0


def test_nome_do_grupo_vem_da_correspondencia_salva():
    nomes = ["boa", "impropria"]
    assert nome_do_grupo(0, [1, 0], nomes) == "impropria"
    assert nome_do_grupo(1, [1, 0], nomes) == "boa"
    assert nome_do_grupo(0, [-1, 0], nomes) is None


def test_autoencoder_reconstroi_no_mesmo_tamanho():
    modelo = AutoencoderConv(tamanho_imagem=64, dimensao_latente=8, canais=(8, 16, 16, 32))
    x = torch.rand(3, 3, 64, 64)
    reconstrucao, z = modelo(x)
    assert reconstrucao.shape == x.shape
    assert z.shape == (3, 8)
    assert 0 <= reconstrucao.min() and reconstrucao.max() <= 1  # sigmoide na saida


def test_atribuicao_suave_soma_um_e_prefere_centroide_proximo():
    camada = CamadaAgrupamento(num_grupos=2, dimensao_latente=2)
    with torch.no_grad():
        camada.centroides.copy_(torch.tensor([[0.0, 0.0], [10.0, 10.0]]))
    q = camada(torch.tensor([[0.1, 0.0], [9.0, 10.0]]))
    assert torch.allclose(q.sum(dim=1), torch.ones(2))
    assert q[0, 0] > 0.9 and q[1, 1] > 0.9


def test_distribuicao_alvo_reforca_confianca():
    q = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    p = distribuicao_alvo(q)
    assert torch.allclose(p.sum(dim=1), torch.ones(2))
    assert p[0, 0] > q[0, 0] and p[1, 1] > q[1, 1]


def test_avaliacao_ignora_numeracao_dos_grupos():
    rotulos = np.array([0, 0, 1, 1, SEM_ROTULO])
    grupos = np.array([1, 1, 0, 0, 0])  # grupos trocados, mas separacao perfeita
    r = avaliar_agrupamento(rotulos, grupos, num_rotulos=2, num_grupos=2)
    assert r.acuracia == 1.0 and r.avaliadas == 4
    assert r.grupo_para_rotulo == {0: 1, 1: 0}


def test_divisao_validacao_e_estavel(tmp_path):
    foto = tmp_path / "a.png"
    Image.new("RGB", (10, 10), (200, 180, 0)).save(foto)
    assert pertence_validacao(foto, 0.2) == pertence_validacao(foto, 0.2)


def test_fotos_sem_pasta_ficam_sem_rotulo(tmp_path):
    (tmp_path / "boa").mkdir()
    (tmp_path / "sem_rotulo").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "boa" / "a.png")
    Image.new("RGB", (8, 8), (9, 9, 9)).save(tmp_path / "sem_rotulo" / "b.png")
    Image.new("RGB", (8, 8), (5, 5, 5)).save(tmp_path / "c.png")
    rotulos = {a.caminho.name: a.rotulo_referencia for a in listar_amostras(tmp_path, ["boa"])}
    assert rotulos == {"a.png": 0, "b.png": SEM_ROTULO, "c.png": SEM_ROTULO}


def test_importacao_reconhece_pastas_do_kaggle():
    assert classificar_pasta("freshbanana") == "boa"
    assert classificar_pasta("rottenbanana") == "impropria"
    assert classificar_pasta("freshapples") is None


def test_fluxo_completo_sem_rotulos(tmp_path):
    """Treina autoencoder e DEC por poucas epocas em fotos sinteticas."""
    from banana_cnn.agrupamento import treinar_dec
    from banana_cnn.treino_autoencoder import treinar_autoencoder

    gerador = np.random.default_rng(0)
    for pasta, cor in (("boa", (230, 200, 40)), ("impropria", (70, 45, 20))):
        (tmp_path / "imagens" / pasta).mkdir(parents=True)
        for i in range(15):
            pixels = np.clip(np.array(cor) + gerador.normal(0, 15, (32, 32, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(tmp_path / "imagens" / pasta / f"{i}.png")

    caminhos = ConfigCaminhos(dados=tmp_path / "imagens", modelos=tmp_path / "modelos",
                              relatorios=tmp_path / "relatorios")
    cfg = ConfigTreino(tamanho_imagem=32, dimensao_latente=4, canais=(8, 8, 16, 16),
                       epocas=3, epocas_dec=3, tamanho_lote=8)
    assert treinar_autoencoder(caminhos, cfg).modelo_atualizado
    resultado = treinar_dec(caminhos, cfg)
    assert caminhos.dec.exists()
    assert resultado.acuracia_final > 0.9  # cores bem diferentes: grupos faceis
    assert min(resultado.tamanho_grupos) > 0  # nenhum grupo ficou vazio
