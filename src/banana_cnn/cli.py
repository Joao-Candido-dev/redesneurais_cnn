"""Interface de linha de comando (camada de apresentacao).

Comandos, na ordem de uso:
    importar    copia as fotos de banana de um dataset baixado (Kaggle)
    info        quantas fotos existem
    treinar     etapa 1: treina o autoencoder (sem rotulos)
    agrupar     etapa 2: Deep Embedded Clustering sobre o encoder (sem rotulos)
    avaliar     compara os grupos com as pastas: baseline x k-means latente x DEC
    visualizar  gera as figuras (reconstrucoes, espaco latente, grupos, curvas)
    grupo       mostra em qual grupo cai uma foto (ou pasta) nova
    interpolar  caminha no espaco latente entre duas fotos
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ConfigCaminhos, ConfigTreino

logger = logging.getLogger("banana_cnn")


def comando_importar(args, caminhos: ConfigCaminhos, cfg: ConfigTreino) -> None:
    from .importacao import importar_bananas

    copiadas = importar_bananas(Path(args.origem), caminhos.dados)
    print(f"Fotos novas copiadas: {copiadas}")


def comando_info(args, caminhos: ConfigCaminhos, cfg: ConfigTreino) -> None:
    from .dados import contar_por_rotulo, descobrir_rotulos_referencia, dividir_treino_validacao, listar_amostras

    rotulos = descobrir_rotulos_referencia(caminhos.dados)
    amostras = listar_amostras(caminhos.dados, rotulos)
    treino, validacao = dividir_treino_validacao(amostras, cfg.fracao_validacao)
    print(f"Pasta de dados: {caminhos.dados}")
    print(f"Total de fotos:  {len(amostras)}  (treino {len(treino)}, validacao {len(validacao)})")
    print(f"Por pasta (so para avaliacao): {contar_por_rotulo(amostras, rotulos)}")
    print(f"Autoencoder salvo: {'sim' if caminhos.autoencoder.exists() else 'nao'}")
    print(f"DEC salvo:         {'sim' if caminhos.dec.exists() else 'nao'}")


def comando_treinar(args, caminhos: ConfigCaminhos, cfg: ConfigTreino) -> None:
    from .treino_autoencoder import treinar_autoencoder
    from .visualizacao import plotar_curvas_autoencoder

    if args.epocas is not None:
        cfg.epocas = args.epocas
    if args.ruido is not None:
        cfg.ruido = args.ruido
    if args.latente is not None:
        cfg.dimensao_latente = args.latente
    resultado = treinar_autoencoder(caminhos, cfg, do_zero=args.do_zero)
    if resultado.modelo_atualizado:
        print(f"\nAutoencoder salvo. Melhor erro de reconstrucao na validacao: "
              f"{resultado.melhor_perda_validacao:.5f}")
    else:
        print("\nO autoencoder anterior continua sendo o melhor.")
    grafico = plotar_curvas_autoencoder(caminhos.historico_autoencoder,
                                        caminhos.relatorios / "curvas_autoencoder.png")
    print(f"Grafico: {grafico}")
    print("Proximo passo: python main.py agrupar")


def comando_agrupar(args, caminhos: ConfigCaminhos, cfg: ConfigTreino) -> None:
    from .agrupamento import treinar_dec
    from .visualizacao import plotar_curvas_dec

    if args.grupos is not None:
        cfg.num_grupos = args.grupos
    if args.epocas is not None:
        cfg.epocas_dec = args.epocas
    resultado = treinar_dec(caminhos, cfg)
    print(f"\nDEC concluido em {resultado.epocas_executadas} epocas.")
    print(f"Acuracia de referencia: k-means no latente {resultado.acuracia_inicial:.2%} "
          f"-> DEC {resultado.acuracia_final:.2%}")
    print(f"Grafico: {plotar_curvas_dec(caminhos.historico_dec, caminhos.relatorios / 'curvas_dec.png')}")
    print("Proximo passo: python main.py avaliar")


def comando_avaliar(args, caminhos: ConfigCaminhos, cfg: ConfigTreino) -> None:
    from sklearn.cluster import KMeans

    from .avaliacao import avaliar_agrupamento, extrair_atribuicoes, extrair_latentes, perda_reconstrucao
    from .baseline import executar_baseline
    from .checkpoint import carregar_autoencoder, carregar_dec
    from .dados import preparar_dados
    from .utils import escolher_dispositivo
    from .visualizacao import plotar_tabela_grupos

    dispositivo = escolher_dispositivo()
    dec, _ = carregar_dec(caminhos.dec, dispositivo)
    autoencoder, _ = carregar_autoencoder(caminhos.autoencoder, dispositivo)
    cfg.tamanho_imagem = autoencoder.tamanho_imagem
    cfg.num_grupos = dec.num_grupos
    dados = preparar_dados(caminhos, cfg)
    rotulos, nomes = dados.rotulos_array, dados.rotulos_referencia
    loader = dados.loader(None, cfg)

    base = executar_baseline(caminhos, cfg)
    z = extrair_latentes(autoencoder, loader, dispositivo)
    grupos_km = KMeans(n_clusters=cfg.num_grupos, n_init=20, random_state=cfg.semente).fit_predict(z)
    latente = avaliar_agrupamento(rotulos, grupos_km, len(nomes), cfg.num_grupos)
    grupos_dec = extrair_atribuicoes(dec, loader, dispositivo).argmax(axis=1)
    final = avaliar_agrupamento(rotulos, grupos_dec, len(nomes), cfg.num_grupos)

    print("\nComparacao (nenhum metodo viu os rotulos durante o treino):")
    print(f"  {'Metodo':<42}{'Acuracia':>10}{'NMI':>8}{'ARI':>8}")
    for nome, r in (("PCA + k-means nos pixels (baseline)", base.agrupamento),
                    ("Autoencoder + k-means no latente", latente),
                    ("Autoencoder + DEC (agrupamento profundo)", final)):
        print(f"  {nome:<42}{r.acuracia:>10.2%}{r.nmi:>8.3f}{r.ari:>8.3f}")
    print(f"\nErro de reconstrucao do autoencoder na validacao: "
          f"{perda_reconstrucao(autoencoder, dados.loader(dados.indices_validacao, cfg), dispositivo):.5f}")
    print("\nDetalhes do DEC:\n" + final.como_texto(nomes))
    if nomes:
        destino = plotar_tabela_grupos(final.matriz, nomes, caminhos.relatorios / "tabela_grupos.png")
        print(f"\nTabela salva em: {destino}")


def comando_visualizar(args, caminhos: ConfigCaminhos, cfg: ConfigTreino) -> None:
    import torch

    from .avaliacao import extrair_atribuicoes, extrair_latentes
    from .checkpoint import carregar_autoencoder, carregar_dec
    from .dados import preparar_dados
    from .utils import escolher_dispositivo
    from .visualizacao import (
        plotar_curvas_autoencoder, plotar_curvas_dec, plotar_espaco_latente,
        plotar_fotos_por_grupo, plotar_reconstrucoes,
    )

    dispositivo = escolher_dispositivo()
    autoencoder, _ = carregar_autoencoder(caminhos.autoencoder, dispositivo)
    cfg.tamanho_imagem = autoencoder.tamanho_imagem
    dados = preparar_dados(caminhos, cfg)
    rel = caminhos.relatorios
    gerados = []

    # Fotos da VALIDACAO: a rede nunca as viu durante o treino do autoencoder.
    gerador = torch.Generator().manual_seed(cfg.semente)
    escolha = torch.randperm(len(dados.indices_validacao), generator=gerador)[:10]
    amostra = dados.imagens[[dados.indices_validacao[i] for i in escolha]].float() / 255.0
    gerados.append(plotar_reconstrucoes(autoencoder, amostra, dispositivo, rel / "reconstrucoes.png"))
    gerados.append(plotar_curvas_autoencoder(caminhos.historico_autoencoder, rel / "curvas_autoencoder.png"))

    loader = dados.loader(None, cfg)
    if caminhos.dec.exists():
        dec, _ = carregar_dec(caminhos.dec, dispositivo)
        q = extrair_atribuicoes(dec, loader, dispositivo)
        z = extrair_latentes(dec.autoencoder, loader, dispositivo)
        centroides = dec.agrupamento.centroides.detach().cpu().numpy()
        gerados.append(plotar_espaco_latente(z, q.argmax(axis=1), dados.rotulos_array,
                                             dados.rotulos_referencia, rel / "espaco_latente.png", centroides))
        gerados.append(plotar_fotos_por_grupo(dados.imagens, q, rel / "fotos_por_grupo.png"))
        gerados.append(plotar_curvas_dec(caminhos.historico_dec, rel / "curvas_dec.png"))
    else:
        print("(DEC ainda nao treinado: rode 'python main.py agrupar' para as figuras de grupos)")
    print("Figuras geradas:")
    for g in gerados:
        print(f"  {g}")


def comando_grupo(args, caminhos: ConfigCaminhos, cfg: ConfigTreino) -> None:
    from .agrupamento import prever_grupos
    from .checkpoint import carregar_dec
    from .dados import carregar_tensor, eh_imagem
    from .utils import escolher_dispositivo

    dispositivo = escolher_dispositivo()
    dec, salvo = carregar_dec(caminhos.dec, dispositivo)
    mapa = salvo["metricas"].get("grupo_para_rotulo", [])
    nomes = salvo["metricas"].get("rotulos_referencia", [])
    alvo = Path(args.caminho)
    if not alvo.exists():
        raise FileNotFoundError(f"Caminho nao encontrado: {alvo}")
    arquivos = sorted(p for p in alvo.iterdir() if eh_imagem(p)) if alvo.is_dir() else [alvo]
    tamanho = dec.autoencoder.tamanho_imagem
    for arquivo in arquivos:
        q = prever_grupos(dec, carregar_tensor(arquivo, tamanho).unsqueeze(0), dispositivo)[0]
        g = int(q.argmax())
        parecido = nomes[mapa[g]] if g < len(mapa) and mapa[g] >= 0 and mapa[g] < len(nomes) else "?"
        probs = "  ".join(f"grupo {i}: {p:.1%}" for i, p in enumerate(q))
        print(f"\n{arquivo.name}\n  Grupo {g} (parecido com a pasta: {parecido})\n  {probs}")


def comando_interpolar(args, caminhos: ConfigCaminhos, cfg: ConfigTreino) -> None:
    from .checkpoint import carregar_autoencoder
    from .dados import carregar_tensor
    from .utils import escolher_dispositivo
    from .visualizacao import plotar_interpolacao

    dispositivo = escolher_dispositivo()
    autoencoder, _ = carregar_autoencoder(caminhos.autoencoder, dispositivo)
    t = autoencoder.tamanho_imagem
    destino = plotar_interpolacao(autoencoder, carregar_tensor(Path(args.foto_a), t),
                                  carregar_tensor(Path(args.foto_b), t), dispositivo,
                                  caminhos.relatorios / "interpolacao.png")
    print(f"Interpolacao salva em: {destino}")


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Bananas sem rotulos: autoencoder convolucional + Deep Embedded Clustering.",
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("importar", help="copia as bananas de um dataset baixado")
    p.add_argument("origem", help="pasta extraida do dataset (ex.: do Kaggle)")
    p.set_defaults(func=comando_importar)

    sub.add_parser("info", help="resumo das fotos").set_defaults(func=comando_info)

    p = sub.add_parser("treinar", help="etapa 1: autoencoder")
    p.add_argument("--epocas", type=int, help="numero maximo de epocas")
    p.add_argument("--do-zero", action="store_true", help="ignora o autoencoder salvo")
    p.add_argument("--ruido", type=float, help="> 0 treina um denoising autoencoder (ex.: 0.1)")
    p.add_argument("--latente", type=int, help="dimensao do espaco latente (padrao 32)")
    p.set_defaults(func=comando_treinar)

    p = sub.add_parser("agrupar", help="etapa 2: Deep Embedded Clustering")
    p.add_argument("--grupos", type=int, help="numero de grupos (padrao 2)")
    p.add_argument("--epocas", type=int, help="numero maximo de epocas do DEC")
    p.set_defaults(func=comando_agrupar)

    sub.add_parser("avaliar", help="compara com as pastas e com o baseline").set_defaults(func=comando_avaliar)
    sub.add_parser("visualizar", help="gera as figuras").set_defaults(func=comando_visualizar)

    p = sub.add_parser("grupo", help="em qual grupo cai uma foto ou pasta")
    p.add_argument("caminho")
    p.set_defaults(func=comando_grupo)

    p = sub.add_parser("interpolar", help="interpola duas fotos no espaco latente")
    p.add_argument("foto_a")
    p.add_argument("foto_b")
    p.set_defaults(func=comando_interpolar)
    return parser


def main(argv: list[str] | None = None) -> int:
    from .utils import configurar_log

    configurar_log()
    args = criar_parser().parse_args(argv)
    try:
        args.func(args, ConfigCaminhos(), ConfigTreino())
    except (OSError, ValueError, RuntimeError, KeyError) as erro:
        logger.error("%s", erro)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrompido pelo usuario.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
