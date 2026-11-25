"""Interface de linha de comando (camada de apresentacao).

Comandos:
    info       mostra quantas fotos existem em cada classe e no treino/validacao
    treinar    treina (ou continua treinando) a CNN
    prever     classifica uma foto ou pasta; com --feedback voce corrige o modelo
    avaliar    metricas completas e matriz de confusao na validacao
    baseline   compara a CNN com PCA + k-means (metodo nao neural)
    mapas      salva os mapas de recursos da 1a convolucao para uma foto
    curvas     gera o grafico de evolucao do treino
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ARQUITETURAS_SUPORTADAS, ConfigCaminhos, ConfigTreino

logger = logging.getLogger("banana_cnn")


# ---------------------------------------------------------------------------
# Handlers de cada comando
# ---------------------------------------------------------------------------
def comando_info(args: argparse.Namespace, caminhos: ConfigCaminhos, cfg: ConfigTreino) -> None:
    from .dados import contar_por_classe, descobrir_classes, dividir_treino_validacao, listar_amostras

    classes = descobrir_classes(caminhos.dados)
    amostras = listar_amostras(caminhos.dados, classes)
    treino, validacao = dividir_treino_validacao(amostras, cfg.fracao_validacao)
    print(f"Pasta de dados: {caminhos.dados}")
    print(f"Total por classe: {contar_por_classe(amostras, classes)}")
    print(f"Treino:           {contar_por_classe(treino, classes)}")
    print(f"Validacao:        {contar_por_classe(validacao, classes)}")
    print(f"Modelo salvo:     {'sim' if caminhos.checkpoint.exists() else 'nao'}")


def comando_treinar(args: argparse.Namespace, caminhos: ConfigCaminhos, cfg: ConfigTreino) -> None:
    from .treinamento import treinar_modelo
    from .visualizacao import plotar_curvas

    if args.epocas:
        cfg.epocas = args.epocas
    resultado = treinar_modelo(caminhos, cfg, do_zero=args.do_zero,
                               arquitetura_solicitada=args.arquitetura)
    if resultado.modelo_atualizado:
        print(f"\nModelo atualizado. Melhor acuracia na validacao: "
              f"{resultado.melhor_acuracia_validacao:.2%}")
    else:
        print("\nO modelo anterior continua sendo o melhor. Adicione mais fotos variadas.")
    grafico = plotar_curvas(caminhos.historico, caminhos.relatorios / "curvas_treino.png")
    print(f"Grafico de evolucao: {grafico}")


def comando_prever(args: argparse.Namespace, caminhos: ConfigCaminhos, cfg: ConfigTreino) -> None:
    from .feedback import registrar_exemplo
    from .predicao import PreditorBanana

    preditor = PreditorBanana.do_checkpoint(caminhos.checkpoint, cfg.limiar_confianca)
    alvo = Path(args.caminho)
    if not alvo.exists():
        raise FileNotFoundError(f"Caminho nao encontrado: {alvo}")

    adicionadas = 0
    for previsao in preditor.prever_varias(alvo):
        probs = "  ".join(f"{c}: {p:.1%}" for c, p in previsao.probabilidades.items())
        print(f"\n{previsao.caminho.name}")
        print(f"  Resultado: {previsao.veredito}")
        print(f"  Probabilidades: {probs}")

        if args.feedback:
            classe_correta = _perguntar_classe_correta(previsao.classe, preditor.classes)
            if classe_correta and registrar_exemplo(previsao.caminho, classe_correta, caminhos.dados):
                adicionadas += 1

    if args.feedback:
        print(f"\n{adicionadas} foto(s) adicionada(s) ao conjunto de treino.")
        if adicionadas:
            print("Rode 'python main.py treinar' para o modelo aprender com elas.")


def _perguntar_classe_correta(prevista: str, classes: list[str]) -> str | None:
    """Pergunta no terminal se a previsao estava certa e qual era a classe."""
    resposta = input("  A previsao esta correta? [s/n/pular]: ").strip().lower()
    if resposta in ("s", "sim"):
        return prevista
    if resposta in ("n", "nao"):
        opcoes = ", ".join(f"{i}={c}" for i, c in enumerate(classes))
        escolha = input(f"  Qual a classe correta? ({opcoes}): ").strip()
        if escolha.isdigit() and int(escolha) < len(classes):
            return classes[int(escolha)]
        if escolha in classes:
            return escolha
        print("  Opcao invalida, foto ignorada.")
    return None


def comando_avaliar(args: argparse.Namespace, caminhos: ConfigCaminhos, cfg: ConfigTreino) -> None:
    from .avaliacao import avaliar_modelo
    from .checkpoint import carregar_checkpoint
    from .dados import preparar_dados
    from .utils import escolher_dispositivo
    from .visualizacao import plotar_matriz_confusao

    dispositivo = escolher_dispositivo()
    carregado = carregar_checkpoint(caminhos.checkpoint, dispositivo)
    cfg.tamanho_imagem = carregado.tamanho_imagem
    dados = preparar_dados(caminhos, cfg)
    resultado = avaliar_modelo(carregado.modelo, dados.loader_validacao, dispositivo, dados.classes)
    print("\n" + resultado.como_texto(dados.classes))
    destino = plotar_matriz_confusao(resultado.matriz_confusao, dados.classes,
                                     caminhos.relatorios / "matriz_confusao.png")
    print(f"\nMatriz de confusao salva em: {destino}")


def comando_baseline(args: argparse.Namespace, caminhos: ConfigCaminhos, cfg: ConfigTreino) -> None:
    from .baseline import executar_baseline

    resultado = executar_baseline(caminhos, cfg)
    print("\nBaseline nao neural (PCA + k-means), apenas para comparacao:")
    print(f"  Componentes PCA: {resultado.componentes_pca} "
          f"({resultado.variancia_explicada:.1%} da variancia)")
    print(f"  Acuracia na validacao: {resultado.acuracia_validacao:.2%}")
    print("Compare com: python main.py avaliar")


def comando_mapas(args: argparse.Namespace, caminhos: ConfigCaminhos, cfg: ConfigTreino) -> None:
    from .checkpoint import carregar_checkpoint
    from .utils import escolher_dispositivo
    from .visualizacao import plotar_mapas_recursos

    dispositivo = escolher_dispositivo()
    carregado = carregar_checkpoint(caminhos.checkpoint, dispositivo)
    destino = plotar_mapas_recursos(
        carregado.modelo, Path(args.imagem), carregado.tamanho_imagem, dispositivo,
        caminhos.relatorios / "mapas_recursos.png",
    )
    print(f"Mapas de recursos salvos em: {destino}")


def comando_curvas(args: argparse.Namespace, caminhos: ConfigCaminhos, cfg: ConfigTreino) -> None:
    from .visualizacao import plotar_curvas

    print(plotar_curvas(caminhos.historico, caminhos.relatorios / "curvas_treino.png"))


# ---------------------------------------------------------------------------
# Montagem do parser
# ---------------------------------------------------------------------------
def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Classificador de bananas com Rede Neural Convolucional (CNN).",
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    sub.add_parser("info", help="resumo das fotos disponiveis").set_defaults(func=comando_info)

    p_treinar = sub.add_parser("treinar", help="treina ou continua treinando o modelo")
    p_treinar.add_argument("--epocas", type=int, help="numero maximo de epocas nesta rodada")
    p_treinar.add_argument("--do-zero", action="store_true",
                           help="ignora o modelo salvo e treina um novo")
    p_treinar.add_argument("--arquitetura", choices=ARQUITETURAS_SUPORTADAS,
                           help="cnn_propria (padrao) ou mobilenet (transferencia)")
    p_treinar.set_defaults(func=comando_treinar)

    p_prever = sub.add_parser("prever", help="classifica uma foto ou pasta")
    p_prever.add_argument("caminho", help="arquivo de imagem ou pasta com imagens")
    p_prever.add_argument("--feedback", action="store_true",
                          help="pergunta se acertou e guarda a foto para o proximo treino")
    p_prever.set_defaults(func=comando_prever)

    sub.add_parser("avaliar", help="metricas na validacao").set_defaults(func=comando_avaliar)
    sub.add_parser("baseline", help="compara com PCA + k-means").set_defaults(func=comando_baseline)

    p_mapas = sub.add_parser("mapas", help="visualiza os mapas de recursos")
    p_mapas.add_argument("imagem", help="foto de banana")
    p_mapas.set_defaults(func=comando_mapas)

    sub.add_parser("curvas", help="grafico de evolucao").set_defaults(func=comando_curvas)
    return parser


def main(argv: list[str] | None = None) -> int:
    from .utils import configurar_log

    configurar_log()
    args = criar_parser().parse_args(argv)
    try:
        args.func(args, ConfigCaminhos(), ConfigTreino())
    # OSError cobre arquivo inexistente e falha de download dos pesos da MobileNet.
    except (OSError, ValueError, RuntimeError) as erro:
        logger.error("%s", erro)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrompido pelo usuario.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
