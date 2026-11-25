# Classificador de Bananas com CNN

Rede Neural Convolucional (CNN) que recebe a foto de uma banana e diz se ela está **boa para consumo** ou **imprópria**. O modelo aprende de forma incremental: cada foto nova que você adiciona (ou corrige) melhora o próximo treino.

> Aviso: o modelo julga apenas pela aparência da casca na foto. Ele não substitui a inspeção real (cheiro, textura, mofo interno).

---

## 1. Conceitos das aulas aplicados no código

| Conceito (Módulo 03) | Onde está no projeto |
|---|---|
| CNN como algoritmo de aprendizado profundo para imagens (Aula 02) | `src/banana_cnn/modelo.py`, classe `BananaCNN` |
| Problema de classificação: cada imagem pertence a uma classe (Aula 03) | Pastas `dados/imagens/boa` e `dados/imagens/impropria` |
| Extração de características: primeiras camadas captam bordas, as profundas captam formas complexas (Aula 03) | 4 blocos convolucionais com 32, 64, 128 e 256 filtros |
| Funções de ativação: sigmoide, ReLU, tanh (Aula 03) | ReLU nos blocos e softmax na saída |
| Agregação (pooling) reduz a dimensionalidade (Aula 03) | `MaxPool2d(2)` em cada bloco: 160 > 80 > 40 > 20 > 10 |
| Camada totalmente conectada e saída em % por classe (Aula 03) | `classificador` em `BananaCNN` e `torch.softmax` em `predicao.py` |
| Kernel inicializado com valores aleatórios e ajustado no treino (Aula 04) | `nn.Conv2d` + otimizador Adam em `treinamento.py` |
| Stride = 1 e geração do mapa de recursos (Aula 04) | `Conv2d(kernel_size=3, stride=1, padding=1)` e o comando `mapas` |

Fluxo da rede (igual ao diagrama "Arquitetura de uma CNN" da Aula 04):

```
Foto 160x160 RGB
  -> [Convolução 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2]  x4
  -> Agregação global (média)
  -> Camada totalmente conectada (256 -> 128 -> 2) com Dropout
  -> Softmax -> "boa: 92%  impropria: 8%"
```

---

## 2. Estrutura do projeto (arquitetura em camadas)

```
classificador-banana/
├── main.py                  # ponto de entrada (python main.py <comando>)
├── requirements.txt
├── dados/imagens/
│   ├── boa/                 # coloque aqui fotos de bananas boas
│   └── impropria/           # coloque aqui fotos de bananas estragadas
├── modelos/                 # modelo treinado (melhor_modelo.pt)
├── relatorios/              # gráficos e histórico de treino
├── src/banana_cnn/
│   ├── config.py            # configurações e hiperparâmetros
│   ├── dados.py             # camada de dados: leitura, divisão e aumento de dados
│   ├── modelo.py            # camada de modelo: arquitetura da CNN
│   ├── checkpoint.py        # salvar/carregar o modelo
│   ├── treinamento.py       # laço de treino incremental e early stopping
│   ├── avaliacao.py         # métricas (acurácia, precisão, revocação, F1)
│   ├── predicao.py          # inferência em novas fotos
│   ├── feedback.py          # suas correções viram novas fotos de treino
│   ├── visualizacao.py      # curvas, matriz de confusão, mapas de recursos
│   ├── baseline.py          # PCA + k-means, só para comparação
│   └── cli.py               # camada de apresentação (linha de comando)
└── tests/test_projeto.py    # testes automatizados
```

Cada módulo tem uma única responsabilidade. A interface (`cli.py`) só chama os serviços; trocar a interface por uma API ou app no futuro não exige mexer no modelo nem nos dados.

---

## 3. Passo a passo

### Passo 1: instalar

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### Passo 2: colocar as fotos

- `dados/imagens/boa/`: bananas amarelas, firmes, com poucas pintas.
- `dados/imagens/impropria/`: bananas muito escuras, moles, com mofo ou podres.

Mínimo de **10 fotos por classe** para começar. O ideal é 50 ou mais em cada.

Confira com:

```bash
python main.py info
```

### Passo 3: primeiro treino

```bash
python main.py treinar
```

O melhor modelo é salvo em `modelos/melhor_modelo.pt` e o gráfico de evolução em `relatorios/curvas_treino.png`.

### Passo 4: testar com fotos novas

```bash
python main.py prever minha_banana.jpg
python main.py prever pasta_com_fotos/
```

Saída de exemplo:

```
minha_banana.jpg
  Resultado: Boa para consumo
  Probabilidades: boa: 94.1%  impropria: 5.9%
```

Se a confiança ficar abaixo de 60%, o resultado aparece como "incerto, confira manualmente".

### Passo 5: ensinar o modelo (ele vai acertando com o tempo)

```bash
python main.py prever pasta_com_fotos/ --feedback
```

Para cada foto o programa pergunta se acertou. Se errou, você informa a classe correta. A foto é copiada para a pasta certa em `dados/imagens/`. Depois rode de novo:

```bash
python main.py treinar
```

O treino **continua a partir do modelo salvo** (não começa do zero) e só substitui o modelo antigo se o novo for melhor nas fotos de validação. Repita os passos 4 e 5 sempre que tiver fotos novas.

### Passo 6: avaliar e comparar

```bash
python main.py avaliar      # acurácia, precisão, revocação, F1 e matriz de confusão
python main.py baseline     # compara com PCA + k-means (método não neural)
python main.py mapas foto.jpg   # mostra os mapas de recursos da 1a convolução
python main.py curvas       # regenera o gráfico de evolução
python -m pytest            # roda os testes
```

### Outras opções

```bash
python main.py treinar --epocas 50          # mais épocas nesta rodada
python main.py treinar --do-zero            # descarta o modelo e recomeça
python main.py treinar --do-zero --arquitetura mobilenet   # transferência de aprendizado
```

A opção `mobilenet` usa uma MobileNetV2 pré-treinada no ImageNet (baixa os pesos na primeira vez). Costuma acertar mais quando você tem poucas fotos. A `cnn_propria` é a que segue a arquitetura das aulas e é a padrão.

---

## 4. Como o modelo melhora com o tempo

1. **Divisão estável de treino e validação:** cada foto é enviada para treino ou validação pelo hash do seu conteúdo. Fotos antigas nunca trocam de grupo, então a validação continua honesta mesmo depois de vários treinos.
2. **Ajuste fino:** ao continuar o treino, a taxa de aprendizado é menor (0,0003) para aprender o novo sem esquecer o que já sabia.
3. **Só salva se melhorar:** antes de treinar, o modelo antigo é medido na validação atual. O novo só é salvo se tiver perda menor.
4. **Histórico acumulado:** `relatorios/historico_treino.csv` guarda todas as épocas de todas as rodadas. No gráfico, as linhas tracejadas marcam cada nova rodada.
5. **Contra overfitting:** aumento de dados (rotação, espelhamento, recorte), Dropout, BatchNorm, regularização L2 e parada antecipada (early stopping).
6. **Classes desbalanceadas:** a função de perda recebe pesos maiores para a classe com menos fotos.

## 5. Dicas para tirar as fotos

- Varie fundo, iluminação, ângulo e distância.
- Uma banana (ou um cacho) por foto, ocupando boa parte da imagem.
- Inclua casos difíceis: bananas com algumas pintas mas ainda boas, e bananas que parecem boas mas estão moles ou com mofo na ponta.
- Use quantidades parecidas nas duas classes.
- Não altere cores com filtros: a cor é o principal sinal para a rede.

---

## 6. Baseline (comparação com métodos não neurais)

Conforme as regras da atividade, k-means, DBSCAN, PCA e regras de associação **não** são o foco do trabalho e aparecem apenas como comparação (Seção 5). O comando `baseline` reduz as fotos com PCA e agrupa com k-means, depois mede a acurácia na mesma validação usada pela CNN. O componente principal do projeto é a rede neural convolucional.

---

## 7. Referências

Livros-texto:

- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. MIT Press, 2016. Capítulo 9 (Convolutional Networks).

Artigos científicos revisados por pares:

- LECUN, Y.; BOTTOU, L.; BENGIO, Y.; HAFFNER, P. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, v. 86, n. 11, p. 2278-2324, 1998.
- KRIZHEVSKY, A.; SUTSKEVER, I.; HINTON, G. E. ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 2012.
- NAIR, V.; HINTON, G. E. Rectified linear units improve restricted Boltzmann machines. *ICML*, 2010.
- IOFFE, S.; SZEGEDY, C. Batch normalization: accelerating deep network training by reducing internal covariate shift. *ICML*, 2015.
- SRIVASTAVA, N. et al. Dropout: a simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, v. 15, p. 1929-1958, 2014.
- KINGMA, D. P.; BA, J. Adam: a method for stochastic optimization. *ICLR*, 2015.
- LIN, M.; CHEN, Q.; YAN, S. Network in network. *ICLR*, 2014 (agregação global pela média).
- YOSINSKI, J. et al. How transferable are features in deep neural networks? *NeurIPS*, 2014.
- SANDLER, M. et al. MobileNetV2: inverted residuals and linear bottlenecks. *CVPR*, 2018.

Baseline:

- JOLLIFFE, I. T. *Principal Component Analysis*. 2. ed. Springer, 2002.
- MACQUEEN, J. Some methods for classification and analysis of multivariate observations. *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability*, 1967.

Documentação oficial:

- PyTorch: https://pytorch.org/docs/stable/
- Torchvision (transforms e modelos): https://pytorch.org/vision/stable/

Material da disciplina:

- UTFPR. Inteligência Artificial, Módulo 03: Aulas 02 (Aplicações de CNNs), 03 (Arquitetura de CNNs) e 04 (Convolução).
