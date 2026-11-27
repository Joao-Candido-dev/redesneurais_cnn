# Bananas sem rótulos: Autoencoder Convolucional + Deep Embedded Clustering

Duas redes neurais aprendem a partir de fotos de bananas **sem que ninguém diga qual banana é boa e qual é imprópria**:

1. Um **autoencoder convolucional** comprime cada foto em 32 números (o espaço latente) e aprende reconstruindo a própria foto.
2. Um **Deep Embedded Clustering (DEC)** reaproveita esse encoder e separa as fotos em grupos, ajustando a representação para que os grupos fiquem bem separados.

Só no final as pastas `boa/` e `impropria/` são usadas, para **conferir** se os grupos que a rede descobriu sozinha correspondem ao que uma pessoa diria.

---

## 1. Escopo do trabalho

| Exigência do enunciado | Como o projeto atende |
|---|---|
| (a) Família generativa ou de aprendizado de representação | **Autoencoder convolucional**, com opção de *denoising autoencoder* (`--ruido`): `src/banana_cnn/modelo.py`, classe `AutoencoderConv` |
| (b) Família de agrupamento profundo | **Deep Embedded Clustering** (Xie et al., 2016), nas variantes IDEC (Guo et al., 2017) e DEC-DA (Guo et al., 2018): `modelo.py`, classes `CamadaAgrupamento` e `ModeloDEC`, e o treino em `agrupamento.py` |
| Aprendizado sem rótulos | A rede recebe só os pixels. O `DatasetImagens` devolve `(imagem, índice)` e nunca devolve um rótulo |
| Técnicas clássicas apenas como baseline | PCA + k-means nos pixels (`baseline.py`). O k-means aparece também só como inicialização dos centróides do DEC, como no artigo original |

---

## 2. Arquiteturas

### Pré-processamento

- **Recorte central (60%)**: antes de redimensionar para 96x96, a foto perde as bordas. Isso tira fundo e os cantos pretos das fotos giradas do dataset, e o que sobra é principalmente a casca.
- **Aumento geométrico no treino**: cada foto do lote é girada (até 180°), aproximada (até 30%), deslocada (até 15%) e espelhada ao acaso. **A cor nunca é alterada**, porque ela é o principal sinal de banana estragada. Ver `aumentar_geometria` em `dados.py`.
- O recorte usado no treino fica salvo no modelo, e as fotos novas (`grupo`) são recortadas do mesmo jeito.

### (a) Autoencoder convolucional

```
Foto 3x96x96 (pixels entre 0 e 1)
  ENCODER
  -> Conv 4x4, stride 2 -> BatchNorm -> ReLU     32 x 48 x 48
  -> Conv 4x4, stride 2 -> BatchNorm -> ReLU     64 x 24 x 24
  -> Conv 4x4, stride 2 -> BatchNorm -> ReLU    128 x 12 x 12
  -> Conv 4x4, stride 2 -> BatchNorm -> ReLU    256 x  6 x  6
  -> Achatar (9216) -> Linear (sem ativação)
  = z: ESPAÇO LATENTE de 32 dimensões
  DECODER (caminho espelhado)
  -> Linear (9216) -> ReLU -> 256 x 6 x 6
  -> ConvTransposta 4x4, stride 2 -> BatchNorm -> ReLU   128 x 12 x 12
  -> ConvTransposta 4x4, stride 2 -> BatchNorm -> ReLU    64 x 24 x 24
  -> ConvTransposta 4x4, stride 2 -> BatchNorm -> ReLU    32 x 48 x 48
  -> ConvTransposta 4x4, stride 2 -> Sigmoide             3 x 96 x 96
= Foto reconstruída
```

- **Gargalo:** a entrada tem 27.648 valores e o código z tem 32. Para conseguir redesenhar a foto, a rede é obrigada a guardar só o essencial: cor da casca, manchas, formato e fundo.
- **Ativações:** ReLU nas camadas ocultas. A camada latente não tem ativação, então z pode assumir qualquer valor real. A sigmoide na saída mantém cada pixel entre 0 e 1.
- **Stride 2:** cada convolução reduz a imagem pela metade, com pesos aprendidos (faz o papel do pooling). As convoluções transpostas fazem o caminho contrário.
- Cerca de 2 milhões de parâmetros.

### (b) Deep Embedded Clustering

```
Foto -> ENCODER (o mesmo do autoencoder) -> z -> CAMADA DE AGRUPAMENTO -> q = (q_1, ..., q_k)
                                              \-> DECODER -> reconstrução (mantida, variante IDEC)
```

A camada de agrupamento guarda **k centróides treináveis** μ_j no espaço latente e calcula a atribuição suave de cada foto a cada grupo, usando uma distribuição t de Student:

```
q_ij = (1 + ||z_i - μ_j||²)^-1  /  Σ_j' (1 + ||z_i - μ_j'||²)^-1
```

---

## 3. Objetivo de treinamento: o sinal de supervisão sem rótulos

| Técnica | Sinal de supervisão | Função de perda |
|---|---|---|
| Autoencoder | A **própria foto**: a entrada também é o alvo | `L_rec = média (x - x̂)²` (MSE) |
| Denoising autoencoder (`--ruido 0.1`) | A foto **limpa**, a partir de uma versão com ruído | `média (x - decoder(encoder(x + ruído)))²` |
| DEC | A **própria confiança da rede**: a distribuição alvo P, obtida "afiando" Q | `L = KL(P ‖ Q) + γ · L_rec` |

**Distribuição alvo do DEC:**

```
p_ij = (q_ij² / f_j)  normalizado em j,    f_j = Σ_i q_ij
```

- Elevar ao quadrado reforça as atribuições em que a rede já está confiante (autotreinamento).
- Dividir por f_j impede que um grupo grande "engula" os outros.
- A KL(P‖Q) puxa cada foto para perto do seu centróide. O gradiente ajusta **o encoder e os centróides ao mesmo tempo**, então a representação muda para que os grupos fiquem mais separados. O k-means não consegue fazer isso, porque só agrupa pontos fixos.
- O termo `γ · L_rec` (IDEC) impede que o espaço latente se deforme a ponto de perder o que representa as fotos.
- **Aumento de dados no DEC (DEC-DA):** P é calculada nas fotos originais, mas Q é calculada na foto girada e deslocada. Para a perda cair, a mesma banana precisa cair no mesmo grupo em qualquer posição e ângulo. O encoder deixa de usar layout e fundo para agrupar e passa a usar a casca.
- **Parada:** quando menos de 0,1% das fotos troca de grupo entre duas épocas, depois de no mínimo 10 épocas (antes disso a parada disparava cedo demais).

### Por que essas escolhas (experimento)

Sem rótulos, a rede agrupa pelo que **mais varia** nas fotos. Neste dataset isso é o layout (ângulo, fundo, cantos pretos, cacho x banana solta), e não a cor da casca. Medimos cada mudança em 300 fotos do dataset (150 boas e 150 impróprias), com várias sementes:

| Configuração | Acurácia do DEC |
|---|---|
| Foto inteira, só espelhamento (versão anterior) | 72% a 82% |
| + recorte central | 77% a 84% |
| + aumento geométrico no autoencoder | 79% a 85% |
| + aumento também no DEC (DEC-DA), **padrão atual** | **97,0% a 97,7%** |

Treinar o autoencoder por mais épocas **piorou** o agrupamento (69% a 74% com 100 épocas): reconstruir melhor significa guardar mais layout. Numa validação com 50 fotos deixadas de fora do treino, o comando `grupo` acertou as 50.

---

## 4. Estrutura do projeto

```
redesneurais_cnn/
├── main.py                     # ponto de entrada: python main.py <comando>
├── dados/imagens/
│   ├── boa/                    # usado SÓ na avaliação final
│   ├── impropria/              # usado SÓ na avaliação final
│   └── sem_rotulo/             # (opcional) fotos sem nenhuma etiqueta
├── modelos/                    # autoencoder.pt, dec.pt e cache das imagens
├── relatorios/                 # gráficos e históricos de treino
├── src/banana_cnn/
│   ├── config.py               # hiperparâmetros
│   ├── dados.py                # leitura, divisão treino/validação e cache
│   ├── modelo.py               # AutoencoderConv, CamadaAgrupamento, ModeloDEC
│   ├── treino_autoencoder.py   # etapa 1
│   ├── agrupamento.py          # etapa 2 (DEC)
│   ├── avaliacao.py            # acurácia de agrupamento, NMI e ARI
│   ├── baseline.py             # PCA + k-means (só comparação)
│   ├── visualizacao.py         # figuras para o vídeo
│   ├── importacao.py           # copia as bananas do dataset do Kaggle
│   ├── checkpoint.py           # salvar e carregar os modelos
│   └── cli.py                  # comandos de linha de comando
└── tests/test_projeto.py
```

---

## 5. Passo a passo

### 1. Instalar (Python 3.10 ou superior)

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Obter as fotos

Baixe o dataset **"Fruits fresh and rotten for classification"** no Kaggle, extraia o `.zip` e rode:

```powershell
python main.py importar C:\caminho\para\a\pasta\extraida
python main.py info
```

As pastas `freshbanana` vão para `boa/` e as `rottenbanana` vão para `impropria/`. Fotos sem classificação podem ir para `dados/imagens/sem_rotulo/`.

### 3. Treinar e analisar

```powershell
python main.py treinar          # etapa 1: autoencoder
python main.py agrupar          # etapa 2: DEC
python main.py avaliar          # tabela: baseline x k-means no latente x DEC
python main.py visualizar       # gera as figuras em relatorios/
python main.py grupo foto.jpg   # em qual grupo cai uma foto nova
python main.py interpolar a.jpg b.jpg   # "morphing" entre duas bananas
python -m pytest                # testes
```

Opções úteis:

```powershell
python main.py treinar --do-zero --ruido 0.1   # denoising autoencoder
python main.py treinar --do-zero --latente 16  # espaço latente menor
python main.py agrupar --grupos 3              # ex.: verde / madura / estragada
python main.py agrupar --sem-aumento           # DEC clássico, para comparar no vídeo
python main.py treinar --do-zero --recorte 1 --aumento 0   # versão sem recorte e sem aumento
python main.py agrupar --semente 7             # outra semente, para ver a variação
```

---

## 6. Como interpretar os resultados

| Figura (`relatorios/`) | O que mostrar no vídeo |
|---|---|
| `curvas_autoencoder.png` | O erro de reconstrução cai: a rede está aprendendo a comprimir |
| `reconstrucoes.png` | Fotos de validação (nunca vistas) e suas reconstruções: o que sobrevive ao gargalo (cor, formato) e o que se perde (detalhes finos) |
| `espaco_latente.png` | Projeção 2D do espaço latente, colorida pelos grupos (esquerda) e pelas pastas reais (direita). Se as duas se parecem, a rede descobriu a divisão sozinha |
| `fotos_por_grupo.png` | As fotos mais típicas de cada grupo: o que a rede considera "parecido" |
| `curvas_dec.png` | A perda KL cai e as trocas de grupo vão a zero (convergência) |
| `tabela_grupos.png` | Grupos descobertos comparados com as pastas reais |
| `interpolacao.png` | O espaço latente é contínuo: pontos intermediários viram bananas intermediárias |

**Métricas** (calculadas só depois do treino):
- **Acurácia de agrupamento:** os números dos grupos são arbitrários, então o algoritmo húngaro escolhe a melhor correspondência entre grupos e pastas antes de contar os acertos.
- **NMI:** 0 significa grupos sem relação com as pastas, e 1 significa grupos idênticos a elas.
- **ARI:** 0 equivale ao acaso, e 1 é a concordância perfeita.

**Limitações para discutir:** sem rótulos, a rede agrupa pelo que mais varia nas fotos. Se o fundo variar mais que a casca, ela pode separar "fundo branco x fundo colorido" em vez de "boa x imprópria". A rede também não sabe o que um grupo *significa*: quem dá o nome aos grupos é uma pessoa, depois do treino. O recorte central e o aumento reduzem esse problema, mas fotos com objetos coloridos grandes (flores, por exemplo) ainda podem confundir o agrupamento. O comando `grupo` avisa quando a certeza fica abaixo de 60%.

---

## 7. Por que neural e não PCA + k-means?

- O **PCA** é uma projeção **linear** dos pixels. O autoencoder empilha convoluções e ReLUs, o que dá uma transformação **não linear** e capaz de reconhecer o mesmo padrão (por exemplo, uma mancha) em qualquer posição da foto.
- O **k-means** agrupa pontos fixos. O **DEC** muda a própria representação durante o treino para separar melhor os grupos.
- O comando `avaliar` mede essa diferença: compara PCA + k-means nos pixels, k-means no espaço latente do autoencoder e o DEC.

---

## 8. Referências

Livros-texto:
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. MIT Press, 2016. Cap. 9 (Convolutional Networks) e cap. 14 (Autoencoders).

Artigos científicos:
- HINTON, G. E.; SALAKHUTDINOV, R. R. Reducing the dimensionality of data with neural networks. *Science*, v. 313, n. 5786, p. 504-507, 2006.
- VINCENT, P.; LAROCHELLE, H.; BENGIO, Y.; MANZAGOL, P.-A. Extracting and composing robust features with denoising autoencoders. *ICML*, 2008.
- MASCI, J.; MEIER, U.; CIREŞAN, D.; SCHMIDHUBER, J. Stacked convolutional auto-encoders for hierarchical feature extraction. *ICANN*, 2011.
- XIE, J.; GIRSHICK, R.; FARHADI, A. Unsupervised deep embedding for clustering analysis. *ICML*, 2016.
- GUO, X.; GAO, L.; LIU, X.; YIN, J. Improved deep embedded clustering with local structure preservation. *IJCAI*, 2017.
- GUO, X.; ZHU, E.; LIU, X.; YIN, J. Deep embedded clustering with data augmentation. *ACML*, PMLR 95, p. 550-565, 2018.
- LECUN, Y.; BOTTOU, L.; BENGIO, Y.; HAFFNER, P. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, v. 86, n. 11, p. 2278-2324, 1998.
- IOFFE, S.; SZEGEDY, C. Batch normalization: accelerating deep network training by reducing internal covariate shift. *ICML*, 2015.
- KINGMA, D. P.; BA, J. Adam: a method for stochastic optimization. *ICLR*, 2015.
- VAN DER MAATEN, L.; HINTON, G. Visualizing data using t-SNE. *JMLR*, v. 9, p. 2579-2605, 2008 (origem da distribuição t de Student usada no DEC).

Métricas e baseline:
- KUHN, H. W. The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, v. 2, p. 83-97, 1955.
- HUBERT, L.; ARABIE, P. Comparing partitions. *Journal of Classification*, v. 2, p. 193-218, 1985 (ARI).
- STREHL, A.; GHOSH, J. Cluster ensembles: a knowledge reuse framework for combining multiple partitions. *JMLR*, v. 3, p. 583-617, 2002 (NMI).
- JOLLIFFE, I. T. *Principal Component Analysis*. 2. ed. Springer, 2002.
- MACQUEEN, J. Some methods for classification and analysis of multivariate observations. *Proc. 5th Berkeley Symposium*, 1967.

Dataset:
- KALLURI, S. R. *Fruits fresh and rotten for classification*. Kaggle, 2018. Confira autor e ano na página do dataset antes de citar.

Documentação oficial:
- PyTorch: https://pytorch.org/docs/stable/
- scikit-learn (métricas de agrupamento): https://scikit-learn.org/stable/modules/clustering.html
