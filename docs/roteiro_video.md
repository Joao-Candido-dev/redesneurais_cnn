# Roteiro do vídeo (máx. 10 minutos, 4 integrantes)

Tema: **Bananas sem rótulos: Autoencoder Convolucional + Deep Embedded Clustering**

Como usar este roteiro:
- As falas são **pontos-chave**, não um texto para decorar. Cada um deve explicar com as próprias palavras. A rubrica avalia se o grupo domina o conteúdo, e o enunciado avisa sobre conteúdo gerado automaticamente.
- Não gastem tempo explicando a diferença entre supervisionado e não supervisionado (o enunciado pede isso explicitamente).
- Toda imagem que não for de vocês precisa da fonte na tela. Isso vale para as fotos do dataset e para figuras de artigos.
- Os números marcados com **[preencher]** saem do `python main.py avaliar` depois do treino com o dataset completo.

## Divisão

| Parte | Quem | Tempo | Assunto | Critério da rubrica |
|---|---|---|---|---|
| 1 | Integrante 1 | 0:00-2:30 | Problema, dados e arquitetura do autoencoder | Precisão arquitetural |
| 2 | Integrante 2 | 2:30-5:00 | Perda do autoencoder, treino e espaço latente | Objetivo de treinamento e experimento |
| 3 | Integrante 3 | 5:00-7:30 | Deep Embedded Clustering: arquitetura e perda | Precisão arquitetural e função de perda |
| 4 | Integrante 4 | 7:30-10:00 | Resultados, baseline, limitações e referências | Experimento, interpretação e fontes |

Deixem uns 15 segundos de folga: 10 minutos é o máximo.

---

## Parte 1: Integrante 1 (0:00-2:30). O problema e o autoencoder

**Tela:** título, nomes dos 4 integrantes e, depois, algumas fotos do dataset com a fonte.

Pontos da fala:
- (0:00-0:30) Apresenta o grupo e a pergunta: *"Uma rede neural consegue separar bananas boas de estragadas olhando só as fotos, sem ninguém dizer qual é qual?"*
- (0:30-0:50) Os dados: cerca de N fotos de bananas **[preencher]** do dataset *Fruits fresh and rotten for classification* (Kaggle). Deixar claro que **a rede nunca vê as pastas**. Elas só são usadas no final para conferir.
  - Mostrar em `src/banana_cnn/dados.py` a classe `DatasetImagens`. O `__getitem__` devolve `(imagem, indice)`, sem rótulo.

**Tela:** `docs/diagrama_arquitetura.png`, painel (a).

- (0:50-2:30) A arquitetura do autoencoder:
  - **Encoder:** 4 convoluções 4×4 com stride 2. Cada uma reduz a imagem pela metade (96 → 48 → 24 → 12 → 6) e aumenta os filtros (32 → 64 → 128 → 256). O stride 2 faz o papel do pooling, mas com pesos aprendidos.
  - Cada bloco tem **BatchNorm**, que estabiliza o treino, e **ReLU**, a ativação não linear.
  - Achatar dá 9.216 valores, e uma camada Linear leva ao **espaço latente z de 32 dimensões, sem ativação**, para que z possa assumir qualquer valor real.
  - **Decoder:** faz o caminho espelhado, com convoluções transpostas que dobram o tamanho. Termina com **sigmoide**, para cada pixel ficar entre 0 e 1 como na entrada.
  - **Por que um gargalo?** A entrada tem 27.648 números e z tem 32. Para conseguir redesenhar a foto, a rede é obrigada a guardar só o essencial: cor da casca, manchas e formato.
  - Mostrar no código: `src/banana_cnn/modelo.py`, classes `BlocoEncoder`, `BlocoDecoder` e `AutoencoderConv`.

---

## Parte 2: Integrante 2 (2:30-5:00). Treino do autoencoder e espaço latente

**Tela:** a fórmula da perda e, depois, o código.

Pontos da fala:
- (2:30-3:15) **Qual é o sinal de supervisão?** A própria foto. A entrada também é o alvo.
  - Perda: **erro quadrático médio** entre a foto e a reconstrução: `L = média (x − x̂)²`.
  - Mostrar em `src/banana_cnn/treino_autoencoder.py` a função `_executar_epoca`, na linha `perda = criterio(reconstrucao, x)`. O alvo é `x`, a própria imagem.
  - Otimizador Adam. Early stopping olhando o erro nas fotos de validação.
- (3:15-3:40) **Variante denoising** (`--ruido 0.1`): a rede recebe a foto com ruído e precisa devolver a foto limpa. Isso a obriga a aprender estrutura, e não a copiar pixels. Mostrar a linha que soma o ruído.

**Tela:** terminal rodando `python main.py treinar` (pode ser acelerado), depois `relatorios/curvas_autoencoder.png`.

- (3:40-4:05) O erro de reconstrução cai ao longo das épocas: treino e validação **[preencher o valor final]**.

**Tela:** `relatorios/reconstrucoes.png`.

- (4:05-4:35) Em cima ficam as fotos originais da **validação** (a rede nunca as viu) e embaixo as reconstruções. O que sobrevive ao gargalo é a cor e o formato. O que se perde é o detalhe fino e a textura do fundo. **Isso mostra o que a rede considerou importante.**

**Tela:** `relatorios/interpolacao.png` (`python main.py interpolar foto_boa.jpg foto_ruim.jpg`).

- (4:35-5:00) Uma analogia para o espaço latente: é como um "mapa" onde cada banana tem um endereço de 32 coordenadas. Andando em linha reta da banana amarela até a podre, o decoder desenha bananas intermediárias. Isso mostra que o espaço latente é **contínuo e organizado**.

---

## Parte 3: Integrante 3 (5:00-7:30). Deep Embedded Clustering

**Tela:** `docs/diagrama_arquitetura.png`, painel (b).

Pontos da fala:
- (5:00-5:40) **Arquitetura:** reaproveita o **mesmo encoder** e acrescenta uma **camada de agrupamento** com k = 2 centróides μ_j, que também são **parâmetros treináveis** no espaço latente.
  - Para cada foto, calcula a atribuição suave com a distribuição t de Student: `q_ij ∝ (1 + ‖z_i − μ_j‖²)^-1`. Quanto mais perto do centróide, maior a "probabilidade" de pertencer àquele grupo.
  - Mostrar em `src/banana_cnn/modelo.py` a classe `CamadaAgrupamento`, com o `nn.Parameter` dos centróides e o `forward`.
- (5:40-6:40) **Função de perda: de onde vem o sinal sem rótulos?**
  - A rede cria o próprio "professor": a **distribuição alvo P** = Q "afiada", com `p_ij ∝ q_ij² / f_j`.
    - Elevar ao quadrado reforça as atribuições em que ela já está confiante.
    - Dividir por `f_j` (o tamanho do grupo) impede que um grupo engula o outro.
  - Perda: **KL(P ‖ Q)**, que puxa cada foto para perto do seu centróide.
  - Usamos a variante **IDEC**: soma-se `γ · MSE de reconstrução`, para o espaço latente não se deformar.
  - O gradiente ajusta **o encoder e os centróides ao mesmo tempo**: a própria representação muda para separar melhor os grupos.
  - Mostrar a função `distribuicao_alvo` em `modelo.py` e o laço de `treinar_dec` em `agrupamento.py`, nas linhas de `perda_kl` e `perda`.
- (6:40-7:00) **Inicialização e parada:** os centróides começam no resultado de um k-means rodado uma vez (como no artigo do DEC). Daí em diante tudo é treinado por gradiente. O treino para quando menos de 0,1% das fotos troca de grupo.

**Tela:** terminal com `python main.py agrupar` e depois `relatorios/curvas_dec.png`.

- (7:00-7:30) A perda KL cai e as trocas de grupo vão a zero, o que indica convergência.

---

## Parte 4: Integrante 4 (7:30-10:00). Resultados, comparação e limitações

**Tela:** saída do `python main.py avaliar`.

Pontos da fala:
- (7:30-8:10) **Como avaliar sem rótulos?** Só agora usamos as pastas, depois de todo o treino.
  - **Acurácia de agrupamento:** o "grupo 0" não tem nome, então o algoritmo húngaro escolhe a melhor correspondência entre grupos e pastas antes de contar os acertos.
  - **NMI e ARI:** 0 significa o mesmo que o acaso e 1 significa grupos idênticos às pastas.
- (8:10-8:50) **Comparação com o baseline** (preencher com os números reais):

  | Método | Acurácia | NMI | ARI |
  |---|---|---|---|
  | PCA + k-means nos pixels (baseline) | [preencher] | [preencher] | [preencher] |
  | Autoencoder + k-means no latente | [preencher] | [preencher] | [preencher] |
  | Autoencoder + DEC | [preencher] | [preencher] | [preencher] |

  - **Por que neural é diferente?** O PCA é uma projeção **linear** dos pixels. O autoencoder empilha convoluções e ReLU, que é **não linear**, e reconhece uma mancha em qualquer posição da foto. O k-means só agrupa pontos fixos, enquanto o DEC **muda a própria representação** para separar os grupos.
  - Interpretem os números de verdade. Se o DEC não superar o k-means no latente, digam isso e expliquem por quê.

**Tela:** `relatorios/espaco_latente.png` e `relatorios/fotos_por_grupo.png`.

- (8:50-9:20) No espaço latente em 2D, à esquerda ficam as cores dos grupos que a rede descobriu e à direita as pastas reais. Comentem o quanto as duas figuras se parecem. Nas fotos típicas de cada grupo, o que a rede considera "parecido"?
- **Demonstração ao vivo** (opcional, 10 s): `python main.py grupo foto_nova.jpg`, com uma foto que vocês mesmos tiraram.

- (9:20-9:40) **Limitações:**
  - Sem rótulos, a rede agrupa pelo que **mais varia** nas fotos. Se o fundo variar mais que a casca, ela pode separar "fundo branco x fundo colorido".
  - A rede não sabe o que um grupo **significa**: quem dá o nome é uma pessoa.
  - Ela vê só a aparência da casca, então não detecta banana mole ou mofo por dentro.
- **Aplicações reais:** triagem em esteira de supermercado ou indústria quando não há fotos rotuladas, detecção de anomalias (foto que reconstrói mal) e compressão de imagens.

**Tela final** (de 9:40 até o fim, deixar parada uns 5 segundos): **REFERÊNCIAS** (autor, título, veículo e ano).
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. MIT Press, 2016.
- HINTON, G. E.; SALAKHUTDINOV, R. R. Reducing the dimensionality of data with neural networks. *Science*, 2006.
- VINCENT, P. et al. Extracting and composing robust features with denoising autoencoders. *ICML*, 2008.
- MASCI, J. et al. Stacked convolutional auto-encoders for hierarchical feature extraction. *ICANN*, 2011.
- XIE, J.; GIRSHICK, R.; FARHADI, A. Unsupervised deep embedding for clustering analysis. *ICML*, 2016.
- GUO, X. et al. Improved deep embedded clustering with local structure preservation. *IJCAI*, 2017.
- KINGMA, D. P.; BA, J. Adam: a method for stochastic optimization. *ICLR*, 2015.
- IOFFE, S.; SZEGEDY, C. Batch normalization. *ICML*, 2015.
- Dataset: *Fruits fresh and rotten for classification*, Kaggle (conferir autor e ano na página).
- Documentação: PyTorch (pytorch.org/docs) e scikit-learn.

---

## Checklist antes de gravar
- [ ] Treinar com o dataset completo e preencher todos os **[preencher]**
- [ ] Rodar `python main.py visualizar` e `interpolar` para gerar as figuras finais
- [ ] Cada integrante aparece e fala na sua parte, de preferência com o nome na tela
- [ ] Fonte citada em cada imagem de terceiros (fotos do dataset)
- [ ] Vídeo com no máximo 10:00, em MP4 ou link aberto (YouTube ou Drive)
- [ ] Ensaiar: cada parte cabe em cerca de 2:30
