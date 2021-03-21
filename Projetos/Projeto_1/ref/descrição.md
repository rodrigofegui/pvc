# Projeto 1 - Visão estéreo

## Objetivo
Esta atividade tem como objetivo a exploração e desenvolvimento de algoritmos de visão estéreo para:
- extrair mapas de profundidade;
- usar dados de calibração de câmeras e
- medir objetos em 3D.

## Material necessário
- **Dados:** Obtenha as imagens e dados disponibilizados para o trabalho 1 no moodle. Todas as atividades praticadas neste trabalho se aplicam especificamente a esses dados.
- **Hardware:** um computador com no mínimo 200MB de espaço em disco livre. Caso não disponha de uma maquina, é possível resolver este trabalho usando plataformas de nuvem como Google CoLab, PaperSpace, etc.
- **Software:** seu código será testado em computadores que tem sistema operacional Linux 64 bits, openCV versão 4.4 e Python 3.7. Você não é obrigado a desenvolver em Linux mas é importante que desenvolva de forma que funcione em Linux.
    - Recomendamos o uso de Python, porém, versões em C ou C++ também são bem vindas, desde que bem documentadas e com compilação baseada em CMake files.
    - Tente minimizar a dependência a bibliotecas que não sejam padrão (além de OpenCV).
- **Wetware** com capacidade de implementar algoritmos de visão computacional.

## Procedimentos
### Requisito 1: estimativa de mapa de profundidade a partir de imagens estéreo retificadas
Explore o site da [base de imagens de Middlebury de 2014](http://vision.middlebury.edu/stereo/data/scenes2014/), com atenção especial às instruções na parte inferior da página. As imagens disponíveis foram retificadas e alinhadas em paralelo (ou seja, linhas epipolares são horizontais e $y_L = y_R$).

Para cada par de imagens estéreo, há duas versões dos dados. Para este trabalho, considere somente a versão **perfect**.

Para cada cena, há um arquivo com informações da calibração de ambas as câmeras retificadas (**calib.txt**), cujos detalhes são explicados na página principal.

Para gerar resultados comparáveis com os de seus colegas, considere as cenas [Jadeplant](https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Jadeplant-perfect/) e [PlayTable](https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Playtable-perfect/) e realize seus experimentos com as imagens default (im0.png e im1.png).

Realize as seguintes tarefas:
1. Desenvolva um algoritmo que, para cada ponto na imagem do lado esquerdo, encontre o ponto respectivo na imagem da direita.
    - Observe que não haverá correlação para todos os pontos das imagens, por exemplo, alguns pontos em 3D são visíveis numa imagem mas são ocluídos ou ficam fora do campo de vista da outra. Há também casos em que uma região da imagem é similar a várias outras regiões, como no caso de regiões sem textura.
    - Há várias estratégia para que tais casos sejam tratados, algumas foram discutidas em sala de aula. Adote uma delas ou proponha alguma heurística.
2. Para cada ponto da imagem da esquerda que tenha correspondência com a imagem da direita, calcule os valores das coordenadas X, Y e Z do mundo.
3. Para todos os pontos calculados, crie dois pares de imagens resultantes:
    - **disparidade.pgm**: mapa de disparidade, em que cada pixel armazena o valor da distância, em pixeis da imagem da esquerda para a da direita, ou seja, de im0.png para im1.png (i.e., im0.png eh usada como referência).
        - Normalmente, o valor da disparidade será relativamente pequeno e a imagem gerada fica muito escura. Em seu relatório, use uma ferramenta que renderize a imagem usando uma paleta normalizada (tal como imagesc em Matlab e matshow em Matplotlib) e inclua um mapa de cores (colorbar) para informar o leitor sobre os valores mostrados.
        - Regiões onde não é possível calcular a disparidade devem ser indicadas, por exemplo, usando o valor 0. Porém, note que nesse caso há uma ambiguidade entre pontos no infinito, ou seja, com disparidade nula, versus pontos onde não foi possível calcular a disparidade. O que você faria para resolver essa ambiguidade?
    - **profundidade.png**: um mapa de profundidade, em que cada pixel armazena o valor da coordenada Z (ao longo do eixo da imagem), em milímetros, dos pontos observados.
        - Note que, como as imagens digitais normalmente armazenam somente valores de pixel em tom de cinza usando apenas 8 bits, será necessário normalizar os valores do mapa de profundidade, de forma que o maior valor seja 254. Seu relatório e seu arquivo read_me.txt devem informar como mapear dos valores da imagem de profundidade de volta para unidades em milímetros.
        - Regiões onde não foi possível calcular a profundidade podem ser indicadas pelo valor **255**.
4. Compare seus resultados de estimativa de disparidade (disparidade.pgm) com os valores verdade contido nas imagens de ground truth disp0-n.pgm.
    - Use isso para quantificar seu resultado de acordo com a principal métrica de Middlebury: _bad2.0, the precentage of bad pixels with disparity error > 2.0 pixels_. No seu relatório, destaque claramente (em negrito) os valores obtidos para cada uma das duas cenas.

### Requisito 2: câmeras estéreo com convergência
Assim como no Requisito 1, o objetivo deste requisito é gerar mapas de profundidade a partir de um par de imagens da mesma cena, porém dessa vez, as câmeras convergem para o objeto, ou seja, seus planos de projeção não são paralelos e $y_L$ é diferente de $y_R$.

Use [as imagens do Morpheus e os dados de calibração contidos no arquivos Morpheus*.txt fornecidos no Moodle](https://aprender3.unb.br/pluginfile.php/835807/mod_assign/introattachment/0/stereo_data.zip?forcedownload=1) (sob diretório FurukawaPonce). Esses arquivos foram obtidos da [3D Photography Dataset, de Furukawa e Ponce](https://www.cse.wustl.edu/~furukawa/research/mview/index.html) (infelizmente a página original está indisponível).

### Requisito 3: paralelepípedo
Usando cliques do mouse, meça as dimensões da que seria a menor caixa possível para colocar dentro dela o objeto observado (boneco de Morpheus sentado em um sofá), ou seja, informe a largura, altura e profundidade da caixa (assumindo que os pés do sofá devem se manter no plano do chão da caixa).

**Nota:** devido à natureza do método de calibração usado pelos autores da base de dados, os parâmetros de distância disponibilizadp foram dados em unidades de pixeis. Porém, dependendo do método usado, pode ser necessário converter a matriz de parâmetros intrínsecos para milimetros.

Para tal, é provavel que as seguintes informações sejam relevantes: foram usadas câmeras [Canon EOS 1D Mark II](https://en.wikipedia.org/wiki/Canon_EOS-1D_Mark_II), sua resolução original é de 3504 x 2336 pixels e seu sensor tem dimensões de 28.7 x 19.1 mm.

## Instruções para Elaboração do Relatório
O relatório deve demonstrar que a respectiva atividade foi realizada com sucesso e que os princípios subjacentes foram compreendidos.

Deverá conter as seguintes partes:

- **Identificação:** Possuir a indicação clara do título do projeto demonstrativo abordado, nome do(s) autor(es), e quando houver, número(s) de matrícula e e-mail(s).
- **Resumo:** Breve resumo do projeto e das suas conclusões.
- **Introdução:** Apresentar de forma clara, porém sucinta, os objetivos do projeto demonstrativo. Deve conter também uma breve explanação do conhecimento básico relacionado ao projeto e uma breve revisão bibliográfica relacionada ao problema. Utilize sempre fontes bibliográficas confiáveis (livros e artigos científicos), evitando utilizar única e exclusivamente fontes de baixa confiabilidade (Wikipedia, Stackoverflow,...).
- Crie uma seção **para cada requisito do projeto**, e em cada uma dessas seções, insira as seguintes subseções:
    - **Metodologia:** É dedicada a uma exposição dos métodos e procedimentos adotados no projeto demonstrativo.
        - Descreva os métodos usando notação matemática ou algoritmos. Não insira código fonte, seu relatório deve ser independente da linguagem de programação utilizada. Apresente uma exposição minuciosa do procedimento do projeto demonstrativo realmente adotado. Porém, deixe detalhes sobre o código, instalação e execução dos programas num arquivo **read_me.txt** ou **README.md**, que deve se localizar no diretório "raiz" da sua submissão.
    - **Resultados:** Nessa parte são apresentados os resultados das implementações efetuadas, na forma de tabelas e figuras, sem se esquecer de identificar em cada caso os parâmetros utilizados.
        - Rotule todos os eixos dos gráficos apresentados e use cores ou padrões que facilitem a visualização caso o relatório seja impresso em preto-e-branco.
- **Discussão e Conclusões:** A discussão visa comparar os resultados obtidos e os previstos pela teoria. Deve-se justificar eventuais discrepâncias observadas. As conclusões resumem a atividade e destacam os principais resultados e aplicações dos conceitos vistos. Se possível, proponha melhorias na implementação realizada.
    - Por exemplo, avalie as características das imagens de profundidade obtidas e a relação existente entre o tamanho da janela de busca (W) e a precisão esperada (por exemplo, observando os contornos dos objetos na cena).
    - Discuta as vantagens e desvantagens da estratégia usada para decidir se uma região é ou não encontrada na outra imagem. Qual é o efeito de regiões uniformes (ex. paredes lisas) nos seus resultados? Se você sabe de um algoritmo melhor (mesmo que não teve tempo de implementá-lo), por favor discuta-o.
- **Bibliografia:** Citar as fontes consultadas, respeitando as regras de apresentação de bibliografia (autor, título, editora, edição, ano, página de início e fim). Inclua o máximo possível de informações nas referências, por exemplo, inclua todos os autores e evite o uso de "et al." na lista de referências. No caso de citação de página da web, tente identificar seus autores e data da última atualização. Somente quando tais informações estão disponíveis, indique a data em que você visitou a página. No caso de uso de entradas do tipo _BIB geradas automaticamente (por exemplo, pelo Google Scholar)_, favor verificar os dados e os formatos, pois na maioria dos casos essas entradas possuem erros!

O relatório deverá ser confeccionado em editor eletrônico de textos com no máximo 7 (sete) páginas (sem contar as referencias bibliográficas), utilizando obrigatoriamente o padrão de formatação descrito no [arquivo de exemplo disponibilizado aqui, para processadores de texto LaTeX](https://aprender3.unb.br/mod/resource/view.php?id=256650). Não serão permitidos relatórios confeccionados em outro processador de texto, ou usando um modelo diferente do padrão LaTeX disponibilizado.

### Instruções para Submissão da atividade de Projeto Demonstrativo
Esta tarefa consiste na submissão de um arquivo único Zipado, contendo um arquivo PDF do relatório elaborado e também o código fonte desenvolvido, obrigatoriamente em C/C++ ou Python, e um arquivo com diretivas de compilação em Linux.

A estrutura de diretórios deverá ser:
```
PrimeiroNome_UltimoNome__PrimeiroNome_UltimoNome
      ├── read_me.txt (ou README.md)
      ├── PrimeiroNome_UltimoNome__PrimeiroNome_UltimoNome.pdf
      ├── /data
      │   ├── /Middlebury (diretorio para requisito 1)
      │   │   ├── /Jadeplant-perfect
      │   │   │   ├── disparidade.pgm
      │   │   │   └── profundidade.png
      │   │   └── /Playtable-perfect
      │   │       ├── disparidade.pgm
      │   │       └── profundidade.png
      │   └── /FurukawaPonce (diretório para os requisitos 2 e 3)
      ├── /relatorio
      └── /src
```

No arquivo de submissão, **não inclua os arquivos originais das bases de imagem no diretório "data"**. Inclua somente os arquivos de seus resultados para cada subdiretório (**disparidade.pgm** e **profundidade.png**).

Lembre que seu código deve ser implementado usando caminhos (paths) relativos!

## Critérios de Avaliação
- read_me.txt: 5%
- Requisito 1: 15%
- Requisito 2: 15%
- Requisito 3: 10%
- Tamanho correto: se o relatório tiver mais de 7 paginas (sem contar referencias), -50% da nota será descontada.
- Resumo: 5%
- Introdução: 5%
- Metodologia: 15% (5% para cada requisito)
- Resultados e análise: 15% (5% para cada requisito)
- Conclusões: 10%
- Bibliografia: 5%
- Bônus: ate 20% (sem ultrapassar 100%)
- **Importante:** no caso de trabalhos feitos em grupo, é obrigatória a inclusão de uma nota de rodapé na primeira página do relatório, indicando quais foram as contribuições de cada membro da equipe (seja específico). Se isso não for feito, haverá uma **penalização de 10% na nota final**.

Um **bônus** poderá ser dado nos casos abaixo.
- Se forem apresentados resultados em outras imagens da base de Middlebury no requisito 1. Ou melhor, se o código gerar todos os resultados prontos para [submissão e avaliação pelo site da base de imagens]( http://vision.middlebury.edu/stereo/submit3/), com imagens - em formato PFM (Portable Float Map).
- Se os resultados do requisito 1 forem relatados usando outras métricas de comparação (o [artigo dos autores da base de imagens](http://www.cs.middlebury.edu/~schar/papers/datasets-gcpr2014.pdf) sugere varias outras métricas).
- Se forem apresentados e avaliados múltiplos métodos para o requisito 2;
- Se forem mostrados resultados em outros pares de imagens da [3D Photography Dataset](https://www.cse.wustl.edu/~furukawa/research/mview/index.html) (caso elas sejam encontradas na web).
- Se forem apresentados resultados com pares de imagens adquiridas pelos próprios estudantes.
- **Bônus especial:** se for mostrado (na teoria e na prática) como resolver os problemas dos requisito 1 e 2 deste projeto sem se saber nada sobre os parâmetros de calibração das câmeras (notando que as medidas do objeto serão definidas como uma função de um fator de escala).

Esta atividade é **individual para alunos da pós-graduação**.
No caso de alunos de **graduação**, ela pode ser realizada em **grupos de até 3 alunos**.

Recomenda-se que alunos de pós-graduação façam o trabalho em inglês.

Caso seja detectado plágio, todos alunos envolvidos ficarão com mencão final SR.
