# Projeto 1 - Visão Estério

Considerando as [especificações do projeto](/ref/descrião.md), o projeto foi divido em três partes:

- Desafio 1: cálculo de mapas de disparidade e profundidade para imagens correspondentes retificadas
- Desafio 2: cálculo de mapas de disparidade para imagens correspondentes
- Desafio 3: cálculo volumétrico para as imagens utilizadas no Desafio 2

Mas, o Desafio 3 não foi implementado.

## Ajuste à hierarquia das pastas
Para o Desafio 2, foi necessário adaptar a estrutura de pastas de:
```
- FurukawaPonce
    - Morpheus*
    - warrior*
```

para:
```
- FurukawaPonce/
    - Morpheus/
        - Morpheus*
    - warrior/
        - warrior*
```

com o intuito de manter a estrutura pesquisa para o Desafio 1.

## Dependências e execução
Sendo desenvolvido em Python 3.7.2 é recomendado o uso de um ambiente virtual.
```
python3.7 -m venv .proj1
```

E instalar suas dependências:
```
# source .proj1/bin/activate
pip install -r requirements.txt
```

Com isso, já é possível executar os desafios desenvolvidos:
```
# source .proj1/bin/activate
python src/challenge_1.py
python src/challenge_2.py
```
