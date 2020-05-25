# exerciciosreinaldo
Exercicios da lista da matéria de Matemática Computacional do professor Reinaldo Rosa.

# resolucaoexercicios.py

Arquivo com a maioria das resoluções de exercícios da lista, ele pede um input do usuário e então faz tudo que é pedido pra aquele tipo de distribuição de uma vez, gerando gráficos que servem como resposta.

Utiliza o mfdfa.py e o statsfuncs.py como módulos e utiliza os arquivos de dados covidbrasil.dat, sol3ghz.dat e surftemp504.txt como input para resolução de exercícios.

# Análises feitas pra os tipos de distribuição:

Listando as análises feitas pra cada série temporal, na ordem que foram feitas

## GRNG:

- Gera um conjunto de dados para N=2^6, 2^7, 2^8, ..., 2^13 do tipo de série GRNG, 10 para cada N, totalizando em 80;
- Plot da série ultima série temporal gerada para cada N, junto com o cálculo e plot do Detrended Fluctuation Analysis (DFA) e cálculo e plot do Power Spectrum Density (PSD);
- Faz o clustering via K-Means;
- Faz o mapa de Cullen e Frey;
- Faz os espaços EPSB-K-means e EDF-K-means.

## Colored Noise:

- Gera um conjunto de dados para N=8192, para os ruídos de cor Branca, Rosa e Vermelha, gerando 20 séries pra cada, totalizando 60;
- Faz um histograma da série e um ajuste pra distribuição normal obtida no próprio histograma;
- Plot da série ultima série temporal gerada para cada cor, junto com o cálculo e plot do Detrended Fluctuation Analysis (DFA) e cálculo e plot do Power Spectrum Density (PSD);
- Faz o clustering via K-Means;
- Faz o mapa de Cullen e Frey;
- Faz os espaços EPSB-K-means e EDF-K-means.

## pmodel:
- Gera um conjunto de dados para N=8192, para 30 valores de p da classe Endógena e 30 valores de p da classe Exógena, 60 séries geradas no total;
- Plot da série ultima série temporal gerada para cada p, junto com o cálculo e plot do Detrended Fluctuation Analysis (DFA) e cálculo e plot do Power Spectrum Density (PSD);
- Faz o clustering via K-Means;
- Faz o mapa de Cullen e Frey;
- Faz os espaços EPSB-K-means e EDF-K-means.

## Chaos Noise:
- Gera um conjunto de dados para N=8192, para 30 valores de rho entre 3.85 e 4 para a classe de série Logística e 30 valores de a entre 1.35 e 4 e b entre 0.21 e 0.29, totalizando 60 séries;
- Plot da série ultima série temporal gerada para cada valor, junto com o cálculo e plot do Detrended Fluctuation Analysis (DFA) e cálculo e plot do Power Spectrum Density (PSD);
- Faz o clustering via K-Means, utilizando o Skewness, Curtose e Variancia de cada série temporal gerada;
- Faz o mapa de Cullen e Frey utilizando o Skewness² e Curtose;
- Faz os espaços EPSB-K-means e EDF-K-means .

## Arquivos
- A fazer, o exercício propõe que façamos um clustering via k-means para todos os dados gerados a partir dos sinais anteriores (GRNG, Colored Noise, pmodel e Chaos Noise) e com eles tentar classificar em quais desses tipos de ruídos os sinais vindo de arquivo devem se encaixar melhor, fazendo uma identificação por cores em cada um deles.
# ex63.py

Resolução dos exercícios 6.3 e 10.2 da lista, utiliza os módulos statsfuncs.py, mfdfa.py e waipy.py, e como input um arquivo completo do https://ourworldindata.org/coronavirus de "Daily confirmed cases", executa uma análise de SOC, de K-Means e Wavelets, tem um arquivo de exemplo no repositório de "daily-cases-covid-19.csv", obtido no dia 21/05/2020.

# ex9.py

Resolução do exercício 9 da lista, utiliza o arquivo waipy.py. Faz uma análise do pmodel de SOC e de Wavelets, faz o SOC para 50 séries exógenas e endógenas diferentes, totalizando 100 séries. Depois disso faz uma análise de wavelets pra última série gerada.

# mfdfa.py, statsfuncs.py e waipy.py

Arquivo de módulo com funções usado em outros programas.

# sol3ghz.dat, surftemp504.txt e covidbrasil.dat

Arquivos de dados utilizados no exercício 6.2.


# daily-cases-covid-19.csv

Arquivo de dados obtido em https://ourworldindata.org/coronavirus no dia 21/05/2020, utilizado para o exercício 6.3.
