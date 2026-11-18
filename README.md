# Perceptron — Classificador Binário em Python

Implementação do algoritmo **Perceptron**, um modelo básico de Rede Neural Artificial utilizado em problemas de classificação binária. O projeto foi desenvolvido em Python como atividade prática da disciplina de Inteligência Artificial, demonstrando o processo de aprendizagem supervisionada e classificação de novas amostras.

## Objetivo

Demonstrar, na prática, como o Perceptron aprende a classificar dados em duas categorias a partir de exemplos previamente rotulados.

A implementação contempla duas etapas principais:

- **Treinamento:** o algoritmo recebe amostras conhecidas, realiza previsões, compara os resultados com as classes esperadas e ajusta seus pesos quando identifica erros.
- **Operação:** após o treinamento, utiliza os pesos aprendidos para classificar novas amostras informadas pelo usuário.

## Tecnologias utilizadas

- **Python:** linguagem utilizada no desenvolvimento.
- **NumPy:** operações matemáticas, manipulação de vetores e cálculos dos pesos.
- **Rich:** formatação visual do terminal, com tabelas, cores e painéis informativos.


## Funcionamento do algoritmo

O Perceptron recebe três características de entrada, associadas a quatro pesos, incluindo o peso correspondente ao bias.

A classificação ocorre por meio de uma soma ponderada:

$$
u = w^T \cdot x
$$

Em seguida, aplica-se a função de ativação degrau bipolar:

$$
y =
\begin{cases}
+1, & \text{se } u \geq 0 \\
-1, & \text{se } u < 0
\end{cases}
$$

Quando a classificação produzida é diferente da desejada, os pesos são atualizados pela regra de aprendizagem:

$$
w \leftarrow w + \eta(d-y)x
$$

Onde:

- $w$: vetor de pesos;
- $\eta$: taxa de aprendizagem;
- $d$: saída desejada;
- $y$: saída produzida pelo Perceptron;
- $x$: vetor de entradas.

O treinamento é repetido até que todas as amostras sejam classificadas corretamente em uma época ou até atingir o limite máximo de épocas definido no código.


## Estrutura do projeto

```text
perceptron-classificador-binario/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Instalação e configuração

**1. Pré-requisitos**

Ter o Python instalado na máquina.

**2. Clonar o repositório**

```bash
git clone URL_DO_REPOSITORIO
```

Acesse a pasta do projeto:

```bash
cd perceptron-classificador-binario
```

**3. Criar o ambiente virtual**

No Windows:

```bash
python -m venv .venv
```

Ativar o ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```

**4. Instalar as dependências**

```bash
pip install -r requirements.txt
```

O arquivo `requirements.txt` deve conter:

```text
numpy
rich
```

## Como executar

Com o ambiente virtual ativado, execute:

```bash
python main.py
```

O programa realizará automaticamente o treinamento antes de iniciar o simulador interativo.

## Manual de utilização e testes

### 1. Verificação do treinamento

Na primeira execução, o programa apresenta:

- Quantidade de épocas realizadas;
- Taxa de aprendizagem;
- Pesos finais ajustados;
- Comparação entre as classificações esperadas e obtidas;
- Quantidade e percentual de acertos no conjunto de treinamento.

Com os parâmetros e dados originais do projeto, o treinamento converge em **104 épocas**, classificando corretamente as quatro amostras utilizadas.

### 2. Simulação interativa

Após o treinamento, o programa disponibiliza um simulador no terminal.

O usuário deve informar três valores numéricos correspondentes às entradas `x1`, `x2` e `x3`.

São aceitos valores entre 0 e 1, utilizando ponto ou vírgula como separador decimal.

O valor do bias (`-1`) é inserido automaticamente pelo programa.

Após informar as três entradas, o sistema apresenta:

- Valores informados;
- Quantidade de épocas utilizadas no treinamento;
- Soma ponderada;
- Saída produzida pelo Perceptron;
- Classe identificada.

### 3. Exemplos de testes

**Teste 1 — Classificação na Classe B**

Entradas:

```text
x1: 0.2
x2: 0.3
x3: 0.4
```

Resultado esperado:

```text
Soma ponderada: 0.3461
Classificação: Classe B (+1)
```

**Teste 2 — Classificação na Classe A**

Entradas:

```text
x1: 0.1
x2: 0.9
x3: 0.1
```

Resultado esperado:

```text
Soma ponderada: -0.9015
Classificação: Classe A (-1)
```

Os resultados acima correspondem à configuração original do projeto, utilizando a taxa de aprendizagem `0.1` e a semente aleatória `42`.

### 4. Encerramento da simulação

Para finalizar a execução, digite:

```text
sair
```

O comando também pode ser informado como `s`.

### 5. Testando outros conjuntos de treinamento

Para experimentar diferentes padrões de aprendizagem, é possível modificar:

- A matriz `X`, que contém as características das amostras;
- O vetor `d`, que contém suas classificações desejadas;
- A taxa de aprendizagem `eta`.

Cada alteração nos dados de treinamento exige uma nova execução do programa, pois os pesos serão inicializados e aprendidos novamente.

**Importante:** cada amostra em `X` deve possuir uma classificação correspondente em `d`. Para que o Perceptron clássico consiga convergir, os dados devem ser linearmente separáveis.

## Observações sobre o aprendizado

O percentual de acertos apresentado inicialmente corresponde às próprias amostras utilizadas no treinamento. Portanto, não representa uma avaliação independente da capacidade de generalização do modelo.

Na fase de operação, os pesos permanecem fixos. Novas entradas são classificadas utilizando os parâmetros aprendidos anteriormente, sem realizar um novo treinamento.

## Finalidade acadêmica

Este projeto tem finalidade didática e busca demonstrar os fundamentos do aprendizado supervisionado, o funcionamento de um neurônio artificial, a atualização de pesos e a aplicação de uma função de ativação em um problema de classificação binária.