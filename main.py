
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# 1. Dados de treinamento
X = np.array([
    [-1, 0.1, 0.4, 0.7],
    [-1, 0.3, 0.7, 0.2],
    [-1, 0.6, 0.9, 0.8],
    [-1, 0.5, 0.7, 0.1]
])

d = np.array([1, -1, -1, 1])

# 2. Configurações
eta = 0.1
rng = np.random.default_rng(42)
w = rng.uniform(-0.1, 0.1, size=4)

# 3. Treinamento
for epoca in range(1, 10001):
    erro = False

    for x, desejado in zip(X, d):
        u = np.dot(w, x)
        y = 1 if u >= 0 else -1

        if y != desejado:
            w = w + eta * (desejado - y) * x
            erro = True

    if not erro:
        break
else:
    raise RuntimeError("O treinamento não convergiu.")

# 4. Apresentação dos resultados
console.print(Panel(
    "[bold cyan]ALGORITMO PERCEPTRON[/bold cyan]\n"
    "Classificação Binária com Aprendizado Supervisionado",
    border_style="cyan"
))

console.print(f"\n[bold]Treinamento concluído![/bold]")
console.print(f"Épocas executadas: [cyan]{epoca}[/cyan]")
console.print(f"Taxa de aprendizagem: {eta}")
console.print(f"Pesos finais: {np.round(w, 4)}")

# 5. Resultados das amostras
tabela = Table(title="CLASSIFICAÇÃO DAS AMOSTRAS")

tabela.add_column("Amostra", justify="center")
tabela.add_column("Esperado", justify="center")
tabela.add_column("Obtido", justify="center")
tabela.add_column("Resultado", justify="center")

acertos = 0

for i, (x, desejado) in enumerate(zip(X, d), start=1):
    u = np.dot(w, x)
    y = 1 if u >= 0 else -1

    if y == desejado:
        status = "[green]Correto ✓[/green]"
        acertos += 1
    else:
        status = "[red]Incorreto ✗[/red]"

    tabela.add_row(
        str(i),
        str(desejado),
        str(y),
        status
    )

console.print()
console.print(tabela)

console.print(
    f"\n[bold green]Acertos: {acertos}/{len(X)} "
    f"({acertos / len(X) * 100:.0f}%)[/bold green]"
)


# 6. SIMULAÇÃO INTERATIVA

console.print()
console.print(Panel(
    "[bold cyan]SIMULADOR PERCEPTRON[/bold cyan]\n"
    "Digite os valores de entrada para classificar novas amostras.\n"
    "Digite 'sair' para encerrar a simulação.",
    border_style="cyan"
))

while True:

    console.print("\n[bold]Informe as características da amostra:[/bold]")

    try:
        entrada = console.input("[cyan]Valor de x1 (ou sair): [/]").strip()

        if entrada.lower() in ("sair", "s"):
            break

        x1 = float(entrada.replace(",", "."))
        x2 = float(console.input("[cyan]Valor de x2: [/]").replace(",", "."))
        x3 = float(console.input("[cyan]Valor de x3: [/]").replace(",", "."))

        # Validação das entradas
        if not all(0 <= valor <= 1 for valor in [x1, x2, x3]):
            console.print(
                "[red]Erro: informe valores entre 0 e 1.[/red]"
            )
            continue

        # O bias é inserido automaticamente
        nova_amostra = np.array([-1, x1, x2, x3])

        # Cálculo e classificação
        u = np.dot(w, nova_amostra)
        resultado = 1 if u >= 0 else -1

        classe = "Classe B (+1)" if resultado == 1 else "Classe A (-1)"

        cor = "green" if resultado == 1 else "yellow"

        # Apresentação do resultado
        console.print(Panel(
            f"[bold]Entradas:[/bold] {x1:.2f}, {x2:.2f}, {x3:.2f}\n"
            f"[bold]Épocas de treinamento:[/bold] {epoca}\n"
            f"[bold]Soma ponderada:[/bold] {u:.4f}\n"
            f"[bold]Saída do Perceptron:[/bold] {resultado}\n\n"
            f"[bold {cor}]CLASSIFICAÇÃO: {classe}[/bold {cor}]",
            title="RESULTADO DA CLASSIFICAÇÃO",
            border_style=cor
        ))

    except ValueError:
        console.print(
            "[red]Entrada inválida. Digite valores numéricos.[/red]"
        )

console.print("\n[bold cyan]Simulação encerrada![/bold cyan]")

