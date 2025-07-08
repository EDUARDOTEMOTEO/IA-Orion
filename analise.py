import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

METAS = {
    'Alimentação': 0.15,
    'Moradia': 0.30,
    'Transporte': 0.10,
    'Lazer': 0.05,
    'Saúde': 0.10,
    'Investimentos': 0.20,
    'Outros': 0.10
}

def analisar_financas_avancado(df: pd.DataFrame) -> str:
    total = df['Valor'].sum()
    if total == 0:
        return "Não há gastos registrados para análise."

    analise = [f"Seu gasto total foi R$ {total:.2f}.\n"]

    agrupado = df.groupby('Categoria')['Valor'].sum()
    percentuais = agrupado / total

    for cat, meta in METAS.items():
        gasto = agrupado.get(cat, 0.0)
        pct = percentuais.get(cat, 0.0)
        analise.append(f"- {cat}: R$ {gasto:.2f} ({pct:.1%} do total) | Meta recomendada: {meta:.1%}")

        if pct > meta * 1.2:
            analise.append(f"  -> Você está gastando muito em {cat}. Considere reduzir esses custos.")
        elif pct < meta * 0.8 and gasto > 0:
            analise.append(f"  -> Seu gasto em {cat} está abaixo do recomendado, talvez possa investir mais nessa área.")

    investimento_pct = percentuais.get('Investimentos', 0)
    if investimento_pct < 0.15:
        analise.append("\nSua alocação em investimentos está abaixo do ideal. Considere aumentar para garantir seu futuro financeiro.")
    else:
        analise.append("\nVocê está investindo uma boa parte dos seus recursos. Continue assim!")

    return "\n".join(analise)

def gerar_grafico(df: pd.DataFrame) -> str:
    agrupado = df.groupby('Categoria')['Valor'].sum()
    categorias = agrupado.index.tolist()
    valores = agrupado.values.tolist()

    fig = go.Figure(data=[go.Pie(labels=categorias, values=valores, hole=0.3)])
    fig.update_layout(title_text="Distribuição dos Gastos por Categoria")
    return pio.to_html(fig, full_html=False)