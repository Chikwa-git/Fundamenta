"""
Etapa 2 — Preparação dos dados
Projeto CS50AI - Análise de Saúde Financeira de Empresas (B3)

Roda com:
    python3 preparar_dados.py

Dependências:
    pip install pandas numpy
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────
# Bancos têm contabilidade diferente —
# serão tratados num modelo separado no futuro
# ─────────────────────────────────────────
BANCOS = ["ITUB4.SA", "BBDC4.SA", "SANB11.SA"]

# ─────────────────────────────────────────
# Carregar CSVs gerados na Etapa 1
# ─────────────────────────────────────────
print("=" * 60)
print("ETAPA 2 — PREPARAÇÃO DOS DADOS")
print("=" * 60)

print("\n📂 Carregando arquivos...")
bal = pd.read_csv("dados/balancos_trimestrais.csv")
ind = pd.read_csv("dados/indicadores_atuais.csv")
pre = pd.read_csv("dados/historico_precos.csv")
print(f"   Balanços:     {len(bal)} linhas")
print(f"   Indicadores:  {len(ind)} linhas")
print(f"   Preços:       {len(pre)} linhas")

# ─────────────────────────────────────────
# PASSO 1 — LIMPEZA
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("PASSO 1 — LIMPEZA")
print("─" * 60)

# Remover bancos dos três DataFrames
bal = bal[~bal["ticker"].isin(BANCOS)].copy()
ind = ind[~ind["ticker"].isin(BANCOS)].copy()
pre = pre[~pre["ticker"].isin(BANCOS)].copy()
print(f"\n✅ Bancos removidos. Empresas restantes: {bal['ticker'].nunique()}")

# Remover linhas do balanço onde todos os valores financeiros são nulos
# (trimestres que o yfinance retornou vazios)
colunas_financeiras = [
    "total_ativos", "total_passivos", "patrimonio_liquido",
    "caixa", "receita", "lucro_liquido"
]
antes = len(bal)
bal = bal.dropna(subset=colunas_financeiras, how="all")
depois = len(bal)
print(f"✅ Trimestres vazios removidos: {antes - depois} linhas removidas")
print(f"   Balanços restantes: {depois} linhas")

# Ordenar por empresa e trimestre (importante para calcular variações)
bal["trimestre"] = pd.to_datetime(bal["trimestre"])
bal = bal.sort_values(["ticker", "trimestre"]).reset_index(drop=True)

pre["trimestre"] = pd.to_datetime(pre["trimestre"])
pre = pre.sort_values(["ticker", "trimestre"]).reset_index(drop=True)

print("✅ Dados ordenados por empresa e trimestre")

# ─────────────────────────────────────────
# PASSO 2 — ENRIQUECIMENTO
# Calcular variações entre trimestres
# O modelo precisa ver TENDÊNCIA, não só snapshot
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("PASSO 2 — ENRIQUECIMENTO (variações trimestrais)")
print("─" * 60)

def variacao_pct(serie):
    """Calcula variação percentual em relação ao trimestre anterior."""
    return serie.pct_change() * 100

# Calcular variações dentro de cada empresa (groupby garante que
# a variação não "vaze" entre empresas diferentes)
for coluna in ["total_ativos", "patrimonio_liquido", "caixa",
               "receita", "lucro_liquido", "ebitda"]:
    if coluna in bal.columns:
        nome_variacao = f"var_{coluna}_pct"
        bal[nome_variacao] = (
            bal.groupby("ticker")[coluna]
            .transform(variacao_pct)
        )
        print(f"   ✅ {nome_variacao}")

# Calcular margens diretamente do balanço (quando disponível)
# Isso é mais confiável que confiar só nos indicadores do snapshot
bal["margem_liquida_calc"] = np.where(
    bal["receita"] > 0,
    (bal["lucro_liquido"] / bal["receita"]) * 100,
    np.nan
)

bal["margem_ebitda_calc"] = np.where(
    (bal["ebitda"].notna()) & (bal["receita"] > 0),
    (bal["ebitda"] / bal["receita"]) * 100,
    np.nan
)

bal["alavancagem"] = np.where(
    bal["patrimonio_liquido"] > 0,
    bal["total_passivos"] / bal["patrimonio_liquido"],
    np.nan
)

print("   ✅ margem_liquida_calc")
print("   ✅ margem_ebitda_calc")
print("   ✅ alavancagem (passivos / patrimônio)")

# ─────────────────────────────────────────
# PASSO 2.5 — TENDÊNCIA DE LUCRO EM JANELA
#
# Em vez de olhar só a variação de 1 trimestre,
# calculamos quantos dos últimos 3 trimestres
# tiveram lucro positivo — isso evita punir
# empresas por um trimestre ruim isolado
# ─────────────────────────────────────────
def trimestres_lucrativos(serie):
    """
    Para cada trimestre, conta quantos dos últimos 3
    (incluindo o atual) tiveram lucro positivo.
    """
    return (
        serie.rolling(window=3, min_periods=1)
        .apply(lambda x: (x > 0).sum())
    )

bal["trimestres_lucro_positivo"] = (
    bal.groupby("ticker")["lucro_liquido"]
    .transform(trimestres_lucrativos)
)
print("   ✅ trimestres_lucro_positivo (janela de 3 trimestres)")

# ─────────────────────────────────────────
# PASSO 3 — CRIAR O LABEL
# Classificação de saúde financeira por trimestre
#
# Lógica revisada com 5 sinais:
#
#   Sinal 1 — Lucratividade atual (peso 2)
#       Empresa está gerando lucro agora?
#       Prejuízo é penalidade forte.
#
#   Sinal 2 — Consistência de lucro (peso 2)
#       Quantos dos últimos 3 trimestres foram lucrativos?
#       Evita punir um trimestre ruim isolado.
#
#   Sinal 3 — Alavancagem (peso 1)
#       Nível de dívida em relação ao patrimônio.
#       Alavancagem alta só penaliza se combinada com
#       outros sinais negativos.
#
#   Sinal 4 — Margem líquida (peso 1)
#       Quanto sobra de cada R$ de receita.
#       Margem positiva indica operação eficiente.
#
#   Sinal 5 — Tendência do caixa (peso 1)
#       Caixa crescendo é sinal de saúde operacional.
#       Queda forte do caixa é alerta.
#
# Resultado:
#   2 = Saudável  (operação sólida e consistente)
#   1 = Atenção   (sinais mistos ou momento difícil)
#   0 = Risco     (prejuízo recorrente ou estrutura frágil)
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("PASSO 3 — CRIANDO LABEL DE SAÚDE FINANCEIRA")
print("─" * 60)

def classificar_saude(row):
    pontos = 0
    penalidades = 0

    # ── Sinal 1: Lucratividade atual (peso duplo) ──
    # É o sinal mais importante — o estado presente da empresa
    if pd.notna(row["lucro_liquido"]):
        if row["lucro_liquido"] > 0:
            pontos += 2       # lucrativa agora: bom sinal
        else:
            penalidades += 2  # prejuízo agora: sinal negativo forte

    # ── Sinal 2: Consistência de lucro (peso duplo) ──
    # Quantos dos últimos 3 trimestres foram lucrativos?
    # Isso diferencia um trimestre ruim pontual de deterioração real
    if pd.notna(row["trimestres_lucro_positivo"]):
        consistencia = row["trimestres_lucro_positivo"]
        if consistencia >= 3:
            pontos += 2      # lucro nos 3 últimos trimestres
        elif consistencia == 2:
            pontos += 1      # 2 de 3: razoável
        elif consistencia == 0:
            penalidades += 2 # prejuízo nos 3 últimos: risco real

    # ── Sinal 3: Alavancagem ──
    # Dívida alta só é problemática quando combinada com
    # lucratividade fraca — empresas grandes operam alavancadas
    if pd.notna(row["alavancagem"]):
        if row["alavancagem"] < 1.5:
            pontos += 1      # dívida bem controlada
        elif row["alavancagem"] > 8:
            penalidades += 1 # dívida muito alta (antes era 5, agora 8)

    # ── Sinal 4: Margem líquida ──
    # Indica eficiência operacional
    if pd.notna(row["margem_liquida_calc"]):
        if row["margem_liquida_calc"] > 10:
            pontos += 1      # margem saudável
        elif row["margem_liquida_calc"] < -5:
            penalidades += 1 # margem negativa relevante

    # ── Sinal 5: Tendência do caixa ──
    # Caixa é o "oxigênio" da empresa
    if pd.notna(row["var_caixa_pct"]):
        if row["var_caixa_pct"] > 10:
            pontos += 1      # caixa crescendo bem
        elif row["var_caixa_pct"] < -40:
            penalidades += 1 # queda severa de caixa

    # ── Classificação final ──
    score = pontos - penalidades

    if score >= 5:
        return 2  # Saudável
    elif score >= 2:
        return 1  # Atenção
    else:
        return 0  # Risco

bal["saude"] = bal.apply(classificar_saude, axis=1)

# Mapear para texto também (útil para visualizar)
mapa_saude = {2: "Saudável", 1: "Atenção", 0: "Risco"}
bal["saude_texto"] = bal["saude"].map(mapa_saude)

# Mostrar distribuição do label
print("\nDistribuição das classificações:")
distribuicao = bal["saude_texto"].value_counts()
for label, count in distribuicao.items():
    pct = count / len(bal) * 100
    print(f"   {label}: {count} trimestres ({pct:.1f}%)")

# Mostrar exemplos por empresa
print("\nExemplo — últimos 2 trimestres por empresa:")
ultimos = (
    bal.groupby("ticker")
    .tail(2)[["empresa", "trimestre", "lucro_liquido",
              "alavancagem", "saude_texto"]]
)
ultimos["trimestre"] = ultimos["trimestre"].dt.strftime("%Y-%m")
ultimos["lucro_liquido"] = (ultimos["lucro_liquido"] / 1e9).round(2)
ultimos = ultimos.rename(columns={"lucro_liquido": "lucro(R$B)"})
print(ultimos.to_string(index=False))

# ─────────────────────────────────────────
# Salvar dataset final
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("SALVANDO DATASET FINAL")
print("─" * 60)

os.makedirs("dados", exist_ok=True)
bal.to_csv("dados/dataset_preparado.csv", index=False)
print(f"\n✅ dados/dataset_preparado.csv")
print(f"   {len(bal)} linhas | {len(bal.columns)} colunas")
print(f"   Empresas: {bal['ticker'].nunique()}")
print(f"   Colunas: {list(bal.columns)}")

print("\n" + "=" * 60)
print("Etapa 2 concluída.")
print("Próximo passo: Etapa 3 — Treinar o modelo.")
print("=" * 60)
