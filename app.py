"""
Etapa 4 — Interface Flask
Projeto CS50AI - Análise de Saúde Financeira de Empresas (B3)

Roda com:
    python3 app.py

Acesse: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# ─────────────────────────────────────────
# Carregar modelo e arquivos auxiliares
# ─────────────────────────────────────────
with open("modelo/modelo_saude.pkl", "rb") as f:
    modelo = pickle.load(f)

with open("modelo/features.pkl", "rb") as f:
    features = pickle.load(f)

with open("modelo/medianas.pkl", "rb") as f:
    medianas = pickle.load(f)

# Nomes legíveis para exibir na interface
NOMES_FEATURES = {
    "margem_liquida_calc":          "Margem Líquida",
    "lucro_liquido":                "Lucro Líquido",
    "trimestres_lucro_positivo":    "Consistência de Lucro",
    "var_caixa_pct":                "Variação do Caixa",
    "var_patrimonio_liquido_pct":   "Variação do Patrimônio",
    "var_total_ativos_pct":         "Variação dos Ativos",
    "margem_ebitda_calc":           "Margem EBITDA",
    "var_lucro_liquido_pct":        "Variação do Lucro",
    "alavancagem":                  "Alavancagem",
    "ebitda":                       "EBITDA",
    "var_ebitda_pct":               "Variação do EBITDA",
    "receita":                      "Receita",
    "total_ativos":                 "Total de Ativos",
    "caixa":                        "Caixa",
    "var_receita_pct":              "Variação da Receita",
    "patrimonio_liquido":           "Patrimônio Líquido",
    "divida_lp":                    "Dívida de Longo Prazo",
}


def buscar_dados_empresa(ticker):
    """
    Busca os dados financeiros mais recentes da empresa via yfinance.
    Retorna um dicionário com as features que o modelo espera.
    """
    if not ticker.endswith(".SA"):
        ticker = ticker.upper() + ".SA"

    ativo = yf.Ticker(ticker)
    info = ativo.info

    # Verificar se o ticker é válido
    if not info or info.get("regularMarketPrice") is None:
        return None, None, "Ticker não encontrado. Verifique se o código está correto."

    nome_empresa = info.get("longName") or info.get("shortName") or ticker
    setor = info.get("sector", "Não informado")

    # ── Verificar setor financeiro ──
    # Bancos e financeiras têm contabilidade diferente e serão
    # suportados por um modelo separado em versão futura
    SETORES_FINANCEIROS = ["Financial Services", "Banks", "Insurance"]
    if setor in SETORES_FINANCEIROS:
        return None, None, (
            f"{nome_empresa} é do setor financeiro ({setor}). "
            "O modelo atual foi treinado com empresas não-financeiras. "
            "O suporte a bancos e financeiras está previsto para uma próxima versão."
        )

    # ── Buscar balanço trimestral ──
    bp = ativo.quarterly_balance_sheet
    dre = ativo.quarterly_financials

    if bp.empty or dre.empty:
        return None, None, "Dados financeiros não disponíveis para esta empresa."

    # Pegar o trimestre mais recente
    trimestre = bp.columns[0]

    def get_valor(df, chave):
        """Extrai valor de um DataFrame de balanço com segurança."""
        try:
            if chave in df.index and trimestre in df.columns:
                val = df.loc[chave, trimestre]
                return float(val) if pd.notna(val) else None
        except Exception:
            pass
        return None

    # Valores do trimestre atual
    total_ativos      = get_valor(bp, "Total Assets")
    total_passivos    = get_valor(bp, "Total Liabilities Net Minority Interest")
    patrimonio        = get_valor(bp, "Stockholders Equity")
    caixa_atual       = get_valor(bp, "Cash And Cash Equivalents")
    divida_lp         = get_valor(bp, "Long Term Debt")
    receita           = get_valor(dre, "Total Revenue")
    lucro_liquido     = get_valor(dre, "Net Income")
    ebitda            = get_valor(dre, "EBITDA")

    # Valores do trimestre anterior (para calcular variações)
    if len(bp.columns) > 1:
        tri_ant = bp.columns[1]

        def get_anterior(df, chave):
            try:
                if chave in df.index and tri_ant in df.columns:
                    val = df.loc[chave, tri_ant]
                    return float(val) if pd.notna(val) else None
            except Exception:
                pass
            return None

        total_ativos_ant  = get_anterior(bp, "Total Assets")
        patrimonio_ant    = get_anterior(bp, "Stockholders Equity")
        caixa_ant         = get_anterior(bp, "Cash And Cash Equivalents")
        receita_ant       = get_anterior(dre, "Total Revenue")
        lucro_ant         = get_anterior(dre, "Net Income")
        ebitda_ant        = get_anterior(dre, "EBITDA")
    else:
        total_ativos_ant = patrimonio_ant = caixa_ant = None
        receita_ant = lucro_ant = ebitda_ant = None

    def variacao(atual, anterior):
        """Calcula variação percentual com segurança."""
        if atual is not None and anterior is not None and anterior != 0:
            return ((atual - anterior) / abs(anterior)) * 100
        return None

    # Calcular indicadores derivados
    margem_liquida = (lucro_liquido / receita * 100
                      if lucro_liquido and receita and receita > 0 else None)
    margem_ebitda  = (ebitda / receita * 100
                      if ebitda and receita and receita > 0 else None)
    alavancagem    = (total_passivos / patrimonio
                      if total_passivos and patrimonio and patrimonio > 0 else None)

    # Consistência de lucro: buscar últimos 3 trimestres da DRE
    try:
        lucros_recentes = dre.loc["Net Income"].iloc[:3]
        trimestres_positivos = float((lucros_recentes > 0).sum())
    except Exception:
        trimestres_positivos = None

    # Montar dicionário de features
    dados = {
        "total_ativos":                 total_ativos,
        "patrimonio_liquido":           patrimonio,
        "caixa":                        caixa_atual,
        "receita":                      receita,
        "lucro_liquido":                lucro_liquido,
        "ebitda":                       ebitda,
        "divida_lp":                    divida_lp,
        "var_total_ativos_pct":         variacao(total_ativos, total_ativos_ant),
        "var_patrimonio_liquido_pct":   variacao(patrimonio, patrimonio_ant),
        "var_caixa_pct":                variacao(caixa_atual, caixa_ant),
        "var_receita_pct":              variacao(receita, receita_ant),
        "var_lucro_liquido_pct":        variacao(lucro_liquido, lucro_ant),
        "var_ebitda_pct":               variacao(ebitda, ebitda_ant),
        "margem_liquida_calc":          margem_liquida,
        "margem_ebitda_calc":           margem_ebitda,
        "alavancagem":                  alavancagem,
        "trimestres_lucro_positivo":    trimestres_positivos,
    }

    # Informações extras para exibir no dashboard (não entram no modelo)
    info_exibicao = {
        "nome":         nome_empresa,
        "ticker":       ticker,
        "setor":        setor,
        "trimestre":    trimestre.strftime("%b/%Y") if hasattr(trimestre, "strftime") else str(trimestre)[:7],
        "lucro_bi":     round(lucro_liquido / 1e9, 2) if lucro_liquido else None,
        "receita_bi":   round(receita / 1e9, 2) if receita else None,
        "caixa_bi":     round(caixa_atual / 1e9, 2) if caixa_atual else None,
        "margem_pct":   round(margem_liquida, 1) if margem_liquida else None,
        "alavancagem":  round(alavancagem, 2) if alavancagem else None,
    }

    return dados, info_exibicao, None


def classificar_empresa(dados):
    """
    Passa os dados pelo modelo e retorna a classificação,
    as probabilidades e a importância das features.
    """
    # Montar vetor de features na ordem certa
    X = pd.DataFrame([dados])[features]

    # Preencher nulos com medianas do treino
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(medianas.get(col, 0))

    # Predição
    predicao = modelo.predict(X)[0]
    probabilidades = modelo.predict_proba(X)[0]

    mapa = {0: "Risco", 1: "Atenção", 2: "Saudável"}
    classificacao = mapa[predicao]

    # Importância das features do modelo
    rf = modelo.named_steps["rf"]
    importancias = dict(zip(features, rf.feature_importances_))

    # Top 5 features mais importantes
    top_features = sorted(importancias.items(), key=lambda x: x[1], reverse=True)[:5]
    top_features_formatadas = [
        {
            "nome": NOMES_FEATURES.get(f, f),
            "importancia": round(v * 100, 1),
            "valor": dados.get(f)
        }
        for f, v in top_features
    ]

    return {
        "classificacao": classificacao,
        "prob_risco":    round(probabilidades[0] * 100),
        "prob_atencao":  round(probabilidades[1] * 100),
        "prob_saudavel": round(probabilidades[2] * 100),
        "top_features":  top_features_formatadas,
    }


# ─────────────────────────────────────────
# Rotas
# ─────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analisar", methods=["POST"])
def analisar():
    ticker = request.form.get("ticker", "").strip().upper()

    if not ticker:
        return jsonify({"erro": "Digite um ticker para analisar."})

    try:
        dados, info, erro = buscar_dados_empresa(ticker)

        if erro:
            return jsonify({"erro": erro})

        resultado = classificar_empresa(dados)
        resultado["info"] = info

        return jsonify(resultado)

    except Exception as e:
        return jsonify({"erro": f"Erro ao buscar dados: {str(e)}"})


if __name__ == "__main__":
    app.run(debug=True)