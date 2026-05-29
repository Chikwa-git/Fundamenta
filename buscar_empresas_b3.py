"""
Utilitário — Buscar lista de empresas da B3
Projeto CS50AI - Análise de Saúde Financeira de Empresas (B3)

Roda com:
    python3 buscar_empresas_b3.py

Dependências:
    pip install pandas yfinance requests lxml
"""

import requests
import pandas as pd
import yfinance as yf
import time
import json
import os
from io import StringIO

SETORES_FINANCEIROS = ["Financial Services", "Banks", "Insurance"]


# ─────────────────────────────────────────
# Fonte principal — Yahoo Finance Screener
#
# Retorna empresas brasileiras ATIVAS com
# dados disponíveis, ordenadas por volume.
# Faz múltiplas páginas para cobrir mais empresas.
# ─────────────────────────────────────────
def buscar_via_yahoo(max_tickers=500):
    print("   → Buscando via Yahoo Finance screener...")
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
    tickers = []
    offset = 0
    pagina = 1
    tamanho_pagina = 250

    while len(tickers) < max_tickers:
        url = (
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            f"?formatted=true&lang=pt-BR&region=BR"
            f"&scrIds=most_actives_br&count={tamanho_pagina}&offset={offset}"
        )
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()

        data = r.json()
        resultado = data.get("finance", {}).get("result", [])

        if not resultado:
            break

        quotes = resultado[0].get("quotes", [])
        if not quotes:
            break

        novos = [q["symbol"] for q in quotes if q["symbol"].endswith(".SA")]
        tickers.extend(novos)

        print(f"   Página {pagina}: {len(novos)} empresas coletadas")

        # Se veio menos que o tamanho da página, chegou ao fim
        if len(novos) < tamanho_pagina:
            break

        offset += tamanho_pagina
        pagina += 1
        time.sleep(1)

    # Remover duplicatas mantendo ordem
    vistos = set()
    unicos = []
    for t in tickers:
        if t not in vistos:
            vistos.add(t)
            unicos.append(t)

    print(f"   ✅ {len(unicos)} tickers únicos via Yahoo Finance")
    return unicos


# ─────────────────────────────────────────
# Fonte fallback — Fundamentus
# Usado se o Yahoo falhar
# ─────────────────────────────────────────
def buscar_via_fundamentus():
    print("   → Tentando Fundamentus como fallback...")
    url = "https://www.fundamentus.com.br/resultado.php"
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    tabelas = pd.read_html(StringIO(r.text), flavor="lxml")
    df = tabelas[0]
    col_ticker = df.columns[0]

    # Fundamentus tem muitos tickers inativos — filtrar só os mais
    # prováveis de ter dados: tickers de 5-6 chars terminando em número
    tickers = [
        str(t).strip() + ".SA"
        for t in df[col_ticker].dropna().unique()
        if isinstance(t, str)
        and len(str(t).strip()) in [5, 6]
        and str(t).strip()[-1].isdigit()
        # Só ações ON (3), PN (4) e Units (11) — mais líquidas
        and str(t).strip()[-2:] in ["3", "4", "11", "34"]
    ]
    print(f"   ✅ {len(tickers)} tickers via Fundamentus")
    return tickers


# ─────────────────────────────────────────
# Orquestrador
# ─────────────────────────────────────────
def buscar_lista_b3():
    print("📡 Buscando lista de empresas da B3...")

    fontes = [
        ("Yahoo Finance", buscar_via_yahoo),
        ("Fundamentus",   buscar_via_fundamentus),
    ]

    for nome, funcao in fontes:
        try:
            tickers = funcao()
            if tickers:
                return tickers
        except Exception as e:
            print(f"   ⚠️  {nome} indisponível: {e}")

    print("\n❌ Nenhuma fonte disponível. Verifique sua conexão.")
    exit(1)


# ─────────────────────────────────────────
# Filtrar setores financeiros via yfinance
# ─────────────────────────────────────────
def filtrar_nao_financeiras(tickers, max_empresas=None):
    print(f"\n🔍 Verificando setores ({len(tickers)} empresas)...")
    print("   Isso pode levar alguns minutos...\n")

    aprovadas = {}
    rejeitadas = []
    sem_dados = []

    total = min(len(tickers), max_empresas) if max_empresas else len(tickers)

    for i, ticker in enumerate(tickers[:total], 1):
        try:
            info = yf.Ticker(ticker).info
            setor = info.get("sector", "")
            nome = info.get("shortName") or info.get("longName") or ticker

            if not setor:
                sem_dados.append(ticker)
                print(f"   [{i:3}/{total}] {ticker:<12} ⚠️  sem setor")
            elif setor in SETORES_FINANCEIROS:
                rejeitadas.append(ticker)
                print(f"   [{i:3}/{total}] {ticker:<12} ❌ {setor}")
            else:
                aprovadas[ticker] = nome
                print(f"   [{i:3}/{total}] {ticker:<12} ✅ {setor}")

        except Exception as e:
            sem_dados.append(ticker)
            print(f"   [{i:3}/{total}] {ticker:<12} ⚠️  erro: {e}")

        if i < total:
            time.sleep(1)

    return aprovadas, rejeitadas, sem_dados


# ─────────────────────────────────────────
# Salvar resultados
# ─────────────────────────────────────────
def salvar_resultado(aprovadas, rejeitadas, sem_dados):
    os.makedirs("dados", exist_ok=True)

    with open("dados/empresas_b3.json", "w", encoding="utf-8") as f:
        json.dump(aprovadas, f, ensure_ascii=False, indent=2)

    pd.DataFrame([
        {"ticker": t, "nome": n} for t, n in aprovadas.items()
    ]).to_csv("dados/empresas_b3.csv", index=False)

    with open("dados/empresas_dict.py", "w", encoding="utf-8") as f:
        f.write("# Cole este dicionário no coletar_historico.py\n\n")
        f.write("EMPRESAS = {\n")
        for ticker, nome in aprovadas.items():
            f.write(f'    "{ticker}": "{nome}",\n')
        f.write("}\n")

    print(f"\n✅ dados/empresas_b3.json  — {len(aprovadas)} empresas")
    print(f"✅ dados/empresas_b3.csv")
    print(f"✅ dados/empresas_dict.py  — cole no coletar_historico.py")
    print(f"\n📊 Resumo:")
    print(f"   Aprovadas (não-financeiras): {len(aprovadas)}")
    print(f"   Rejeitadas (financeiras):    {len(rejeitadas)}")
    print(f"   Sem dados / inativas:        {len(sem_dados)}")
    if rejeitadas:
        print(f"\n   Financeiras removidas: {', '.join(rejeitadas)}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("BUSCADOR DE EMPRESAS DA B3")
    print("=" * 60)

    tickers = buscar_lista_b3()

    print(f"\nForam encontrados {len(tickers)} tickers.")
    print("Quantos deseja verificar agora?")
    print("  [1] Todas")
    print("  [2] Primeiras 100 (recomendado)")
    print("  [3] Primeiras 50 (rápido)")

    escolha = input("\nEscolha (1/2/3): ").strip()
    limites = {"1": None, "2": 100, "3": 50}
    limite = limites.get(escolha, 100)

    aprovadas, rejeitadas, sem_dados = filtrar_nao_financeiras(tickers, limite)

    print("\n" + "─" * 60)
    print("SALVANDO RESULTADOS")
    print("─" * 60)
    salvar_resultado(aprovadas, rejeitadas, sem_dados)

    print("\n" + "=" * 60)
    print("Próximo passo:")
    print("  1. Substitua EMPRESAS em coletar_historico.py")
    print("     pelo conteúdo de dados/empresas_dict.py")
    print("  2. python3 coletar_historico.py")
    print("  3. python3 preparar_dados.py")
    print("  4. python3 treinar_modelo.py")
    print("=" * 60)