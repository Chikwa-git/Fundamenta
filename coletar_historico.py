"""
Etapa 1 — Coleta de dados históricos
Projeto CS50AI - Análise de Saúde Financeira de Empresas (B3)

Roda com:
    python3 coletar_historico.py

Dependências:
    pip install yfinance pandas
"""

import yfinance as yf
import pandas as pd
import time
import os

# ─────────────────────────────────────────
# lista de empresas — setores variados
# para o modelo ver perfis bem diferentes
# ─────────────────────────────────────────

EMPRESAS = {
    "RAIZ4.SA": "RAIZEN      PN      N2",
    "CSAN3.SA": "COSAN       ON      NM",
    "PETR4.SA": "PETROBRAS   PN      N2",
    "BEEF3.SA": "MINERVA     ON      NM",
    "ITSA4.SA": "ITAUSA      PN      N1",
    "ABEV3.SA": "AMBEV S/A   ON",
    "RADL3.SA": "RAIADROGASILON      NM",
    "ASAI3.SA": "ASSAI       ON      NM",
    "CPLE3.SA": "COPEL       ON      NM",
    "MGLU3.SA": "MAGAZ LUIZA ON      NM",
    "USIM5.SA": "USIMINAS    PNA     N1",
    "ONCO3.SA": "ONCOCLINICASON      NM",
    "CYRE3.SA": "CYRELA REALTON      NM",
    "COGN3.SA": "COGNA ON    ON      NM",
    "SBSP3.SA": "SABESP      ON      NM",
    "CSNA3.SA": "SID NACIONALON",
    "VALE3.SA": "VALE        ON      NM",
    "NATU3.SA": "NATURA      ON      NM",
    "VAMO3.SA": "VAMOS       ON      NM",
    "GGBR4.SA": "GERDAU      PN  ED  N1",
    "CVCB3.SA": "CVC BRASIL  ON      NM",
    "CMIG4.SA": "CEMIG       PN      N1",
    "RAIL3.SA": "RUMO S.A.   ON      NM",
    "MOTV3.SA": "MOTIVA SA   ON      NM",
    "EQTL3.SA": "EQUATORIAL  ON      NM",
    "RDOR3.SA": "REDE D OR   ON      NM",
    "LIGT3.SA": "LIGHT S/A   ON  ES  NM",
    "CSMG3.SA": "COPASA      ON      NM",
    "GOAU4.SA": "GERDAU MET  PN  ED  N1",
    "TOTS3.SA": "TOTVS       ON      NM",
    "PMAM3.SA": "PARANAPANEMAON  ES  NM",
    "DIRR3.SA": "DIRECIONAL  ON      NM",
    "JHSF3.SA": "JHSF PART   ON      NM",
    "MRVE3.SA": "MRV         ON      NM",
    "AXIA3.SA": "AXIA ENERGIAON      N1",
    "RENT3.SA": "LOCALIZA    ON      NM",
    "PRIO3.SA": "PRIO        ON      NM",
    "AMBP3.SA": "AMBIPAR     ON      NM",
    "LREN3.SA": "LOJAS RENNERON      NM",
    "MBRF3.SA": "MARFRIG     ON      NM",
    "CMIN3.SA": "CSNMINERACAOON      N2",
    "BRAV3.SA": "BRAVA       ON      NM",
    "POMO4.SA": "MARCOPOLO   PN      N2",
    "PETR3.SA": "PETROBRAS   ON      N2",
    "PGMN3.SA": "PAGUE MENOS ON      NM",
    "CEAB3.SA": "CEA MODAS   ON      NM",
    "VBBR3.SA": "VIBRA       ON      NM",
    "GMAT3.SA": "GRUPO MATEUSON      NM",
    "ECOR3.SA": "ECORODOVIAS ON  ED  NM",
    "SUZB3.SA": "SUZANO S.A. ON      NM",
    "AZEV4.SA": "AZEVEDO     PN",
    "WEGE3.SA": "WEG         ON      NM",
    "CURY3.SA": "CURY S/A    ON  ED  NM",
    "UGPA3.SA": "ULTRAPAR    ON      NM",
    "GGPS3.SA": "GPS         ON      NM",
    "ENEV3.SA": "ENEVA       ON      NM",
    "KLBN4.SA": "KLABIN S/A  PN      N2",
    "SMFT3.SA": "SMART FIT   ON      NM",
    "ALOS3.SA": "ALLOS       ON  ED  NM",
    "SAUD3.SA": "BRADSAUDE   ON      NM",
    "BRKM5.SA": "BRASKEM     PNA     N1",
    "HBSA3.SA": "HIDROVIAS   ON      NM",
    "MOVI3.SA": "MOVIDA      ON      NM",
    "VIVT3.SA": "TELEF BRASILON",
    "GRND3.SA": "GRENDENE    ON  EDJ NM",
    "RCSL4.SA": "RECRUSUL    PN",
    "TIMS3.SA": "TIM         ON      NM",
    "BHIA3.SA": "CASAS BAHIA ON      NM",
    "ANIM3.SA": "ANIMA       ON      NM",
    "LWSA3.SA": "LWSA        ON      NM",
    "FLRY3.SA": "FLEURY      ON      NM",
    "EMBJ3.SA": "EMBRAER     ON      NM",
    "CPFE3.SA": "CPFL ENERGIAON      NM",
    "VIVA3.SA": "VIVARA S.A. ON      NM",
    "PLPL3.SA": "PLANOEPLANO ON      NM",
    "AUAU3.SA": "PETZCOBASI  ON  ED  NM",
    "LJQQ3.SA": "QUERO,QUERO ON      NM",
    "SLCE3.SA": "SLC AGRICOLAON      NM",
    "MULT3.SA": "MULTIPLAN   ON      N2",
    "DXCO3.SA": "DEXCO       ON      NM",
    "YDUQ3.SA": "YDUQS PART  ON      NM",
    "AZEV3.SA": "AZEVEDO     ON",
    "RECV3.SA": "PETRORECSA  ON  EJ  NM",
    "MILS3.SA": "MILLS       ON      NM",
    "ISAE4.SA": "ISA ENERGIA PN      N1",
    "PCAR3.SA": "P.ACUCAR,CBDON      NM",
    "DASA3.SA": "DASA        ON      NM",
    "CBAV3.SA": "CBA         ON      NM",
    "AXIA6.SA": "AXIA ENERGIAPNB     N1",
    "CASH3.SA": "MELIUZ      ON      NM",
    "SMTO3.SA": "SAO MARTINHOON      NM",
    "QUAL3.SA": "QUALICORP   ON      NM",
    "EGIE3.SA": "ENGIE BRASILON      NM",
    "SIMH3.SA": "SIMPAR      ON      NM",
    "SBFG3.SA": "GRUPO SBF   ON      NM",
    "PASS3.SA": "COMPASS     ON      NM",
    "SAPR4.SA": "SANEPAR     PN      N2",
    "AZTE3.SA": "AZT ENERGIA ON",
    "AURE3.SA": "AUREN       ON      NM",
    "AZZA3.SA": "AZZAS 2154  ON      NM",
    "AMER3.SA": "AMERICANAS  ON      NM",
    "TEND3.SA": "TENDA       ON      NM",
    "EVEN3.SA": "EVEN        ON      NM",
    "RAPT4.SA": "RANDON PART PN      N1",
    "AXIA7.SA": "AXIA ENERGIAPNC     N1",
    "HYPE3.SA": "HYPERA      ON      NM",
    "INTB3.SA": "INTELBRAS   ON      NM",
    "JALL3.SA": "JALLESMACHADON      NM",
    "ALPA4.SA": "ALPARGATAS  PN      N1",
    "KEPL3.SA": "KEPLER WEBERON      NM",
    "SHUL4.SA": "SCHULZ      PN",
    "RIAA3.SA": "RIACHUELO   ON  EJ  NM",
    "CAML3.SA": "CAMIL       ON      NM",
    "KLBN3.SA": "KLABIN S/A  ON      N2",
    "SOJA3.SA": "BOA SAFRA   ON      NM",
    "MLAS3.SA": "MULTILASER  ON      NM",
    "TTEN3.SA": "3TENTOS     ON      NM",
    "ORVR3.SA": "ORIZON      ON      NM",
    "MTRE3.SA": "MITRE REALTYON  ED  NM",
    "VULC3.SA": "VULCABRAS   ON      NM",
    "VTRU3.SA": "VITRUEDUCA  ON      NM",
    "FIQE3.SA": "UNIFIQUE    ON      NM",
    "DESK3.SA": "DESKTOP     ON      NM",
    "EZTC3.SA": "EZTEC       ON  ED  NM",
    "MATD3.SA": "MATER DEI   ON      NM",
    "PNVL3.SA": "DIMED       ON      NM",
    "FESA4.SA": "FERBASA     PN      N1",
    "RENT4.SA": "LOCALIZA    PN      NM",
    "BMOB3.SA": "BEMOBI TECH ON  EJ  NM",
    "VITT3.SA": "VITTIA      ON      NM",
    "POSI3.SA": "POSITIVO TECON      NM",
    "AMAR3.SA": "LOJAS MARISAON      NM",
    "MYPK3.SA": "IOCHP,MAXIONON      NM",
    "MDNE3.SA": "MOURA DUBEUXON      NM",
    "ESPA3.SA": "ESPACOLASER ON      NM",
    "TUPY3.SA": "TUPY        ON      NM",
    "LOGG3.SA": "LOG COM PROPON  ED  NM",
    "USIM3.SA": "USIMINAS    ON      N1",
    "OPCT3.SA": "OCEANPACT   ON      NM",
    "GFSA3.SA": "GAFISA      ON      NM",
    "SEER3.SA": "SER EDUCA   ON      NM",
    "VVEO3.SA": "VIVEO       ON      NM",
    "JSLG3.SA": "JSL         ON      NM",
    "HBOR3.SA": "HELBOR      ON      NM",
    "ARML3.SA": "ARMAC       ON      NM",
    "SEQL3.SA": "SEQUOIA LOG ON      NM",
    "MELK3.SA": "MELNICK     ON      NM",
    "MDIA3.SA": "M.DIASBRANCOON  ED  NM",
    "PTBL3.SA": "PORTOBELLO  ON      NM",
    "BLAU3.SA": "BLAU        ON      NM",
    "ALLD3.SA": "ALLIED      ON      NM",
    "RANI3.SA": "IRANI       ON      NM",
    "TASA4.SA": "TAURUS ARMASPN      N2",
}

# ─────────────────────────────────────────
# Indicadores que vamos coletar
# ─────────────────────────────────────────
INDICADORES = [
    "returnOnEquity",       # ROE
    "returnOnAssets",       # ROA
    "profitMargins",        # Margem Líquida
    "ebitdaMargins",        # Margem EBITDA
    "debtToEquity",         # Dívida / Patrimônio
    "currentRatio",         # Liquidez Corrente
    "revenueGrowth",        # Crescimento de Receita
    "earningsGrowth",       # Crescimento de Lucro
    "freeCashflow",         # Fluxo de Caixa Livre
    "trailingPE",           # P/L
    "priceToBook",          # P/VP
]

# ─────────────────────────────────────────
# Funções auxiliares
# ─────────────────────────────────────────

def coletar_indicadores(ticker, nome):
    """Coleta snapshot atual dos indicadores fundamentalistas."""
    try:
        ativo = yf.Ticker(ticker)
        info = ativo.info

        linha = {
            "ticker": ticker,
            "empresa": nome,
        }

        for indicador in INDICADORES:
            linha[indicador] = info.get(indicador, None)

        return linha

    except Exception as e:
        print(f"      Erro nos indicadores: {e}")
        return None


def coletar_balanco_trimestral(ticker, nome):
    """
    Coleta balanço patrimonial e DRE trimestrais.
    Retorna um DataFrame com os últimos trimestres disponíveis.
    """
    try:
        ativo = yf.Ticker(ticker)

        # Balanço Patrimonial
        bp = ativo.quarterly_balance_sheet
        # Demonstração de Resultados
        dre = ativo.quarterly_financials

        if bp.empty and dre.empty:
            print(f"      Sem dados de balanço")
            return None

        linhas = []

        # Iterar sobre os trimestres disponíveis no balanço
        trimestres = bp.columns if not bp.empty else dre.columns

        for trimestre in trimestres:
            linha = {
                "ticker": ticker,
                "empresa": nome,
                "trimestre": trimestre.strftime("%Y-%m-%d"),
            }

            # Dados do Balanço Patrimonial
            if not bp.empty and trimestre in bp.columns:
                col = bp[trimestre]
                linha["total_ativos"] = col.get("Total Assets", None)
                linha["total_passivos"] = col.get(
                    "Total Liabilities Net Minority Interest", None
                )
                linha["patrimonio_liquido"] = col.get(
                    "Stockholders Equity", None
                )
                linha["caixa"] = col.get("Cash And Cash Equivalents", None)
                linha["divida_lp"] = col.get("Long Term Debt", None)
                linha["divida_cp"] = col.get(
                    "Current Debt And Capital Lease Obligation", None
                )

            # Dados da DRE
            if not dre.empty and trimestre in dre.columns:
                col = dre[trimestre]
                linha["receita"] = col.get("Total Revenue", None)
                linha["lucro_bruto"] = col.get("Gross Profit", None)
                linha["lucro_liquido"] = col.get("Net Income", None)
                linha["ebitda"] = col.get("EBITDA", None)

            linhas.append(linha)

        return pd.DataFrame(linhas)

    except Exception as e:
        print(f"      Erro no balanço trimestral: {e}")
        return None


def coletar_historico_preco(ticker, nome):
    """
    Coleta preço de fechamento trimestral dos últimos 5 anos.
    Calcula variação percentual entre trimestres.
    """
    try:
        ativo = yf.Ticker(ticker)
        hist = ativo.history(period="5y", interval="3mo")

        if hist.empty:
            print(f"      Sem histórico de preços")
            return None

        hist = hist[["Close"]].copy()
        hist.columns = ["preco_fechamento"]
        hist["variacao_pct"] = hist["preco_fechamento"].pct_change() * 100
        hist["ticker"] = ticker
        hist["empresa"] = nome
        hist.index = hist.index.strftime("%Y-%m-%d")
        hist.index.name = "trimestre"
        hist = hist.reset_index()

        return hist

    except Exception as e:
        print(f"      Erro no histórico de preços: {e}")
        return None


# ─────────────────────────────────────────
# Coleta principal
# ─────────────────────────────────────────

def main():
    os.makedirs("dados", exist_ok=True)

    todos_indicadores = []
    todos_balancos = []
    todos_precos = []

    total = len(EMPRESAS)

    for i, (ticker, nome) in enumerate(EMPRESAS.items(), 1):
        print(f"\n[{i}/{total}] {nome} ({ticker})")

        # 1. Indicadores atuais
        print("   → Indicadores fundamentalistas...")
        indicadores = coletar_indicadores(ticker, nome)
        if indicadores:
            todos_indicadores.append(indicadores)
            print("      ✅ OK")

        # 2. Balanço trimestral histórico
        print("   → Balanço trimestral...")
        balanco = coletar_balanco_trimestral(ticker, nome)
        if balanco is not None:
            todos_balancos.append(balanco)
            print(f"      ✅ {len(balanco)} trimestres coletados")

        # 3. Histórico de preços
        print("   → Histórico de preços...")
        precos = coletar_historico_preco(ticker, nome)
        if precos is not None:
            todos_precos.append(precos)
            print(f"      ✅ {len(precos)} trimestres coletados")

        # Pausa entre empresas para não sobrecarregar a API
        if i < total:
            print("   ⏳ Aguardando 2s...")
            time.sleep(2)

    # ─────────────────────────────────────────
    # Salvar CSVs
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SALVANDO ARQUIVOS...")
    print("=" * 60)

    if todos_indicadores:
        df_ind = pd.DataFrame(todos_indicadores)
        df_ind.to_csv("dados/indicadores_atuais.csv", index=False)
        print(f"✅ dados/indicadores_atuais.csv — {len(df_ind)} empresas")

    if todos_balancos:
        df_bal = pd.concat(todos_balancos, ignore_index=True)
        df_bal.to_csv("dados/balancos_trimestrais.csv", index=False)
        print(f"✅ dados/balancos_trimestrais.csv — {len(df_bal)} linhas")

    if todos_precos:
        df_pre = pd.concat(todos_precos, ignore_index=True)
        df_pre.to_csv("dados/historico_precos.csv", index=False)
        print(f"✅ dados/historico_precos.csv — {len(df_pre)} linhas")

    print("\n" + "=" * 60)
    print("Coleta concluída.")
    print("Próximo passo: Etapa 2 — Preparação dos dados.")
    print("=" * 60)


if __name__ == "__main__":
    main()
