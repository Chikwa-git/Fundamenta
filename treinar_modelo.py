"""
Etapa 3 — Treinamento do modelo
Projeto CS50AI - Análise de Saúde Financeira de Empresas (B3)

Roda com:
    python3 treinar_modelo.py

Dependências:
    pip install pandas numpy scikit-learn
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import os

# ─────────────────────────────────────────
# Carregar dataset preparado na Etapa 2
# ─────────────────────────────────────────
print("=" * 60)
print("ETAPA 3 — TREINAMENTO DO MODELO")
print("=" * 60)

print("\n📂 Carregando dataset...")
df = pd.read_csv("dados/dataset_preparado.csv")
print(f"   {len(df)} linhas | {df['ticker'].nunique()} empresas")

# ─────────────────────────────────────────
# Definir features (entradas do modelo)
#
# Não usamos: ticker, empresa, trimestre, saude_texto
# Essas colunas são identificadores, não sinais financeiros.
# O modelo deve aprender padrões nos números, não nos nomes.
# ─────────────────────────────────────────
FEATURES = [
    # Valores absolutos do balanço
    "total_ativos",
    "patrimonio_liquido",
    "caixa",
    "receita",
    "lucro_liquido",
    "ebitda",
    "divida_lp",

    # Variações trimestrais (tendência)
    "var_total_ativos_pct",
    "var_patrimonio_liquido_pct",
    "var_caixa_pct",
    "var_receita_pct",
    "var_lucro_liquido_pct",
    "var_ebitda_pct",

    # Indicadores calculados
    "margem_liquida_calc",
    "margem_ebitda_calc",
    "alavancagem",
    "trimestres_lucro_positivo",
]

TARGET = "saude"

# ─────────────────────────────────────────
# Preparar X e y
# ─────────────────────────────────────────
print("\n📊 Preparando features...")

# Filtrar só as colunas que existem no dataset
features_disponiveis = [f for f in FEATURES if f in df.columns]
features_ausentes = [f for f in FEATURES if f not in df.columns]

if features_ausentes:
    print(f"   ⚠️  Features ausentes (ignoradas): {features_ausentes}")

X = df[features_disponiveis].copy()
y = df[TARGET].copy()

print(f"   Features usadas: {len(features_disponiveis)}")
print(f"   Amostras totais: {len(X)}")

# ─────────────────────────────────────────
# Tratar valores nulos nas features
#
# Random Forest não aceita NaN.
# Estratégia: preencher com a mediana da coluna.
# Mediana é mais robusta que média para dados financeiros
# (não é distorcida por valores extremos como o lucro da Petrobras)
# ─────────────────────────────────────────
print("\n🔧 Tratando valores nulos...")
nulos_antes = X.isnull().sum().sum()
X = X.fillna(X.median(numeric_only=True))
nulos_depois = X.isnull().sum().sum()
print(f"   Nulos preenchidos com mediana: {nulos_antes} → {nulos_depois}")

# ─────────────────────────────────────────
# Divisão treino / teste
#
# 80% para treinar, 20% para testar.
# stratify=y garante que as 3 classes (Risco, Atenção,
# Saudável) fiquem proporcionalmente representadas
# nos dois conjuntos — importante com dataset pequeno.
# ─────────────────────────────────────────
print("\n✂️  Dividindo treino e teste (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,   # garante reprodutibilidade
    stratify=y
)
print(f"   Treino: {len(X_train)} amostras")
print(f"   Teste:  {len(X_test)} amostras")

# ─────────────────────────────────────────
# Construir o modelo
#
# Random Forest: conjunto de árvores de decisão.
# Cada árvore aprende padrões diferentes nos dados,
# e a classificação final é a "votação" entre todas.
#
# Por que Random Forest?
# - Lida bem com features em escalas muito diferentes
#   (lucro em bilhões vs margem em percentual)
# - Retorna importância das features nativamente
# - Robusto com dataset pequeno
# - Difícil de overfitar com os parâmetros certos
# ─────────────────────────────────────────
print("\n🌲 Construindo Random Forest...")

modelo = Pipeline([
    ("scaler", StandardScaler()),   # normaliza as escalas
    ("rf", RandomForestClassifier(
        n_estimators=200,      # 200 árvores na floresta
        max_depth=6,           # profundidade máxima de cada árvore
        min_samples_leaf=3,    # mínimo de amostras por folha (evita overfitting)
        class_weight="balanced",  # compensa o desbalanceamento das classes
        random_state=42
    ))
])

# ─────────────────────────────────────────
# Treinar
# ─────────────────────────────────────────
print("🏋️  Treinando...")
modelo.fit(X_train, y_train)
print("   ✅ Treinamento concluído")

# ─────────────────────────────────────────
# Avaliar no conjunto de teste
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("AVALIAÇÃO DO MODELO")
print("─" * 60)

y_pred = modelo.predict(X_test)

# Acurácia geral
acuracia = (y_pred == y_test).mean()
print(f"\n🎯 Acurácia no teste: {acuracia:.1%}")

# Relatório detalhado por classe
mapa = {0: "Risco", 1: "Atenção", 2: "Saudável"}
y_test_texto = y_test.map(mapa)
y_pred_texto = pd.Series(y_pred).map(mapa)

print("\n📋 Relatório por classe:")
print(classification_report(
    y_test_texto, y_pred_texto,
    target_names=["Atenção", "Risco", "Saudável"]
))

# Matriz de confusão
print("🔢 Matriz de confusão:")
print("   (linhas = real | colunas = previsto)\n")
classes = ["Risco", "Atenção", "Saudável"]
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
print(cm_df.to_string())

# ─────────────────────────────────────────
# Validação cruzada
#
# Treinar e testar em 5 subconjuntos diferentes dos dados.
# Dá uma noção mais confiável da performance real do modelo,
# especialmente com dataset pequeno.
# ─────────────────────────────────────────
print("\n🔄 Validação cruzada (5 folds)...")
scores = cross_val_score(modelo, X, y, cv=5, scoring="accuracy")
print(f"   Acurácia por fold: {[f'{s:.1%}' for s in scores]}")
print(f"   Média: {scores.mean():.1%} | Desvio: {scores.std():.1%}")

# ─────────────────────────────────────────
# Importância das features
#
# O coração do projeto: quais indicadores o modelo
# considerou mais relevantes para classificar saúde?
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("IMPORTÂNCIA DAS FEATURES")
print("─" * 60)
print("(quanto cada indicador pesou na decisão do modelo)\n")

rf = modelo.named_steps["rf"]
importancias = pd.Series(
    rf.feature_importances_,
    index=features_disponiveis
).sort_values(ascending=False)

for feature, importancia in importancias.items():
    barra = "█" * int(importancia * 100)
    print(f"   {feature:<35} {importancia:.3f}  {barra}")

# ─────────────────────────────────────────
# Testar com exemplos reais do dataset
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("EXEMPLOS DE PREDIÇÃO")
print("─" * 60)

# Pegar o último trimestre de algumas empresas conhecidas
empresas_exemplo = ["PETR4.SA", "MGLU3.SA", "WEGE3.SA", "BHIA3.SA", "VALE3.SA"]
exemplos = (
    df[df["ticker"].isin(empresas_exemplo)]
    .sort_values("trimestre")
    .groupby("ticker")
    .tail(1)
)

if not exemplos.empty:
    X_exemplo = exemplos[features_disponiveis].fillna(X.median(numeric_only=True))
    predicoes = modelo.predict(X_exemplo)
    probabilidades = modelo.predict_proba(X_exemplo)

    print()
    for i, (_, row) in enumerate(exemplos.iterrows()):
        pred = mapa[predicoes[i]]
        probs = probabilidades[i]
        empresa = row["empresa"]
        trimestre = row["trimestre"][:7]

        print(f"   {empresa} ({trimestre})")
        print(f"   → Classificação: {pred}")
        print(f"   → Confiança: Risco {probs[0]:.0%} | Atenção {probs[1]:.0%} | Saudável {probs[2]:.0%}")
        print()

# ─────────────────────────────────────────
# Salvar modelo treinado
# ─────────────────────────────────────────
print("─" * 60)
print("SALVANDO MODELO")
print("─" * 60)

os.makedirs("modelo", exist_ok=True)

# Salvar modelo
with open("modelo/modelo_saude.pkl", "wb") as f:
    pickle.dump(modelo, f)

# Salvar lista de features (importante para usar na interface)
with open("modelo/features.pkl", "wb") as f:
    pickle.dump(features_disponiveis, f)

# Salvar medianas para preencher nulos na interface
medianas = X.median(numeric_only=True)
with open("modelo/medianas.pkl", "wb") as f:
    pickle.dump(medianas, f)

print(f"\n✅ modelo/modelo_saude.pkl")
print(f"✅ modelo/features.pkl")
print(f"✅ modelo/medianas.pkl")

print("\n" + "=" * 60)
print("Etapa 3 concluída.")
print("Próximo passo: Etapa 4 — Interface Flask.")
print("=" * 60)
