# Fundamenta

**Machine Learning-powered financial health analysis for Brazilian stocks (B3)**

> Independent final project developed as an extension of CS50AI (Harvard) — applying Machine Learning concepts to real data from the Brazilian stock market.

---

## The Problem

Beginner investors have access to more information than ever — but too much information without context is just noise. Balance sheets, income statements, and dozens of financial indicators are intimidating for anyone starting out.

Fundamenta approaches this differently: instead of displaying every indicator, it **learns which ones historically mattered most** for each company — and presents only that signal, in plain language.

---

## How It Works

```
Ticker → Data via yfinance → ML Model → Classification + Explanation
```

1. The user enters a B3 ticker (e.g. `PETR4`, `WEGE3`)
2. The system fetches the latest financial data via **yfinance**
3. A **Random Forest** model trained on 153 companies and ~760 quarters classifies the company's financial health
4. The dashboard displays the classification, per-class probabilities, and the indicators that most influenced the decision

---

## What the Model Learns

The model was trained on quarterly historical data from 153 non-financial Brazilian companies across sectors including energy, retail, agribusiness, healthcare, technology, and utilities.

The target variable is a **financial health classification** with three classes:

| Class | Description |
|-------|-------------|
| ✅ Healthy | Consistent profitability, solid margins, controlled debt |
| ⚠️ Watch | Mixed signals — may reflect a difficult quarter or early deterioration |
| 🔴 Risk | Recurring losses, negative margins, or fragile financial structure |

**Top features learned by the model:**

| Indicator | Importance |
|-----------|------------|
| Net Margin | 25.8% |
| Net Income | 21.2% |
| Profit Consistency (3-quarter window) | 15.8% |
| Leverage | 6.7% |
| Cash Variation | 6.3% |

> The model discovered that profitability and margin matter more than debt alone — large companies operate with leverage and remain financially healthy.

---

## Performance

| Metric | Value |
|--------|-------|
| Test accuracy (80/20 split) | 90.8% |
| Cross-validation accuracy (5 folds) | 91.7% |
| CV standard deviation | 1.5% |
| Companies in training set | 153 |
| Quarters in dataset | 759 |

No **Risk** case was misclassified as **Healthy** or vice versa — errors occur only at the natural boundary between adjacent classes.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Web interface | Flask |
| ML model | scikit-learn (Random Forest) |
| Financial data | yfinance |
| Data processing | pandas, numpy |
| Frontend | HTML / CSS / Vanilla JS |

---

## Project Structure

```
fundamenta/
├── app.py                    # Flask server + prediction logic
├── coletar_historico.py      # Step 1 — historical data collection
├── preparar_dados.py         # Step 2 — cleaning, features, labels
├── treinar_modelo.py         # Step 3 — training and evaluation
├── buscar_empresas_b3.py     # Utility — fetch active B3 tickers
├── dados/
│   ├── dataset_preparado.csv
│   ├── empresas_b3.json
│   └── historico_precos.csv
├── modelo/
│   ├── modelo_saude.pkl
│   ├── features.pkl
│   └── medianas.pkl
└── templates/
    └── index.html
```

---

## Running Locally

**Requirements:** Python 3.10+, pip

```bash
# Clone the repository
git clone https://github.com/Chikwa-git/fundamenta
cd fundamenta

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python3 app.py
```

Open `http://localhost:5000` in your browser.

> The model is already trained and saved under `modelo/`. You do not need to run the data collection or training scripts to use the app.

---

## Retraining the Model

To update the dataset with more recent companies or quarters:

```bash
# 1. Update company list
python3 buscar_empresas_b3.py

# 2. Collect historical data
python3 coletar_historico.py

# 3. Prepare dataset
python3 preparar_dados.py

# 4. Retrain
python3 treinar_modelo.py
```

---

## Known Limitations

- **Not investment advice.** Analysis is based on historical data and fundamentals — it does not predict price movements.
- **Financial sector** companies (banks, insurers) are not supported in this version. Banking accounting has a different structure and requires a separate model — planned for a future release.
- **External factors** (macroeconomic conditions, political decisions, sector-wide events) are not captured by the model.
- Data covers ~7 quarters via yfinance. For longer history, Brazil's CVM provides standardized financial statements (DFPs) since 2010 at `dados.cvm.gov.br`.

---

## Background

This project was built as an independent final project after completing **CS50AI** (Harvard University), applying the Machine Learning and Neural Networks weeks to a real-world problem with Brazilian market data.

It is part of a portfolio connecting software development with business domain knowledge:

- [Radar Político](https://github.com/Chikwa-git/radar-politico) — Brazilian federal deputies voting analysis with Django + Groq
- [DataComex](https://github.com/Chikwa-git/datacomex) — Brazilian foreign trade analysis with Flask + ComexStat API

---

## Author

**Lincoln** — Career transitioner into software development.
Studying Systems Analysis and Development at Faculdade Descomplica, São Paulo, Brazil.

[GitHub](https://github.com/Chikwa-git) · [LinkedIn](https://linkedin.com/in/seu-perfil)
