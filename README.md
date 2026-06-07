# M5 Retail Demand Forecasting

End-to-end demand forecasting on the **M5 Walmart** dataset. Forecasts the next **28 days** of unit sales at four business levels — **store, category, department, and item** — using SARIMAX with weekly seasonality, two feature setups, and a small grid search per level. Results are surfaced through a Streamlit dashboard.

> **TL;DR** — Aggregation level matters more than model tuning. Store-level forecasts land at **~5.9% MAPE**, item-level at **~54% MAPE**. SARIMAX is a strong baseline for aggregated series and a starting point for item-level work.

---

## Table of Contents
1. [Objective](#objective)
2. [The M5 Dataset](#the-m5-dataset)
3. [System Architecture](#system-architecture)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Key Findings](#key-findings)
7. [Business Recommendations](#business-recommendations)
8. [Project Structure](#project-structure)
9. [How to Run](#how-to-run)
10. [Tech Stack](#tech-stack)
11. [Roadmap](#roadmap)

---

## Objective

Retail demand is not equally predictable at every level. Aggregated series (store, category) are smoother and easier to forecast; granular series (single items) are noisy, sparse, and demand-spiked.

This project answers four questions on the M5 data:

- Which forecasting level gives the most reliable predictions?
- Does SARIMAX work better on aggregated or granular demand?
- Do calendar features (weekday, month, events, SNAP) actually improve accuracy?
- What business decisions can each forecast level reliably support?

---

## The M5 Dataset

The M5 dataset is the public dataset from the **M5 Forecasting — Accuracy** Kaggle competition. It contains **1,941 days** (~5.4 years) of unit-sales history for Walmart products across three US states.

### Files used

| File | Size | Description |
|---|---:|---|
| `sales_train_validation.csv` | 120 MB | Daily unit sales per item per store, columns `d_1 … d_1913` (wide format). |
| `sales_train_evaluation.csv` | 122 MB | Same shape as above but extends through `d_1941` (used as the held-out evaluation window). |
| `calendar.csv` | 103 KB | One row per date: weekday, month, year, event names/types, SNAP flags per state. |
| `sell_prices.csv` | 203 MB | Weekly selling price per item per store (loaded but not yet used as a SARIMAX exog — see [Roadmap](#roadmap)). |
| `sample_submission.csv` | 5 MB | Kaggle submission template. |

### Geographic & product hierarchy

```
3 states            CA, TX, WI
10 stores           CA_1..CA_4, TX_1..TX_3, WI_1..WI_3
3 categories        FOODS, HOUSEHOLD, HOBBIES
7 departments       FOODS_1, FOODS_2, FOODS_3,
                    HOUSEHOLD_1, HOUSEHOLD_2,
                    HOBBIES_1, HOBBIES_2
~3,049 items / store
~30,490 unique item x store series
```

### Long-format transformation

The raw wide format:

```
item_id | store_id | dept_id | cat_id | state_id | d_1 | d_2 | ... | d_1913
```

is reshaped into a tidy time-series:

```
date | item_id | store_id | dept_id | cat_id | sales | snap | weekday | month | event
```

This is the shape every downstream module consumes. Most experiments in this project use store **CA_1**; the store-level experiment uses all 10 stores.

---

## System Architecture

A single horizontal flow from raw files to dashboard. Each stage is a thin module in `src/`.

```
                       +---------------------+
                       |     M5 DATASET      |
                       |  sales | calendar   |
                       |       prices        |
                       +----------+----------+
                                  |
                                  v
                       +---------------------+
                       |   PREPROCESSING     |
                       |    wide -> long     |
                       |   merge calendar    |
                       +----------+----------+
                                  |
                +-----------------+-----------------+
                |                                   |
                v                                   v
        +---------------+                   +---------------+
        | FEATURE SET A |                   | FEATURE SET B |
        |   SNAP-only   |                   | + weekday +   |
        |               |                   | month + event |
        +-------+-------+                   +-------+-------+
                |                                   |
                +-----------------+-----------------+
                                  |
                                  v
                       +---------------------+
                       |  AGGREGATE BY LEVEL |
                       +----------+----------+
                                  |
        +-------------+-----------+-----------+-------------+
        |             |                       |             |
        v             v                       v             v
    +--------+   +---------+            +----------+   +--------+
    | STORE  |   |  CATEG  |            |   DEPT   |   | ITEMS  |
    |  (10)  |   |   (3)   |            |   (7)    |   |  (10)  |
    +----+---+   +----+----+            +-----+----+   +----+---+
         |            |                       |             |
         +------------+-----------+-----------+-------------+
                                  |
                                  v
                       +---------------------+
                       |      MODELING       |
                       |    Naive | MA-7     |
                       | SARIMAX grid search |
                       +----------+----------+
                                  |
                                  v
                       +---------------------+
                       |  SELECTION FUNNEL   |
                       |    A vs B  ->       |
                       |    converged  ->    |
                       |    lowest MAPE      |
                       +----------+----------+
                                  |
                                  v
                       +---------------------+
                       |     EVALUATION      |
                       |   28-day holdout    |
                       | MAE | RMSE | MAPE   |
                       +----------+----------+
                                  |
                                  v
                       +---------------------+
                       |       OUTPUTS       |
                       | forecasts | models  |
                       |  recommendations    |
                       +----------+----------+
                                  |
                                  v
                       +---------------------+
                       | STREAMLIT DASHBOARD |
                       +---------------------+
```

---

## Methodology

### Model

SARIMAX is the workhorse. The general form used is:

```
SARIMAX(order=(p, d, q), seasonal_order=(P, D, Q, 7), exog=external_features)
```

Seasonal period **7** captures the weekly retail pattern.

### Baselines

Two simple baselines are trained first so SARIMAX's lift can be measured:

- **Naive** — repeats the last observed value for all 28 forecast days.
- **7-day moving average** — uses the mean of the last 7 days.

### Grid search

For each series, a small grid of SARIMAX configurations is fit. Item-level uses a smaller grid because the series are noisier and the fits are more expensive.

```
(1,1,1)(1,1,1,7)
(1,1,0)(1,1,1,7)
(0,1,1)(1,1,1,7)
(1,1,1)(0,1,1,7)
(1,1,1)(1,1,0,7)
(0,1,1)(0,1,1,7)
```

Selection criteria, in order:

1. Successful convergence
2. Lowest validation MAPE
3. Forecast stability
4. Simpler feature setup when accuracy is similar

### Feature experiments

Two exogenous feature setups are compared per series:

- **SNAP-only** — `sales ~ history + SNAP`
- **Calendar** — `sales ~ history + SNAP + weekday + month + event`

**Finding:** calendar features help in isolated cases but did not consistently improve performance, and caused convergence problems on several lower-level series. Final models are chosen on validation stability, not feature richness.

### Evaluation

| Metric | Why |
|---|---|
| MAE | Average absolute error in units. |
| RMSE | Penalizes large misses. |
| MAPE | Business-readable %; primary comparison metric. Safe MAPE (`evaluate.py`) handles zero-sales days. |

Holdout is a single 28-day window per series (see [Roadmap](#roadmap) for rolling-origin CV).

---

## Results

| Forecast Level | Scope | Series | Best Series | Best MAPE | Weakest Series | Weakest MAPE | Avg MAPE |
|---|---|---:|---|---:|---|---:|---:|
| Store | All 10 stores | 10 | CA_1 | 4.20% | WI_2 | 9.44% | **5.93%** |
| Category | CA_1 categories | 3 | FOODS | 4.10% | HOBBIES | 13.47% | **7.51%** |
| Department | CA_1 departments | 7 | FOODS_3 | 4.25% | FOODS_1 | 25.30% | **12.97%** |
| Item | Top 10 CA_1 items | 10 | FOODS_3_586 | 17.46% | FOODS_3_120 | 143.95% | **54.13%** |

---

## Key Findings

**1. Store-level forecasting is the strongest** — 5.93% average MAPE. Aggregating across thousands of items per store cancels random noise. Reliable enough for demand planning, inventory allocation, and staffing.

**2. Category-level forecasting is close behind** — 7.51% MAPE. FOODS and HOUSEHOLD are clean; HOBBIES is noisier because its demand pattern is irregular.

**3. Department-level is mixed** — 12.97% MAPE. High-volume departments like FOODS_3 forecast cleanly; low-volume ones like FOODS_1 and HOBBIES_2 are unstable.

**4. Item-level is hard** — 54.13% MAPE. Single items have zero-sales days, sudden spikes, and sensitivity to price / promotions / stockouts that SARIMAX cannot see. Treat as a baseline only.

**5. Aggregation > tuning.** The single most important takeaway: the more aggregated the series, the more predictable it is. Don't expect item-level models to perform like store-level models.

---

## Business Recommendations

| Forecast Level | Reliability | Recommended Use |
|---|---|---|
| Store | High | Demand planning, staffing, inventory allocation |
| Category | High | Category replenishment and sales planning |
| Department | Medium | Department stock planning |
| Item | Low–Medium | Baseline only; requires richer features for production |

**Strategy:** use store and category forecasts for confident planning; department forecasts for medium-confidence decisions; item-level SARIMAX as a baseline to beat with price, promotion, and stockout features.

---

## Project Structure

```
m5_retail_demand_forecasting/
├── data/
│   ├── raw/                         # M5 source CSVs
│   └── processed/                   # Aggregated daily series per level
├── notebooks/
│   ├── category_level_forecasting.ipynb
│   ├── department_level_forecasting.ipynb
│   ├── store_level_forecasting.ipynb
│   ├── item_level_forecasting.ipynb
│   └── final_project_summary.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── baseline_models.py
│   ├── train_sarimax.py
│   ├── tune_sarimax.py
│   ├── evaluate.py
│   ├── forecast.py
│   └── recommendations.py
├── dashboard/
│   └── streamlit_app.py
├── outputs/                         # Forecasts, selected models, summaries
├── main.py                          # Paths, loaders, dashboard helpers
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Environment

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
source .venv/bin/activate
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Run the notebooks in order
```
category_level_forecasting.ipynb
department_level_forecasting.ipynb
store_level_forecasting.ipynb
item_level_forecasting.ipynb
final_project_summary.ipynb
```

### 4. Launch the dashboard
```bash
streamlit run dashboard/streamlit_app.py
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3 |
| Data | Pandas, NumPy |
| Modeling | Statsmodels (SARIMAX), Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| App | Streamlit |
| Persistence | Joblib |
| Notebooks | Jupyter |

---

<!-- ## Roadmap

Planned next iterations:

- **Use `sell_prices.csv`** as a SARIMAX exog (biggest expected lift on item-level).
- **Rolling-origin cross-validation** instead of a single 28-day holdout.
- **LightGBM / XGBoost** with lag, rolling-mean, calendar, and price features for item-level.
- **Hierarchical reconciliation** so store / category / department / item forecasts agree.
- **CLI entrypoint** so the pipeline runs end-to-end without notebooks.
- **Experiment tracking** (MLflow) and a model registry.
- **Forecast monitoring** dashboard for drift and accuracy decay.

--- -->

## Conclusion

SARIMAX is effective for stable, aggregated retail series — store and category forecasts come in at single-digit MAPE with no exogenous price data. Item-level forecasting is significantly harder: it needs price, promotion, and stockout signals plus more flexible models. The headline lesson from this project is simple:

> **Forecasting accuracy improves with aggregation; granular item-level forecasting needs richer signals and more advanced models.**
