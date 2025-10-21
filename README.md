# EverBrook_US_LS_Backtester

Developed at **Columbia University** by a team of students specializing in **quantitative finance**, this project harnesses algorithmic modeling, data analytics, and machine learning to design and evaluate systematic investment strategies.  
Our goal is to create a robust framework for training future quantitative analysts and portfolio engineers.

---

## 📊 Quant Research Project Overview

This repository contains **factor models** and **long–short backtesting engines** built and maintained in Google Colab.  
It provides a complete workflow — from factor construction and normalization to portfolio optimization, monthly rebalancing, and performance evaluation.

---

## 🧩 Core Dependencies
- **numpy** — numerical computing  
- **pandas** — data processing and alignment  
- **matplotlib** — visualization  
- **cvxpy** — portfolio optimization  

---

## 🔍 Repository Structure
- `data/` — stores Russell 3000 daily data (open, high, low, volume, and sector codes)  
- `normalize/` — includes equity beta factors such as beta, momentum, turnover, volatility, and P/B ratio  
- `factor/` — defines the factor pipeline used to compute daily factor exposures for backtesting  
- `backtest/` — implements the backtesting engine to calculate monthly factor returns, 5-basket performance, and IC metrics  

---

## ⚙️ Key Functions
- **`neutralize_cross_sectional_lagged()`** — cross-sectionally neutralizes alpha factors by beta-style factors  
- **`optimize_portfolio()`** — enforces optimization constraints within each basket to align backtest results with production portfolios  
- **`chain_mtd_rebalance()`** — rebalances monthly at the start of each month using prior-day factor values; computes daily returns and reinvests gains  
- **`compute_monthly_ic()`** — computes the Pearson correlation between factor values and next-month returns  
- **`backtest()`** — end-to-end backtest engine producing monthly factor returns, basket spreads, and IC summaries  

---

## 📉 Demo: Bollinger-Style Factor

The **Bollinger factor** is defined as:

$$
\text{Factor} = \frac{P - \text{Low}}{\text{High} - \text{Low}} \\
,\text{High} = \text{MA} + k \times \text{STD}, \quad \text{Low} = \text{MA} - k \times \text{STD}
$$


The chart (to be imported) illustrates Russell 3000 performance from 2015 to 2025, revealing a **momentum-like structure** — past returns exhibit predictive power for the next 30 days.  
Top-basket outperformance and IC > 0.05 indicate this factor’s potential inclusion in composite signal construction.

---

## 🔧 Ongoing Improvements
- Streamline and modularize risk-factor computation  
- Migrate computation to **Polars** or **PyTorch** for GPU acceleration  
- Evaluate full long–short portfolio performance metrics  
- Incorporate **market-impact and trading-cost** approximations  

---

## 🚀 How to Run (unavailable)
1. Open notebooks in [Google Colab](https://colab.research.google.com).  
2. Install dependencies from `requirements.txt`.  
3. Run cells sequentially to reproduce results.

---

## 📈 Example Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<username>/<repo>/blob/main/<notebook>.ipynb)

