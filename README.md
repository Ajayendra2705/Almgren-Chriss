# Almgren-Chriss
# 📈 Almgren–Stoikov Optimal Execution Model

This project implements the **Almgren–Stoikov model** for optimal trade execution.  
The goal is to determine how to split a large order (e.g., selling 10,000 shares) into smaller trades over time while minimizing **market impact costs** and **execution risk**.

---

## 🔍 Motivation

When executing large stock orders, placing the entire order at once can:
- Cause **significant slippage** due to liquidity constraints
- Impact the market price unfavorably
- Increase transaction costs

The **Almgren–Stoikov framework** provides an optimal trading trajectory by balancing:
1. **Market impact** (temporary + permanent)
2. **Price volatility risk** over the execution horizon

---

## 📊 Model Overview

- **Inputs**
  - Initial stock price `S0`
  - Total shares to trade `Q`
  - Time horizon `T` (e.g., 1 trading day)
  - Number of intervals `N`
  - Market volatility `σ` (estimated from intraday returns)
  - Market impact parameters (`η`, `γ`)

- **Outputs**
  - Optimal execution schedule (shares to trade per interval)
  - Cumulative execution curve
  - Simulated price path with impact
  - VWAP benchmark comparison
