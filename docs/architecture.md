# ATLAS V12+ Architecture

11-layer hierarchical trading engine with regime-aware execution and probabilistic framework.

## Pipeline

```
Slack @mention
  -> bot.py: handle_mention()
    -> data_fetcher.fetch_live_data()         # yfinance
    -> atlas_engine.run_atlas()               # 11-layer composite
    -> v8_data.fetch_v8_data()                # extended data + Monte Carlo DCF
    -> v8_report.format_v8_report()           # Slack report formatter
    -> web_report.generate_and_store_report() # SQLite persistence
    -> Slack: threaded messages
```

## Layers (atlas_engine.py)

| # | Layer | Function | Output |
|---|-------|----------|--------|
| 0 | Data Integrity | `compute_data_confidence` | DC: 0-100 |
| 1 | Regime Vector | `compute_regime_vector` + `compute_soft_regime` | 10-feature vector, GMM probs, reliability |
| 2 | Score Normalization | `normalize_engine_scores` | `e_norm = tanh(score/s)` per engine |
| 3 | Meta-Learning | `update_meta_learning` | Base weights via exponentiated gradient |
| 4 | Dynamic Weights | `compute_dynamic_weights` | `w = softmax(log(w0) + alpha * A @ r)` |
| 5 | Composite Score | `compute_composite` | `C = 100 * (w'e + e'A(r)e)` |
| 6 | Risk Governor | `compute_risk_governor` | Gate G, adjusted composite |
| 7 | Trade Quality | `compute_trade_quality` | `TQ = |C|/100 * Rel * G * DC/100` |
| 8 | Portfolio Policy | `compute_portfolio_policy` | Regime-conditioned allocation |
| 9 | Execution | `compute_execution_micro` | Entry, stop, TP, mode |
| 10 | Report | `generate_pyramid_report` | Full text report |

## 8 Engines

| Engine | Score Range | Signal |
|--------|------------|--------|
| Trend | -100 to +100 | MA crossovers, breakouts, volume |
| Valuation | -40 to +40 | PE/EV z-scores, FCF yield |
| Consensus | -50 to +50 | Revisions, surprises, target |
| Volatility | -50 to +30 | VIX regime, IV/RV, put/call |
| Macro | -50 to +30 | Yields, curve, credit spreads |
| Liquidity | -50 to +50 | Volume ratio, breadth, NH/NL |
| Global | -50 to +50 | Asia/Europe overnight, futures |
| Correlation | 0 to 100 | SPX-bond corr, instability (risk) |

## Regime Model

Soft GMM with 5 regimes: Calm, Chop, Tightening Shock, Crisis Trend, Credit Stress.

- 10D feature space: TS, CH, VL, VS, CI, RS, CS, GR, BM_f, BEI
- Expert-calibrated centroids with isotropic Gaussian kernels
- EMA-smoothed transition probabilities (alpha=0.3)
- Shannon entropy for dynamic threshold widening

## Probabilistic Framework

- **Variance**: `sigma2_i = base * f_data * f_extremity * f_agreement * f_regime`
- **Delta-method transform**: `sigma2_norm = sigma2_raw / s^2` (normalized units)
- **Covariance**: `Sigma = rho * (sigma * sigma')`, PSD-enforced
- **Composite variance**: `sigma2_C = w' Sigma w`
- **Confidence**: `P(C>0) = Phi(mu_C / sigma_C)`
- **CVaR gate**: `CVaR_alpha = mu - sigma * phi(z_alpha) / alpha`, threshold theta=0.30
- **Half-Kelly**: `f* = mu / (lambda * sigma2_C)`, lambda=2.0, regime-capped

## Interaction Term

Regime-conditioned: `A(r) = sum_k pi_k(r) * A_k`

6 economically motivated pairs across 5 regimes. Spectral bound enforced at 0.05.

## Meta-Learning

Exponentiated gradient with inertia (lambda=0.70):
- `w' = softmax(log(w) + eta * Q)`
- Floors [0.04, 0.30], re-normalized
- Per-ticker state isolation (atomic POSIX writes)
- Cold-start bypass for first 10 runs

## Risk Governor

```
SR_s = 0.5*(CI+CS) + 0.3*RS + 0.2*BEI    (structural)
SR_t = 0.4*|TS| + 0.3*BM_f + 0.15*VS     (tactical)
SR   = alpha*SR_s + (1-alpha)*SR_t         (dynamic blend)
G    = 1 - 1/(1 + exp(-(SR-tau)/s))       (sigmoid gate)
```

Quadratic fade below g_floor=0.35. Crisis Trend floor at 0.30.

## Smooth Verdict

Sigmoid decision mapping with dynamic threshold:

```
tau_eff = tau_base * regime_mult * (1 + k1*sigma_C) * (1 + k2*H_entropy)
P_buy = 1 / (1 + exp(-C_adj / tau_eff))
```

TQ gate: `TQ < 0.12 -> CASH / STAND ASIDE` (overrides sigmoid).

## File Map

| File | Purpose |
|------|---------|
| `bot.py` | Slack entry point, event handlers |
| `atlas_engine.py` | Core 11-layer engine |
| `data_fetcher.py` | yfinance live data fetch |
| `v8_data.py` | Extended data + Monte Carlo DCF |
| `v8_report.py` | 11-section Slack report formatter |
| `valuation_config.py` | Governance config (single source of truth) |
| `web_report.py` | SQLite report persistence |
| `web_server.py` | FastAPI dashboard |
| `gemini_qa.py` | LLM Q&A (Groq, lazy-loaded) |
| `test_data_integrity.py` | Regression tests |

## Config

- `valuation_config.py`: Valuation governance (beta, WACC, MOS, fragility, Monte Carlo)
- `atlas_engine.py ATLAS_CONFIG`: Engine-specific (normalization, meta-learning, decision)
- `.env`: Tokens (SLACK_BOT_TOKEN, SLACK_APP_TOKEN, GROQ_API_KEY, FRED_API_KEY, CAPITAL)

## Import Graph

```
bot.py
  -> atlas_engine    (numpy, stdlib)
  -> data_fetcher    (yfinance, numpy)
  -> v8_data         (data_fetcher, valuation_config)
  -> v8_report       (valuation_config)
  -> web_report      (sqlite3, stdlib)
  -> gemini_qa       (groq lazy) [lazy-imported]
```

Clean DAG. No circular dependencies. Max depth: 2.
