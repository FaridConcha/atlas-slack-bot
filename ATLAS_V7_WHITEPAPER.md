# ATLAS V7 Technical Whitepaper

## Systematic Trading Analysis Engine — Architecture, Methodology, and Output Specification

---

## 1. EXECUTIVE SUMMARY

ATLAS is an institutional-grade systematic trading analysis engine that processes 7 categories of market data through an 11-layer hierarchical framework to produce actionable trading intelligence. The system ingests live price, fundamental, consensus, volatility, macro, breadth, and global data, then synthesizes it into a single composite signal with regime-aware risk management and adaptive weight optimization.

The engine outputs a 4-message trader desk note delivered via Slack, combining quantitative rigor with the readability of a senior trader's morning brief.

**Key Characteristics:**
- 8 independent scoring engines normalized through tanh compression
- 10-feature regime vector for market environment classification
- Sigmoid-gated risk governor with structural and tactical risk decomposition
- Exponentiated gradient meta-learning for adaptive engine weighting
- Regime-conditioned portfolio policy with 5 allocation states
- Execution microstructure layer with mode-adaptive position sizing

---

## 2. SYSTEM ARCHITECTURE

```
Layer 0:  Data Integrity Check          → Data Confidence (DC: 0-100%)
Layer 1:  Regime Vector                 → 10 features: TS, CH, VL, VS, CI, RS, CS, GR, BM_f, BEI
Layer 2:  Score Normalization           → 8 engines: tanh(raw / scale)
Layer 3:  Meta-Regime Learning          → Adaptive base weights w0 via exponentiated gradient
Layer 4:  Dynamic Weight Matrix         → w(t) = softmax(log(w0) + alpha * A @ r_bar)
Layer 5:  Composite Score               → C_raw = 100 * sum(w_i * e_norm_i)
Layer 6:  Risk Governor                 → Gate G(t) via sigmoid on SR
Layer 7:  Trade Quality Gate            → TQ = |C_raw|/100 * Rel * G * DC/100
Layer 8:  Portfolio Meta Policy         → Regime-conditioned allocation + exposure multiplier
Layer 9:  Execution Microstructure      → Mode, entry, stop, TP, execution gate
Layer 10: Pyramid Report                → Raw text report (internal)
Layer 11: Trader Abstraction (V7)       → 4 Slack messages (presentation layer)
```

### Data Flow

```
Yahoo Finance / FRED APIs
    ↓
7 data files (OHLCV, fundamentals, consensus, volatility, macro, breadth, global)
    ↓
data_fetcher.py  →  atlas_engine.py (Layers 0-10)  →  message_formatter.py (Layer 11)
    ↓
summary dict (~30 fields)
    ↓
4 Slack messages
```

---

## 3. DATA INPUTS

| Source | File | Contents | Update Frequency |
|--------|------|----------|-----------------|
| yfinance | ohlcv.csv | 1yr daily OHLCV bars | Per request |
| yfinance | fundamentals.json | PE, EV/EBITDA, FCF yield, margins, debt/equity, historical multiples | Per request |
| yfinance | consensus.json | Analyst ratings, revisions, target prices, earnings surprises | Per request |
| yfinance | volatility.csv | VIX, VIX3M, put/call ratio | Per request |
| yfinance/FRED | macro_rates.csv | US 10Y, 2Y, 2s10s spread, HY spread, real yield | Per request |
| yfinance | breadth.csv | Advancing/declining, new highs/lows, % above 50d/200d MAs | Per request (approximated from SPY) |
| yfinance | global_overnight.json | Nikkei, Hang Seng, DAX, FTSE, Stoxx, ES/NQ futures, USD/JPY, EUR/USD | Per request |

---

## 4. THE 8 SCORING ENGINES (Layer 2)

Each engine processes its domain of market data and produces a raw score, which is then normalized through tanh compression to the [-1, +1] range.

### 4.1 Trend Engine
**Raw Range:** -100 to +100
**Normalization Scale:** 100

**Inputs:** OHLCV prices, volume, breadth

**Scoring Components:**
| Component | Max Points | Logic |
|-----------|-----------|-------|
| Price > SMA20 | +20 | Immediate momentum |
| Price > SMA50 | +30 | Intermediate trend |
| Price > SMA200 | +20 | Long-term trend |
| MA alignment (20>50>200) | +15 / -15 | Trend structure confirmation |
| SMA20 slope (5-period polyfit) | +/-30 | Momentum acceleration |
| 5-day breakout | +10 to +20 | New highs detection |
| Volume confirmation | +10 / -5 | Volume validates price |
| Breadth confirmation | +10 / -10 | Market participation |

**Interpretation:** Positive score = bullish trend structure with momentum. Negative = broken trend, momentum negative.

### 4.2 Valuation Engine
**Raw Range:** -40 to +40 (intentionally narrow — valuation is a weak timing signal)
**Normalization Scale:** 40

**Inputs:** Trailing PE, EV/EBITDA, FCF yield (each with historical series), real yield

**Scoring Components:**
| Component | Max Points | Logic |
|-----------|-----------|-------|
| PE z-score vs history | +/-20 | Current PE relative to 1yr average |
| EV/EBITDA z-score | +/-12 | Enterprise value multiple relative to history |
| FCF yield vs average | +/-8 | Free cash flow attractiveness |
| Real yield penalty | -10 | High real rates compress valuations |

**Interpretation:** Positive = cheap relative to history. Negative = stretched multiples.

### 4.3 Consensus Engine
**Raw Range:** -50 to +50
**Normalization Scale:** 50

**Inputs:** Analyst up/down revisions, earnings surprises, target price, buy/hold/sell ratings

**Scoring Components:**
| Component | Max Points | Logic |
|-----------|-----------|-------|
| Revision ratio (up vs down) | +/-30 | >65% up = bullish, <35% = bearish |
| Earnings surprise streak | +/-25 | 3 consecutive beats or misses |
| Target price premium | +/-20 | Consensus upside/downside |
| Crowdedness penalty | -15 | >85% buy = sentiment extreme |

**Interpretation:** Positive = street is getting more bullish, revisions trending up. Negative = analysts cutting numbers.

### 4.4 Volatility Engine
**Raw Range:** -50 to +30
**Normalization Scale:** 50

**Inputs:** VIX (current and 5-day prior), VIX3M, put/call ratio, 20-day realized vol

**Scoring Components:**
| Component | Max Points | Logic |
|-----------|-----------|-------|
| VIX regime level | +15 to -35 | <15 = opportunity, >30 = extreme caution |
| VIX declining from high | +15 | Recovering vol = healthier |
| Term structure (VIX vs VIX3M) | +5 to -20 | Near-term > long-term = stress |
| Put/call extremes | +/-20 | >1.3 = excess fear (contrarian), <0.5 = complacency |

**Interpretation:** Positive = calm, trending-friendly environment. Negative = elevated uncertainty, wider ranges.

### 4.5 Macro Engine
**Raw Range:** -50 to +30
**Normalization Scale:** 50

**Inputs:** US 10Y yield (current, 5d, 20d), 2Y yield, real yield, HY spread (current, 5d)

**Scoring Components:**
| Component | Max Points | Logic |
|-----------|-----------|-------|
| 5-day yield shock | -20 | >20bp move = significant tightening |
| Real yield level | +10 to -15 | >2% = headwind, <0% = tailwind |
| Yield curve slope | +15 to -20 | Inversion = recession signal, steep = growth |
| Credit spread momentum | +10 to -30 | 5-day widening >50bp = severe |

**Interpretation:** Positive = stable rates, benign credit. Negative = rate shock or credit stress.

### 4.6 Liquidity Engine
**Raw Range:** -50 to +50
**Normalization Scale:** 50

**Inputs:** Volume (current and 20d MA), advancing/declining, new highs/lows, % above 50d/200d MAs

**Scoring Components:**
| Component | Max Points | Logic |
|-----------|-----------|-------|
| Volume surge ratio | +20 / -15 | >2x average = conviction, <0.5x = thin |
| Breadth extremes | +30 / -15 | >80% advancing = broad, <35% = narrow |
| New highs vs lows | +/-15 | >2:1 ratio = healthy, <0.5:1 = deteriorating |
| Market participation (% > 50d MA) | +/-10 | >80% = broad, <35% = thin |

**Interpretation:** Positive = healthy breadth, broad participation. Negative = narrow leadership, fragile.

### 4.7 Global Overnight Engine
**Raw Range:** -50 to +50
**Normalization Scale:** 50

**Inputs:** Nikkei, Hang Seng, DAX, FTSE, Stoxx returns; ES/NQ futures overnight; USD/JPY, EUR/USD

**Scoring Components:**
| Region | Max Points | Logic |
|--------|-----------|-------|
| Asia (Nikkei + Hang Seng) | +/-20 | >0.5% move = significant |
| Europe (DAX + FTSE) | +/-16 | >0.5% move = significant |
| US Futures (ES + NQ) | +/-25 | >0.3% overnight = directional |
| VIX multiplier | 0.6x to 1.3x | High VIX amplifies global signals |

**Interpretation:** Positive = risk-on overnight, supportive flows. Negative = risk-off, caution.

### 4.8 Correlation Engine
**Raw Range:** 0 to 100 (RISK SCORE — inverted for composite)
**Normalization Scale:** 50

**Inputs:** SPX daily returns, US 10Y yield changes, VIX, HY spread (60-day windows)

**Scoring Components:**
| Component | Max Points | Logic |
|-----------|-----------|-------|
| Base risk | 50 | Neutral starting point |
| High corr + negative market | +20 | Flight-to-safety breakdown |
| Correlation instability | +15 | 20d rolling corr deviation > 0.3 |
| Credit stress context | +10 | HY spread > 350bp |
| Vol stress context | +10 | VIX > 25 |

**Inversion for composite:** `corr_signal = -(corr_risk - 50) / 50`

**Interpretation:** High risk score = unstable cross-asset relationships. Used defensively — when correlations break, all other signals become less reliable.

---

## 5. REGIME VECTOR (Layer 1)

The regime vector is a 10-dimensional feature vector that characterizes the current market environment. Each feature is normalized to [0, 1] (except TS which is [-1, 1]).

| Feature | Name | Measurement | Formula |
|---------|------|-------------|---------|
| TS | Trend Score | Normalized trend direction | trend_score / 100 |
| CH | Choppiness | ATR-to-range ratio | sum(ATR_14d) / (H14 - L14) / 3.0 |
| VL | Vol Level | Normalized VIX level | (VIX - 12) / 40 |
| VS | Vol Stress | VIX momentum above mean | max(0, VIX - VIX_20ma) / VIX_20ma |
| CI | Corr Instability | Rolling correlation std | std(rolling_corr_20d over 60d) |
| RS | Rates Shock | 5-day yield magnitude | abs(10Y_now - 10Y_5d) / 0.50 |
| CS | Credit Stress | Spread premium | (HY_spread - 300) / 400 |
| GR | Global Risk | ES overnight magnitude | abs(ES_overnight) / 1.5 |
| BM_f | Bad Mix Frequency | VIX>25 AND breadth<40% | count_bad_days / 20 |
| BEI | Bond-Equity Flip | Correlation sign change | 1.0 if sign_flip else 0.0 |

### Regime Classification (Priority Order)

```
1. Credit Stress:      CS > 0.6
2. Crisis Trend:       |TS| > 0.7 AND (VS > 0.7 OR BEI > 0.8)
3. Chop:               CH > 0.6
4. Tightening Shock:   RS > 0.6 AND CS < 0.4
5. Calm:               (default)
```

### Regime Reliability

```
Rel = (1 - CH)^1.2 * (1 - VS)^1.0 * (1 - CI)^0.8
Rel = clip(Rel, 0.05, 1.0)
```

Low choppiness + low vol stress + stable correlations = high reliability. This multiplicative structure means ANY single factor being extreme can suppress reliability significantly.

---

## 6. ADAPTIVE WEIGHT SYSTEM (Layer 3 + Layer 4)

### Layer 3: Meta-Learning (Exponentiated Gradient)

The system learns which engines perform best over time using a utility-based gradient update.

**Cold Start (runs 0-9):** All weights = 1/8 (equal). No learning until the model has 10 data points.

**Utility Computation (runs 10+):**
```
Q[engine] = |e_norm[engine]| * (1 - CH) * 0.5 + |e_norm[engine]| * Rel * 0.5
```
Rewards engines producing strong signals in non-choppy, reliable environments.

**Weight Update:**
```
eta = 0.15 (learning rate)
lambda = 0.70 (inertia)

logits[engine] = log(max(w0[engine], 0.01)) + eta * Q[engine]
w0_prime = softmax(logits)
w0_new = lambda * w0_prev + (1 - lambda) * w0_prime

Bounds: floor = 0.04, ceiling = 0.30
Final: w0_new = w0_bounded / sum(w0_bounded)
```

State persists across runs via `state/atlas_meta_state.json`.

### Layer 4: Dynamic Weights (Affinity Matrix)

Real-time regime-conditioned weight adjustment:

```
alpha = 0.25 (affinity sensitivity)
A = 8x10 affinity matrix (predefined)
r_bar = regime_vector - mean(regime_vector)

w_dynamic = softmax(log(w0) + alpha * A @ r_bar)
```

The affinity matrix encodes which engines benefit from which regime features. For example, the trend engine has high affinity with TS (trend score) and CH (choppiness) — it gets more weight in trending, non-choppy environments.

---

## 7. COMPOSITE SCORE AND RISK GOVERNOR (Layers 5-6)

### Layer 5: Composite Score

```
C_raw = 100 * sum(w_dynamic[i] * e_norm[i])  for i in 8 engines

Direction:
  C_raw > 0  → LONG
  C_raw < 0  → SHORT
  C_raw = 0  → FLAT
```

### Layer 6: Risk Governor

**Structural Risk (long-term):**
```
SR_s = 0.5*(CI + CS) + 0.3*RS + 0.2*BEI
```

**Tactical Risk (short-term):**
```
SR_t = 0.4*|TS| + 0.3*BM_f + 0.3*(VS/2.0)
```

**Combined Risk:**
```
SR = 0.6*SR_s + 0.4*SR_t
```

**Gate Function (sigmoid):**
```
tau = 0.40 (risk threshold)
s = 0.15 (sharpness)

G = 1.0 - 1.0 / (1.0 + exp(-(SR - tau) / s))
```

When SR < tau, the gate is near 1.0 (fully open). As SR exceeds tau, the gate closes toward 0 (position suppressed). The sigmoid creates smooth, continuous gating rather than a binary cutoff.

**Adjusted Composite:**
```
C_adjusted = C_raw * G * (DC / 100)
```

---

## 8. TRADE QUALITY AND DECISION RULES (Layer 7)

### Trade Quality Score

```
TQ = (|C_raw| / 100) * Rel * G * (DC / 100)
TQ = clip(TQ, 0, 1)
```

TQ is the product of four factors:
- **Signal strength** (|C_raw| / 100): How strong is the directional lean?
- **Regime reliability** (Rel): Can we trust the current environment?
- **Risk gate** (G): Is systemic risk acceptable?
- **Data confidence** (DC / 100): Do we have good data?

All four must be reasonable for TQ to be meaningful. Any single factor near zero collapses the entire score.

### Verdict Decision Rules

| Condition | Verdict |
|-----------|---------|
| TQ < 0.12 | CASH — no edge, stay flat |
| Rel < 0.25 | CASH — can't read the market |
| Gate < 0.35 | CASH — systemic risk too high |
| 0.12 <= TQ < 0.20 | WAIT — lean forming, not enough conviction |
| TQ >= 0.20, C_raw > 0 | BUY — signals aligned long |
| TQ >= 0.20, C_raw < 0 | SELL — signals aligned short |

---

## 9. POSITION SIZING (Layer 8)

### Regime-Conditioned Portfolio Policy

Each regime maps to a probability distribution over 5 allocation states:

| Regime | CASH | SMALL | NORMAL | LARGE | HEDGE |
|--------|------|-------|--------|-------|-------|
| Calm | 5% | 10% | 60% | 20% | 5% |
| Chop | 20% | 25% | 40% | 10% | 5% |
| Tightening Shock | 15% | 15% | 50% | 10% | 10% |
| Crisis Trend | 40% | 15% | 20% | 5% | 20% |
| Credit Stress | 35% | 20% | 25% | 10% | 10% |

### Exposure Multiplier

| State | Multiplier |
|-------|-----------|
| CASH | 0.0 |
| SMALL | 0.20 |
| NORMAL | 0.50 |
| LARGE | 1.00 |
| HEDGE | -0.50 |

```
mu = sum(policy[state] * multiplier[state])
```

### Risk Budget

```
b = 0.01 * capital * mu
```

### Execution Gate (Layer 9)

```
Mode selection:
  Calm + C_adjusted > 10  → Momentum (penalty: 0.02)
  Chop                    → MeanRev (penalty: 0.05)
  Otherwise               → Confirm (penalty: 0.04)

exec_pen = base_penalty + VS * 0.10
liq_win = clip(0.5 + vol_ratio * 0.4, 0.5, 1.5)
haz_open = clip(0.02 + VS * 0.15, 0.02, 0.20)

g_exec = (1 - exec_pen) * liq_win * (1 - haz_open)

size_final = b * g_exec
```

---

## 10. TRADE LEVEL DETERMINATION (Layer 9)

### Entry

```
entry = SMA50 * 1.003  (0.3% above intermediate support)
```

### Stop Loss (dual constraint)

```
atr_stop = price - (ATR_14 * 2.0)       # Volatility-based
structural_stop = SMA200 * 0.97          # 3% below 200-day MA
stop_loss = min(atr_stop, structural_stop)  # Use tighter
```

### Take Profit

```
52w_high = max(closes over 252 days)
tp_low = 52w_high * 0.98    # Near target
tp_high = 52w_high * 1.05   # Aggressive extension
```

### Risk/Reward

```
entry_mid = (entry_lo + entry_hi) / 2
tp_mid = (tp_low + tp_high) / 2
R:R = |tp_mid - entry_mid| / |entry_mid - stop_loss|
```

### Buy Zone (structural support)

```
buy_zone_lo = SMA200 * 0.97
buy_zone_hi = SMA200
```

---

## 11. V7 OUTPUT SPECIFICATION (Layer 11)

Layer 11 is a read-only presentation layer that transforms the engine's summary dict into 4 Slack messages. It never modifies engine math — only formats and explains results.

### Message 1: VERDICT + THE BOTTOM LINE

**Verdict Card (4 lines):**
- Verdict (BUY/SELL/WAIT/CASH) | Bias (Bullish/Neutral/Bearish) | Risk Environment
- Conviction label + TQ score | Readability label + Rel score
- Position size in $ and % of capital | Execution mode
- Risk gate value | Data confidence %

**Narrative (3-4 sentences):**
- Price vs moving average context (exact MA values)
- Regime environment description
- Verdict justification with TQ
- Earnings/valuation context from consensus + valuation scores

**Quick Reference Strip:**
- SMA20, SMA50, SMA200, ATR(14), VIX — one line for instant cross-reference

### Message 2: WHAT'S DRIVING THIS + ENGINE SCOREBOARD

**Driver Narratives (top 2 positive, top 2 negative by contribution magnitude):**
Each driver gets a context-aware paragraph with actual data values (e.g., "VIX at 14.9 signals calm...").

Drivers are selected when contribution > 0.5 (positive) or < -0.5 (negative).

**Engine Scoreboard Table (all 8 engines):**
```
Engine         Score     Wt   Contrib
trend          +45.0   .148    +5.82
valuation      -12.0   .098    -1.02
consensus      +28.0   .132    +3.21
volatility     +10.0   .120    +1.04
macro           -5.0   .118    -0.51
liquidity      +22.0   .135    +2.59
global         +15.0   .122    +1.59
correlation     50.0   .127    -0.01
                       -----  ------
TOTAL                  1.000  +12.7
```

**Weight Shift Line:**
Shows the engine with the largest adaptive weight change from base (if meta-learning active).

### Message 3: TRADE PLAN (LEVELS + RISK + TRIGGERS)

**Trade Levels (conditional on verdict):**

For BUY/SELL:
- Entry zone (ATR-based range)
- Buy zone (200-day structural support)
- Stop loss with dual context (ATR-based + structural)
- Target 1 + stretch target
- Risk/reward ratio

For WAIT:
- Watch levels (50d, 200d MAs) with directional context
- Buy zone if triggered

For CASH:
- Key levels to watch (MAs + ATR daily range)

**Risk Check (expanded):**
- Gate | TQ | Conviction label
- Structural risk (SR_s) | Tactical risk (SR_t)
- Regime | Reliability
- Risk driver warnings (from engine)
- Contradiction alerts (from self-audit)
- Saturation warnings (regime inputs at extremes)
- Calibration notice (if run_count < 10)

**What Would Change My Mind (3-5 triggers):**
Specific, measurable conditions with current values:
- Price-based (MA reclaims/breaks)
- Model-based (TQ thresholds, readability)
- Regime-based (choppiness, VIX, credit)
- Earnings catalysts

**Conditional Trigger Line:**
Binary condition format:
- WAIT: `BUY IF price closes above $XX AND TQ > 0.20 AND gate > 0.50`
- CASH: `RE-EVALUATE IF price reclaims $XX AND TQ > 0.20 AND readability > 0.25`
- BUY: `EXIT IF price breaks below $XX AND gate < 0.40`
- SELL: `COVER IF price reclaims $XX AND trend > +0.30`

### Message 4: MODEL INTERNALS

**Composite Decomposition:**
- Raw → Adjusted signal flow
- Data confidence + learning status + run count
- Execution mode with human explanation

**Regime Vector (full 10-feature display):**
- All 10 features with numeric values
- Human-readable annotations for extreme values only (>0.70 or <0.15)

**Weight Matrix (if meta-learning active, run >= 10):**
- Base w0 vs dynamic w(t) with shift for all 8 engines

---

## 12. SUPPRESSION RULES

| Condition | Suppressed Content |
|-----------|-------------------|
| Verdict = CASH | Entry zone, stop loss, take profit, R:R, execution mode details |
| Verdict = WAIT | Trade levels (shows watch levels instead) |
| TQ < 0.12 | Position sizing (shows $0) |
| Readability < 0.25 | Directional bias confidence |
| No dominant drivers | Driver narratives (shows "no dominant driver" message) |
| run_count < 10 | Weight matrix (shows "learning begins at run 10") |

---

## 13. CONFIGURATION PARAMETERS

| Parameter | Value | Layer | Purpose |
|-----------|-------|-------|---------|
| Normalization scales | 40-100 | 2 | Tanh compression per engine |
| Affinity alpha | 0.25 | 4 | Regime-weight sensitivity |
| Meta-learning eta | 0.15 | 3 | Gradient step size |
| Meta-learning lambda | 0.70 | 3 | Weight inertia |
| Weight floor/ceiling | 0.04 / 0.30 | 3 | Prevents engine dominance |
| Risk threshold (tau) | 0.40 | 6 | Sigmoid gate center |
| Sigmoid sharpness (s) | 0.15 | 6 | Gate transition speed |
| TQ thresholds | 0.12 / 0.20 / 0.35 | 7 | CASH/WAIT/BUY boundaries |
| Reliability threshold | 0.25 | 7 | Minimum readability for trade |
| Gate threshold | 0.35 | 7 | Minimum gate for trade |
| ATR stop multiple | 2.0x | 9 | Stop distance from price |
| Structural stop | 3% below SMA200 | 9 | Long-term support buffer |

---

## 14. DATA CONFIDENCE (Layer 0)

```
DC starts at 100%

Deductions:
  Price data < 200 bars:    -40%
  Macro data < 5 days:      -25%
  Vol data < 2 days:        -20%
  No fundamentals:          -10%
  No consensus:             -10%
  No global overnight:       -5%

DC = clip(DC, 0, 100)
```

DC is used as a multiplicative cap on the adjusted composite and TQ. If major data is missing, the model automatically reduces confidence and position sizing.

---

## 15. SELF-AUDIT SYSTEM

The engine includes a self-audit layer that detects:

**Contradictions:** Internal inconsistencies such as:
- Bullish composite but bearish trend
- High TQ but low reliability
- Large position but high risk gate

**Weight Concentration:** Flags when any single engine weight exceeds normal bounds, indicating potential over-reliance.

**Regime Stability:** Monitors reliability score changes that could indicate rapid environment shifts.

These are surfaced in Message 3 as warnings to the user.

---

## 16. VERSION HISTORY

| Version | Changes |
|---------|---------|
| V1-V3 | Initial builds, basic Slack output |
| V4 | Complete 11-layer engine rewrite from mathematical specification |
| V5 | Added Layer 11 trader abstraction with decision rules and suppression |
| V6 | Narrative rewrite: _build_narrative(), _expand_driver(), context-aware triggers, 4-message format |
| V7 (current) | Middle-ground format: engine scoreboard, buy zones, dual stops, R:R, risk decomposition, conditional triggers, regime annotations, weight matrix, execution mode explanations |

---

## DISCLAIMER

ATLAS is a systematic model output, NOT financial advice. The system is designed for analytical purposes and should be used as one input among many in a trading decision process. Always do your own due diligence before trading. Past model performance does not guarantee future results.
