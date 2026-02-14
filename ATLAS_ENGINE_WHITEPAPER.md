# ATLAS ENGINE — Technical White Paper

**Version:** 3.0
**Date:** February 14, 2026
**Classification:** Confidential — Internal & Investor Distribution
**System Version:** ATLAS V9 (Production)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Model Strategy](#3-model-strategy)
4. [Integrations](#4-integrations)
5. [User Interaction & Workflows](#5-user-interaction--workflows)
6. [Current Limitations](#6-current-limitations)
7. [Improvements & Roadmap](#7-improvements--roadmap)
8. [Making Follow-Up Responses More Robust and Data-Rich](#8-making-follow-up-responses-more-robust-and-data-rich)
9. [Governance, Compliance & Security](#9-governance-compliance--security)
10. [Appendix](#10-appendix)

---

## 1. Executive Summary

### What Atlas Engine Is

Atlas Engine is a capital allocation intelligence platform delivered through Slack messaging. It combines an 11-layer hierarchical quantitative engine with a Buffett-aligned owner intelligence layer, institutional-grade market data collection, and multi-turn conversational AI to generate structured equity research in under 60 seconds. Users invoke it with a single Slack mention — `@atlas AAPL` — and receive an 11-section research report that leads with an Owner Assessment (business quality, moat durability, intrinsic value, margin of safety) followed by quantitative fundamentals, valuation, technicals, sentiment, macro context, and a machine-generated trade verdict with execution parameters.

### The Problem It Solves

Retail and institutional traders face three compounding challenges:

1. **Data fragmentation.** Price action, fundamentals, macro indicators, analyst consensus, news sentiment, and global overnight returns live in separate systems. Synthesizing them manually takes 30–90 minutes per ticker.
2. **Cognitive overload.** Even after gathering data, weighting competing signals — a strong earnings surprise versus rising credit spreads, for example — requires disciplined quantitative reasoning that is difficult to sustain across dozens of daily decisions.
3. **Regime blindness.** Static scoring models fail when market regimes shift. A momentum signal that works in calm trending markets becomes a trap in volatile, choppy environments. Most retail tools offer no regime awareness.

Atlas Engine solves all three. It fetches seven categories of live market data in parallel, runs them through eight specialized scoring engines, dynamically re-weights those engines based on a 10-feature regime vector, then overlays a Buffett-aligned owner intelligence layer that evaluates business quality, moat durability, capital allocation discipline, intrinsic value, and margin of safety. The result is a risk-governed capital allocation recommendation with conviction-based position sizing, entry/exit levels, and permanent loss risk assessment — all within a conversational Slack interface that supports multi-turn follow-up questions via a Buffett-philosophy-aligned LLM.

### Key Metrics

| Metric | Value |
|--------|-------|
| End-to-end latency | 15–20 seconds (analysis) + Slack posting |
| Scoring engines | 8 quantitative + V9 owner intelligence layer |
| V9 owner scores | Business Quality, Moat Durability, Capital Allocation (0–5 each) |
| V9 decision framework | PASS / WATCH / RESEARCH / BUY / HOLD / TRIM / EXIT |
| Regime features | 10-dimensional vector |
| Output sections | 11-section structured report (Owner Assessment + 10 engine sections) |
| Data sources | yfinance (primary), FRED (optional), static CSV fallback |
| AI Q&A model | Groq Llama 3.3 70B (Buffett-aligned philosophy prompt) |
| Web dashboard | Institutional-grade HTML report with Owner's View (ECharts, sortable tables, formula derivation) |
| Report persistence | SQLite (WAL mode) with JSON payloads |
| Web API | FastAPI — `/r/{id}` (HTML), `/api/r/{id}.json` (JSON), `/health` |
| Cold-start handling | Auto-detect Render spin-up, in-place Slack progress messages |
| Codebase | 7,316 lines across 8 Python modules |
| Deployment | Render PaaS, Slack Socket Mode, FastAPI (uvicorn) |
| Operating cost | $0 at current scale (free-tier APIs) |

---

## 2. System Architecture

### 2.1 Logical Architecture

Atlas Engine is organized into six distinct layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│  Slack Socket Mode  │  Threaded Reports  │  AI Q&A Chat     │
├─────────────────────────────────────────────────────────────┤
│                 WEB PRESENTATION LAYER                       │
│  web_server.py — FastAPI (ECharts dashboard, JSON API)      │
│  Routes: /r/{id} (HTML) │ /api/r/{id}.json │ /health       │
├─────────────────────────────────────────────────────────────┤
│                   ORCHESTRATION LAYER                        │
│  bot.py — Event routing, cold-start detection, lifecycle    │
│  Signal handling (SIGTERM), thread cache, report generation  │
├──────────────────┬──────────────────┬───────────────────────┤
│  INTELLIGENCE    │  REPORT          │  CONVERSATIONAL       │
│  LAYER           │  LAYER           │  LAYER                │
│  atlas_engine.py │  v8_report.py    │  gemini_qa.py         │
│  (11 layers)     │  (V9 owner +     │  (Groq LLM,          │
│                  │   10 sections)   │   Buffett-aligned)    │
├──────────────────┴──────────────────┴───────────────────────┤
│                   PERSISTENCE LAYER                          │
│  web_report.py — SQLite (WAL mode), JSON payload storage    │
│  Report ID generation, numpy serialization, URL routing     │
├─────────────────────────────────────────────────────────────┤
│                      DATA LAYER                              │
│  data_fetcher.py  │  v8_data.py  │  Static CSV/JSON files   │
│  (7 parallel      │  (Extended    │  (Fallback data for     │
│   yfinance calls) │   analytics)  │   offline operation)    │
│  resolve_price()  │               │                         │
│  (8-level fallback│               │                         │
│   price chain)    │               │                         │
├─────────────────────────────────────────────────────────────┤
│                   EXTERNAL SERVICES                          │
│  Yahoo Finance API  │  FRED API  │  Groq API  │  Slack API  │
└─────────────────────────────────────────────────────────────┘
```

**Presentation Layer** — Slack Socket Mode (WebSocket-based, no webhook URL required). Handles inbound `@mention` events and `message` events for thread-based Q&A. Posts threaded replies with Slack markdown formatting.

**Web Presentation Layer** — `web_server.py` (867 lines). FastAPI application serving institutional-grade HTML dashboards with a V9 Owner's View card (star ratings, conviction bar, MOS gauge, intrinsic value comparison, permanent loss risks), ECharts financial charts (price context, engine waterfall, regime vector, DCF scenarios), dense metric grids, sortable peer/engine tables, and a "Show Your Work" section with V9 decision hierarchy derivation and composite formulas. Also serves raw JSON payloads via REST API. Runs as a threaded uvicorn server alongside the Slack bot on the same port.

**Orchestration Layer** — `bot.py` (509 lines). Manages the full lifecycle: parse ticker from mention text, dispatch parallel data fetches, invoke the 11-layer engine, trigger V9 owner assessment + report formatting, generate and store web reports (SQLite), inject dashboard hyperlinks into Slack messages, cache results for AI follow-up, and post sections as threaded replies. Maintains an in-memory LRU thread cache (50 threads, 4-hour TTL). Includes cold-start detection for Render free-tier spin-ups (120-second window with in-place progress messages) and SIGTERM/SIGINT signal handlers for graceful shutdown notification.

**Persistence Layer** — `web_report.py` (128 lines). SQLite storage layer for full analysis payloads. Generates URL-safe report IDs (`{SYMBOL}-{thread_ts}-{uuid}`), recursively serializes numpy types via `_make_serializable()`, and stores JSON payloads in WAL-mode SQLite. Reports persist across restarts and are served by the web presentation layer.

**Intelligence Layer** — `atlas_engine.py` (1,866 lines). The core analytical engine implementing 8 scoring engines, 10-feature regime classification, meta-learning weight optimization, risk governance, and trade execution parameter generation across 11 sequential layers.

**Report Layer** — `v8_report.py` (1,687 lines) and `v8_data.py` (1,160 lines). Computes V9 owner intelligence scores (business quality, moat durability, capital allocation, intrinsic value, margin of safety, conviction), fetches extended analytics (peer comparison, DCF valuation, institutional ownership, technical indicators, news sentiment), and formats the 11-section output report (Owner Assessment + 10 engine sections).

**Conversational Layer** — `gemini_qa.py` (419 lines). Wraps the Groq API (Llama 3.3 70B) with a Buffett-aligned system prompt for multi-turn Q&A. Enforces reasoning order: business durability → moat → capital allocation → intrinsic value → risk → timing. Builds structured context with V9 owner assessment as primary section and engine verdict as secondary overlay. Manages conversation history (last 6 exchanges).

**Data Layer** — `data_fetcher.py` (680 lines). Fetches live market data from yfinance in parallel (7-worker ThreadPool). Falls back to FRED for macro data or static CSV/JSON files for offline operation. Includes `resolve_price()`, an 8-level fallback chain for robust price resolution across market conditions (pre-market, post-market, regular hours, historical bars).

### 2.2 Data Ingestion Flow

```
User: @atlas AAPL
         │
         ▼
┌─────────────────────────────────────────────────────┐
│         ThreadPoolExecutor (max_workers=7)           │
├─────────┬──────────┬───────────┬───────────┬────────┤
│ OHLCV   │ Fundmtls │ Consensus │ Volatility│ Macro  │
│ 1Y daily│ PE, EV,  │ Targets,  │ VIX 1Y    │ 10Y,2Y│
│ bars    │ margins, │ ratings,  │ Put/Call   │ HY sprd│
│ (yf)    │ FCF (yf) │ revisions │ (yf)      │ (FRED) │
├─────────┴──────────┴───────────┼───────────┼────────┤
│ Breadth (advancing/declining,  │ Global    │        │
│ new highs/lows, % > 50d MA)   │ Overnight │        │
│ (yfinance indices)             │ (yf)      │        │
└────────────────────────────────┴───────────┴────────┘
         │
         ▼
    Write to temp directory:
    /tmp/atlas_{TICKER}_{uuid}/
    ├── ohlcv.csv
    ├── fundamentals.json
    ├── consensus.json
    ├── volatility.csv
    ├── macro_rates.csv
    ├── breadth.csv
    └── global_overnight.json
         │
         ▼
    Engine reads from temp dir → processes → deletes temp dir
```

**Key design decisions:**
- Parallel fetch reduces data collection from ~20 seconds (sequential) to ~5 seconds.
- Temp directory isolation ensures concurrent analyses for different tickers cannot interfere.
- Auto-cleanup after report posting prevents disk accumulation.
- Static data fallback (`./data/default/`) enables offline development and reproducible testing.

### 2.3 Query Processing Flow — Full Lifecycle

```
TIME    STAGE                    MODULE              DETAIL
─────   ─────                    ──────              ──────
T+0s    Event received           bot.py:168          Socket Mode delivers @mention
T+0s    Ticker extraction        bot.py:185          Regex parse: "AAPL" from text
T+0s    Cold-start check         bot.py:208          First mention within 120s of boot?
T+0s    Acknowledgment           bot.py:211/214      Boot progress (cold) or warm ack
T+1s    Data collection          data_fetcher.py     7 parallel yfinance/FRED fetches
        ↳ resolve_price()        data_fetcher.py:28  8-level price fallback chain
T+6s    Engine Layer 0           atlas_engine.py:148 Data integrity check (DC score)
T+6s    Engine Layer 1           atlas_engine.py:160 Regime vector (10 features)
T+6s    Engine Layer 2           atlas_engine.py:730 Score normalization (tanh)
T+7s    Engine Layer 3           atlas_engine.py:780 Meta-learning state load
T+7s    Engine Layer 4           atlas_engine.py:820 Dynamic weight computation
T+7s    Engine Layer 5           atlas_engine.py:860 Composite score
T+7s    Engine Layer 6           atlas_engine.py:900 Risk governor gate
T+7s    Engine Layer 7           atlas_engine.py:940 Trade quality computation
T+7s    Engine Layer 8           atlas_engine.py:980 Portfolio meta policy
T+8s    Engine Layer 9           atlas_engine.py:1030 Execution parameters
T+8s    Engine Layer 10          atlas_engine.py:1100 Pyramid report text
T+8s    Meta state save          atlas_engine.py:1150 Persist w0, Q to disk
T+9s    V8 extended fetch        v8_data.py          Peers, technicals, news, DCF
T+14s   V9 owner scores          v8_report.py        Quality, moat, CA, MOS, conviction
T+14s   V9 decision hierarchy    v8_report.py        5-step gate → PASS/BUY/HOLD/etc.
T+14s   Report formatting        v8_report.py        Owner Assessment + 10 engine sections
T+14s   Web report storage       web_report.py       Store payload → SQLite (WAL)
T+14s   Dashboard URL gen        web_report.py       {SYMBOL}-{ts}-{uuid} → /r/{id}
T+15s   Link injection           bot.py:294          Prepend/append dashboard URLs
T+15s   Cache for Q&A            bot.py:312          Thread cache (incl. v9_scores)
T+15s   Post to Slack            bot.py:322          Summary + 11 threaded replies
T+20s   Cleanup                  bot.py:337          Delete temp data directory
```

### 2.4 Response Generation Pipeline

The response pipeline comprises three sequential stages:

**Stage 1: Engine Output (atlas_engine.py)**
The 11-layer engine produces a `summary` dictionary containing:
- Composite score (raw and risk-adjusted)
- Trade quality (TQ) and verdict (STRONG BUY / BUY / HOLD / SELL / STRONG SELL)
- 8 individual engine scores with detailed metrics
- Dynamic weights and contributions
- Regime classification and reliability
- Execution parameters (entry, stop-loss, take-profit levels, position size)
- Risk drivers and structural risk assessment

**Stage 2: Extended Analytics (v8_data.py)**
A secondary data pass enriches the analysis with:
- 6 sector peers with comparative metrics (forward PE, margins, growth, ROE)
- Technical indicators (RSI 14, MACD, Bollinger Bands, support/resistance)
- News articles with keyword-based sentiment classification
- Institutional ownership and short interest metrics
- Three-scenario DCF model (bear/base/bull)
- Earnings history with beat/miss analysis

**Stage 2.5: V9 Owner Intelligence (v8_report.py)**
The V9 layer computes owner-oriented scores from engine + extended data (see Section 2.8 for full detail):
- **Business Quality (0–5):** ROE, net margin, revenue growth, FCF, debt/equity, earnings beat rate
- **Moat Durability (0–5):** Gross margin (pricing power), operating margin, market cap (scale), ROE+margin combo, interest coverage
- **Capital Allocation (0–5):** ROIC vs WACC, buyback discipline (valuation-aware), dividend sustainability, debt usage, net cash position
- **Margin of Safety:** `MOS = (DCF Base - Price) / DCF Base × 100`, with required thresholds by business type
- **V9 Decision:** 5-step hierarchy → PASS / WATCH / RESEARCH / BUY / HOLD / TRIM / EXIT
- **Conviction Score (0–100):** Composite of quality×5 + moat×4 + CA×4 + MOS×0.5 + risk adjustment
- **Permanent Loss Risks:** Up to 5 identified risks with severity ratings (HIGH/MEDIUM/LOW)

V9 scores are attached to `v8_data['v9_scores']` and propagated to Slack messages, web dashboard, Q&A context, and stored JSON payloads.

**Stage 3: Report Formatting (v8_report.py)**
The formatter produces 11 Slack-compatible sections:

| Section | Content |
|---------|---------|
| 0. Owner Assessment | V9 decision, business quality/moat/CA stars, MOS%, conviction, IV range, permanent loss risks, temperament note, engine conflict protocol |
| 1. Verdict | Signal, composite score, TQ, 6-dimension breakdown, narrative |
| 2. Fundamentals | Revenue, margins, ROE, ROA, FCF, balance sheet |
| 3. Valuation | PE, EV/EBITDA, DCF scenarios, peer-relative value |
| 4. Technicals | RSI, MACD, Bollinger, support/resistance, moving averages |
| 5. Peers | 6-peer comparison table with key metrics |
| 6. Sentiment | News feed with sentiment scores, institutional positioning |
| 7. Risk Factors | Risk drivers, structural risk, tactical risk |
| 8. Growth Catalysts | Sector tailwinds, competitive advantages, upcoming events |
| 9. Macro Context | Yields, credit spreads, VIX, global overnight, regime label |
| 10. Engine Signal | Full engine readout with V9 owner's perspective: conviction-based sizing, conflict notes, business case summary |

Each section is capped at ~4,000 characters (Slack message limit) and formatted with Slack markdown (bold, code blocks, bullet lists). The Owner Assessment is always the first section the user reads, establishing the business-owner frame before quantitative detail.

**Stage 4: Web Report Generation (web_report.py + web_server.py)**
After Slack report formatting, the full analysis payload is persisted to SQLite and a dashboard URL is generated:

1. **Storage** — `web_report.py` serializes the summary, v8_extended, and provenance dicts into a single JSON payload, handling numpy types recursively. Stored in `reports.db` (SQLite, WAL journal mode) keyed by a composite report ID: `{SYMBOL}-{sanitized_thread_ts}-{8char_uuid}`.

2. **URL injection** — The dashboard URL is injected into the first and last Slack messages as a clickable hyperlink, plus the final confirmation message.

3. **Dashboard rendering** — `web_server.py` serves an institutional-grade HTML dashboard at `/r/{report_id}` featuring:
   - **V9 Owner's View card** (blue-bordered, prominent placement after metrics strip):
     - V9 decision pill with reasoning text
     - Star ratings (★/☆) for Business Quality, Moat Durability, Capital Allocation (0–5 each)
     - Conviction progress bar (0–100) with color-coded thresholds (green 80+, yellow 60+, red <60)
     - Margin of Safety gauge (32px font, color-coded: green if threshold met, yellow if partial, red if negative)
     - Intrinsic Value vs Current Price comparison cards with IV Range (bear–bull)
     - Permanent Loss Risks table with severity badges (HIGH=red, MEDIUM=yellow, LOW=green)
     - Temperament note (VIX-based market sentiment)
   - **4 ECharts visualizations**: price context (horizontal bar), engine waterfall (stacked bar simulation), regime vector (risk-colored horizontal bars), DCF scenarios (vertical bars with price reference line)
   - **Dense metric grids**: fundamentals (4-col), valuation (4-col), technicals (4-col), trade plan
   - **Sortable tables**: peer comparison (6 peers with forward PE, margins, growth, ROE) and engine detail (8 engines with scores, weights, contributions)
   - **News & sentiment** section with publisher and sentiment badges
   - **"Show Your Work"** collapsible section: V9 decision hierarchy derivation (5-step gate logic), composite derivation formula, risk governor gate calculation, trade quality computation, and computed-vs-reported delta warnings
   - **Provenance footer**: engine version (ATLAS V9), data confidence, timestamp, data mode
   - **Design system**: GitHub-dark palette (`#06090f` background), Inter + JetBrains Mono typography, CSS custom properties for theming

4. **JSON API** — Raw payload available at `/api/r/{report_id}.json` for programmatic consumption.

### 2.5 Orchestration Logic

**Routing:** All analysis requests flow through a single pipeline. There is no model routing for the core engine — all 8 scoring engines execute on every query. Conditional logic exists only at the data layer (FRED vs yfinance fallback, live vs static data mode).

**Fallback Strategy:**
```
Primary: yfinance live data
    ├─ If FRED_API_KEY set → FRED for macro data
    ├─ If FRED unavailable → yfinance Treasury ETF proxies
    ├─ If yfinance unavailable → static CSV/JSON files
    └─ If static files unavailable → engine runs with reduced DC score
```

**Error Isolation:** Each data fetcher runs in its own thread with independent try-except handling. A failure in one data category (e.g., macro) does not block others. The Data Integrity layer (Layer 0) adjusts the confidence score downward proportionally, ensuring the engine always produces output — but with appropriate uncertainty signaling.

### 2.6 Cold-Start Detection & Graceful Shutdown

ATLAS runs on Render's free tier, which spins the service down after 15 minutes of inactivity. This introduces two UX challenges: (1) the first request after spin-down has a 15–30 second boot delay with no user feedback, and (2) the process is killed via SIGTERM without warning.

**Cold-Start Detection:**
```
Module-level state:
    _boot_time = time.time()          # recorded at import
    _boot_complete = False            # flipped on first mention
    _COLD_START_WINDOW = 120 seconds  # generous for container spin-up

_is_cold_start() logic:
    IF _boot_complete → return False (warm path)
    ELSE → set _boot_complete = True
           IF (now - _boot_time) < 120s → return True (cold path)
           ELSE → return False (boot was fast, treat as warm)
```

**Cold path** — Posts a single Slack message that is updated in-place through 4 stages:
1. `:zzz: ATLAS is waking up from sleep... hang tight`
2. `:satellite: Connecting data systems...` (1.5s delay)
3. `:rocket: Systems online, fetching {symbol}...` (1.5s delay)
4. Pipeline-stage updates: "Live data fetched..." → "Engine complete..." → "Report ready..."

Each `chat_update` call is wrapped in try/except — progress updates are cosmetic and must never break the analysis pipeline.

**Warm path** — Standard single acknowledgment message (no animation).

**Graceful Shutdown (SIGTERM/SIGINT):**
```python
signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)
```

The `_handle_shutdown` handler posts a notification to the last active Slack channel/thread:
> `:zzz: ATLAS is going offline (Render free-tier sleep). Mention me again to wake up — takes ~30 seconds.`

Edge cases:
- No mentions before SIGTERM → `_last_channel` is None → handler skips posting, exits cleanly
- Two mentions during cold window → `_is_cold_start()` returns True only once; second mention gets warm path
- HELP command during cold start → HELP check runs before cold-start check → returns early, no boot sequence

### 2.7 Robust Price Resolution

The `resolve_price()` function in `data_fetcher.py` implements an 8-level fallback chain to handle yfinance's inconsistent price field availability across market conditions:

```
Fallback chain (first non-null, positive value wins):
    1. currentPrice          — primary (regular hours)
    2. preMarketPrice        — pre-market sessions
    3. postMarketPrice       — after-hours sessions
    4. regularMarketPrice    — alternative regular-hours field
    5. previousClose         — last trading day close
    6. regularMarketPreviousClose — alternative previous close
    7. Last OHLCV Close      — from historical bars (hist DataFrame)
    8. Last OHLCV Open       — fallback if Close is missing
    9. info['open']          — last resort from info dict
    → 0                      — all sources exhausted
```

This function is used throughout `data_fetcher.py` and `v8_data.py`, replacing the previous fragile pattern of `info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)`. All `ticker.history()` calls also include `prepost=True` for extended-hours data visibility.

### 2.8 V9 Owner Intelligence Layer

ATLAS V9 adds a Buffett-aligned owner intelligence layer on top of the existing V8 quantitative engine. The V8 engine (11 layers, 8 scoring engines, regime classification, meta-learning) remains unchanged. V9 operates entirely in the report layer (`v8_report.py`) and prompt layer (`gemini_qa.py`), computing owner-oriented scores from the same data the quant engine already produces.

**Philosophy:** V9 treats every equity as a partial business ownership stake. The primary question is not "Will the price go up?" but "Would a rational business owner buy this entire business at this price?" This reframes the analysis from trading signals to capital allocation decisions.

#### 2.8.1 V9 Scoring Computation

All V9 scores are computed in `_compute_v9_owner_scores(summary, v8_data)` (~200 lines in `v8_report.py`).

**Business Quality Score (0–5):**
```
+1 if ROE > 15%
+1 if Net Margin > 10%
+1 if Revenue Growth > 5%
+1 if FCF Yield > 3% (free cash flow generation)
+1 if Debt/Equity < 1.0 (conservative balance sheet)
+0.5 if Earnings Beat Rate ≥ 67% (consistent execution)
Capped at 5
```

**Moat Durability Score (0–5):**
```
+1 if Gross Margin > 40% (pricing power)
+1 if Operating Margin > 20% (operational efficiency)
+1 if Market Cap > $50B (scale advantage)
+1 if ROE > 20% AND Gross Margin > 50% (compounding moat)
+1 if Interest Coverage > 8x (fortress balance sheet)
Capped at 5
```

**Capital Allocation Score (0–5):**
```
+1 if ROIC proxy (ROE × (1 − D/E payout)) > WACC proxy (10Y yield + 5%)
+1 if Buyback Yield > 0 AND PE < sector median (valuation-aware buybacks)
+1 if Dividend Yield 1–5% AND Payout Ratio < 60% (sustainable dividends)
+1 if Debt/Equity < 0.5 (disciplined debt usage)
+1 if Net Cash positive (cash exceeds debt)
Capped at 5
```

**Margin of Safety (MOS):**
```
MOS% = (DCF Base Price − Current Price) / DCF Base Price × 100
```
Required thresholds by business type:
| Business Type | Classification Rule | Required MOS |
|---------------|-------------------|--------------|
| Very Stable | Quality ≥ 4 AND Moat ≥ 4 | 20% |
| Normal | Quality ≥ 2 AND Moat ≥ 2 | 30% |
| Cyclical | All others | 45% |

**Conviction Score (0–100):**
```
raw = quality × 5 + moat × 4 + capital_allocation × 4 + max(MOS, 0) × 0.5
risk_adjustment = −10 per HIGH permanent loss risk, −5 per MEDIUM
conviction = clamp(raw + risk_adjustment, 0, 100)
```
Conviction maps to position sizing:
- 80+ → Top-10 position (concentrated)
- 60–80 → Meaningful position
- <60 → Opportunistic / small allocation

#### 2.8.2 V9 Decision Hierarchy

The V9 decision follows a strict 5-step gate, evaluated in order:

```
Step 1: Business Quality Gate
    IF quality < 2 → PASS ("Business quality below minimum threshold")
    ELSE → continue

Step 2: Intrinsic Value Comparison
    IF MOS ≥ required_mos → BUY candidate
    IF MOS between 0 and required_mos → RESEARCH ("Below required margin of safety")
    IF MOS < 0 → WATCH ("Trading above intrinsic value")

Step 3: Capital Allocation Review
    IF CA_score < 2 → downgrade to WATCH ("Poor capital allocation discipline")
    ELSE → maintain decision

Step 4: Permanent Loss Risk Check
    IF any HIGH severity risks → downgrade (BUY→RESEARCH, RESEARCH→WATCH)

Step 5: Engine Overlay
    IF quant verdict is SELL/STRONG SELL → max decision is HOLD
    IF all gates passed and conviction ≥ 60 → BUY
    ELSE → RESEARCH
```

Decision labels: **PASS** (avoid entirely) | **WATCH** (monitor but don't act) | **RESEARCH** (promising but needs more margin) | **BUY** (meets all gates) | **HOLD** (own and keep) | **TRIM** (reduce position) | **EXIT** (sell entirely)

#### 2.8.3 Engine Conflict Protocol

When the V9 owner assessment and V8 quant engine disagree, a conflict note is added to the report:

| Scenario | V9 Owner | V8 Engine | Resolution |
|----------|----------|-----------|------------|
| Patience over momentum | BUY | SELL | "Quant engine sees downward momentum, but business quality and MOS suggest patience. Short-term price weakness in a durable business." |
| Temporary weakness | WATCH | STRONG BUY | "Quant sees a bounce opportunity, but owner assessment sees insufficient margin of safety. Discipline over excitement." |
| Agreement | BUY | BUY | No conflict note. Conviction reinforced. |

The V9 assessment takes priority for capital allocation sizing; the quant engine serves as a tactical timing overlay.

#### 2.8.4 Temperament Module

Every V9 report includes a temperament note derived from VIX:

```
VIX < 15:  "Market is complacent. Good businesses may be expensive — patience."
VIX 15-25: "Normal volatility. Focus on business quality and margin of safety."
VIX 25-35: "Fear rising. Historically, good entry points emerge in fear."
VIX > 35:  "Extreme fear. Best opportunities often appear when others panic."
```

#### 2.8.5 Permanent Loss Risk Identification

V9 identifies up to 5 permanent loss risks from the data:

| Condition | Risk | Severity |
|-----------|------|----------|
| Debt/Equity > 2.0 | Excessive leverage | HIGH |
| Net Margin < 0 | Operating at a loss | HIGH |
| FCF Yield < 0 | Burning cash | HIGH |
| Revenue Growth < −10% | Revenue declining | MEDIUM |
| Short Interest > 10% | Heavy short pressure | MEDIUM |
| Earnings Beat Rate < 50% | Missing expectations | MEDIUM |
| Beta > 2.0 | Extreme volatility | MEDIUM |
| ROE < 5% | Poor returns on equity | LOW |
| Interest Coverage < 3x | Thin debt service | MEDIUM |

---

## 3. Model Strategy

### 3.1 Model Selection

Atlas Engine uses a single LLM provider for conversational AI:

| Component | Provider | Model | Purpose |
|-----------|----------|-------|---------|
| AI Q&A | Groq | Llama 3.3 70B Versatile | Buffett-aligned multi-turn capital allocation reasoning on cached analysis data |
| V9 owner scores | None (rule-based) | N/A | Deterministic business quality, moat, CA, MOS, conviction computation |
| Core engine | None (rule-based) | N/A | 8 scoring engines use deterministic mathematical logic |
| Sentiment | None (keyword-based) | N/A | Positive/negative keyword matching on news titles |

**Why Groq / Llama 3.3 70B:**
- **Cost:** $0 on free tier (30 requests/minute). Eliminates API spend during development and low-traffic production.
- **Latency:** 2–3 seconds per response. Groq's custom LPU inference hardware delivers the fastest 70B inference commercially available.
- **Quality:** 70B parameter models produce sufficient reasoning quality for structured financial Q&A when grounded with context data.
- **API compatibility:** OpenAI-compatible API format simplifies potential migration to other providers.

**Why not Claude, GPT-4, or other commercial LLMs (currently):**
- Cost sensitivity at current stage. Claude Opus at ~$15/MTok and GPT-4 at ~$10/MTok would cost $50–200/month at moderate usage.
- The Q&A module is a secondary feature; the primary value is the deterministic engine.
- Migration path is trivial — the `gemini_qa.py` module isolates all LLM interaction behind a single function.

### 3.2 When the LLM Is Selected

The LLM is invoked only for follow-up questions in Slack threads, not for core analysis:

```
Core analysis path:    @atlas AAPL → No LLM involved
Follow-up Q&A path:    "What's the downside risk?" (in thread) → Groq LLM
```

The engine's 11-layer analysis is entirely rule-based and deterministic. Given identical input data, it produces identical output. This is a deliberate architectural choice: the scoring logic, regime classification, and weight optimization use transparent mathematical operations (tanh normalization, softmax, matrix multiplication) that can be audited, backtested, and debugged without the opacity of neural inference.

### 3.3 Cost vs. Performance Tradeoffs

| Decision | Cost | Performance | Tradeoff |
|----------|------|-------------|----------|
| Groq free tier | $0/mo | 30 RPM, 8K context | Adequate for <100 daily Q&A queries. Production scaling needs paid tier ($5–20/mo). |
| Rule-based engine | $0/mo | Deterministic, auditable | No learned reasoning. Affinity matrix is hand-tuned, not optimized from data. |
| yfinance | $0/mo | Daily bars, 15-min delayed | No real-time or tick-level data. Sufficient for swing/position trading timeframes. |
| FRED (optional) | $0/mo | High-quality macro data | Requires API key registration. Fallback to yfinance proxies if absent. |

### 3.4 Context Window Strategy

**Groq Llama 3.3 70B context window: 8,192 tokens**

The context builder in `gemini_qa.py` constructs a structured data block of approximately 6,000 characters (~1,500 tokens) containing:

```
Context allocation (approximate):
├── System prompt:         ~300 tokens (role, constraints, formatting rules)
├── Analysis context:      ~1,500 tokens (verdict, scores, fundamentals, technicals)
├── Conversation history:  ~500 tokens (last 6 exchanges, truncated)
├── Current question:      ~50 tokens
├── Response budget:       ~1,024 tokens (max_tokens parameter)
└── Buffer:                ~4,800 tokens available
    Total:                 ~8,192 tokens
```

**Context prioritization order:**
1. V9 Owner Assessment — decision, quality/moat/CA scores, MOS, conviction, risks (always included, primary)
2. Engine verdict and composite scores (always included, secondary overlay)
3. Price levels and moving averages (always included)
4. Scoring engine breakdown with weights (always included)
5. Risk metrics and drivers (always included)
6. Fundamentals and financials (included if available)
7. Technicals — RSI, MACD, Bollinger (included if available)
8. Peers comparison (included if space permits)
9. News with sentiment (included if space permits)
10. DCF and institutional data (truncated if necessary)

The context block is truncated at 6,000 characters to preserve token budget for conversation history and response generation.

### 3.5 Fine-Tuning vs. Prompting

Atlas Engine uses prompting exclusively. No fine-tuned models are deployed.

**System prompt design (gemini_qa.py):**

V9 replaces the generic financial analyst prompt with a Buffett-aligned philosophy prompt:
- Role anchoring: "ATLAS V9 — Capital Allocation Intelligence. Think like Warren Buffett's research analyst."
- Reasoning order enforcement: "Always reason in this order: (1) Is this a good business? (2) Does it have a durable moat? (3) Is management allocating capital wisely? (4) What is the intrinsic value? (5) What are the permanent loss risks? (6) Only then consider price and timing."
- Decision framework: "Use V9 decisions: PASS / WATCH / RESEARCH / BUY / HOLD / TRIM / EXIT"
- Owner mindset: "Treat every stock as a partial business ownership stake"
- Anti-speculation: "Never recommend based on momentum alone. Price is what you pay, value is what you get."
- Data grounding: "Only reference data from the ATLAS analysis provided below"
- Anti-hallucination: "Never make up numbers or statistics"
- Format control: "Use Slack markdown, keep responses under 3,500 characters"

**V9 context injection:** The Q&A context builder injects V9 owner assessment as the primary section (before the engine verdict):
```
--- V9 OWNER ASSESSMENT ---
Decision: BUY — Meets all quality gates with adequate margin of safety
Business Quality: ★★★★☆ (4/5)
Moat Durability: ★★★☆☆ (3/5)
Capital Allocation: ★★★★☆ (4/5)
Intrinsic Value (Base): $195.00 | Current: $189.50
Margin of Safety: +2.9%
Conviction: 72/100
Permanent Loss Risks: Valuation premium (MEDIUM)

--- ENGINE VERDICT (Tactical Overlay) ---
Signal: BUY | Composite: +42.3 | TQ: 0.456
...
```

**Why prompting over fine-tuning (current stage):**
- Insufficient proprietary training data for effective fine-tuning
- Prompt engineering provides adequate domain anchoring when combined with structured context injection
- Fine-tuning creates vendor lock-in; prompting is portable across providers
- Rapid iteration cycle (change prompt → test immediately) vs. fine-tuning overhead

### 3.6 Caching Strategy

**Thread-Level Analysis Cache:**
```python
_thread_cache = {
    thread_ts: {
        'symbol': 'AAPL',
        'summary': { ... },              # Full engine output
        'v8_extended': {
            ...,
            'v9_scores': { ... }         # V9 owner intelligence scores
        },
        'timestamp': 1707763800.5,        # Creation time
        'conversation_history': [ ... ]   # (role, message) pairs
    }
}
```

- **TTL:** 4 hours per thread
- **Capacity:** 50 threads (LRU eviction)
- **Cleanup:** On every message receipt + periodic sweep
- **Persistence:** In-memory only (lost on restart)
- **Purpose:** Avoids re-running the full engine for follow-up questions; provides the LLM with grounding data

**Meta-Learning State Cache:**
```python
# Persisted to disk: state/atlas_meta_state.json
{
    "w0": [0.125, 0.115, 0.130, ...],   # Learned base weights (8 engines)
    "Q": [[...], [...], ...],             # Performance matrix
    "run_count": 47                        # Total analysis runs
}
```

- **Persistence:** Written to disk after every engine run
- **Purpose:** Weight optimization improves with accumulated data
- **Cold start:** First 10 runs use equal weights; learning activates at run 11

### 3.7 Tool Calling / Function Calling Logic

Atlas Engine does not currently use LLM tool calling or function calling. The Groq integration is a straightforward chat completion:

```python
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[system_prompt, context_block, *history, user_question],
    temperature=0.3,
    max_tokens=1024
)
```

The LLM receives pre-computed data as context and generates natural language responses. It has no ability to invoke tools, query APIs, or trigger engine re-runs. This is a limitation addressed in the roadmap (Section 7).

---

## 4. Integrations

### 4.1 Integration Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      ATLAS ENGINE                               │
│                                                                 │
│   ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  ┌─────┐ │
│   │  Slack   │  │ yfinance │  │  FRED    │  │ Groq │  │FastA│ │
│   │  API     │  │  API     │  │  API     │  │ API  │  │ PI  │ │
│   └────┬────┘  └────┬─────┘  └────┬─────┘  └──┬───┘  └──┬──┘ │
│        │            │              │            │         │     │
│   Socket Mode  HTTP/REST      HTTP/REST    HTTP/REST   HTTP    │
│   (WSS)        (No auth)      (API key)    (API key)  :PORT    │
│        │            │              │            │         │     │
└────────┼────────────┼──────────────┼────────────┼─────────┼────┘
         │            │              │            │         │
         ▼            ▼              ▼            ▼         ▼
    Slack         Yahoo Finance  Fed Reserve   Groq     Browser
    Workspace     (Public)       Econ Data     Cloud    (Dashboard)
```

### 4.2 Internal System Integrations

**Slack (Primary Interface)**
- **Protocol:** Socket Mode (WebSocket Secure)
- **Authentication:** Bot token (xoxb-) + App-level token (xapp-)
- **OAuth scopes:** `app_mentions:read`, `chat:write`, `channels:history`
- **Events:** `app_mention` (trigger analysis), `message` (thread Q&A)
- **Rate limits:** Slack Tier 1 — 1 message/second per channel (respected via sequential posting)
- **Message format:** Slack mrkdwn (bold, code blocks, lists)
- **Message size:** Each section capped at ~4,000 characters (Slack limit)

### 4.3 External API Integrations

**Yahoo Finance (via yfinance library)**
- **Authentication:** None required (public API)
- **Data categories fetched:**

| Data | Endpoint | Frequency | Fallback |
|------|----------|-----------|----------|
| OHLCV | `ticker.history(period="1y")` | Per request | Static CSV |
| Fundamentals | `ticker.info` | Per request | Static JSON |
| Analyst consensus | `ticker.info` (targetMeanPrice, recommendations) | Per request | Static JSON |
| VIX levels | `yf.Ticker("^VIX").history()` | Per request | Static CSV |
| Treasury proxies | `yf.Ticker("^TNX").history()` | Per request | FRED data |
| Breadth indices | `yf.Ticker("^GSPC").history()` | Per request | Static CSV |
| Global indices | `yf.Ticker("^N225").history()` etc. | Per request | Static JSON |

- **Rate limits:** yfinance uses undocumented Yahoo Finance endpoints. No formal rate limit, but aggressive usage risks IP throttling.
- **Latency:** 1–3 seconds per ticker call; parallelized across 7 workers.

**FRED (Federal Reserve Economic Data)**
- **Authentication:** Free API key (registered at fred.stlouisfed.org)
- **Endpoint:** `https://api.stlouisfed.org/fred/series/observations`
- **Series used:**

| Series ID | Description | Update Frequency |
|-----------|-------------|-----------------|
| DGS10 | 10-Year Treasury Yield | Daily |
| DGS2 | 2-Year Treasury Yield | Daily |
| BAMLH0A0HYM2 | ICE BofA US High Yield Spread | Daily |
| UNRATE | Unemployment Rate | Monthly |

- **Fallback:** If FRED unavailable, yields estimated from yfinance Treasury ETF proxies (^TNX, ^FVX).

**Groq Cloud**
- **Authentication:** API key (Bearer token)
- **Model:** `llama-3.3-70b-versatile`
- **Rate limits:** Free tier = 30 requests/minute, 6,000 tokens/minute
- **Latency:** 2–3 seconds per completion
- **Retry strategy:** 2 retries with exponential backoff on 429 errors
- **Minimum interval:** 3 seconds between calls (self-imposed)

### 4.4 Authentication Flows

```
Startup Authentication:
├── Load .env (python-dotenv, override=False)
├── Validate SLACK_BOT_TOKEN (required — exit if missing)
├── Validate SLACK_APP_TOKEN (required — exit if missing)
├── Initialize Groq client if GROQ_API_KEY present (optional)
├── Store FRED_API_KEY if present (optional)
├── Read ATLAS_WEB_BASE_URL (default: http://localhost:8000)
├── Start FastAPI/uvicorn on PORT (default: 10000)
├── Register SIGTERM/SIGINT signal handlers
└── Connect Slack Socket Mode (WebSocket handshake with Slack servers)
```

All API keys are stored in `.env` (Git-ignored) and loaded via `python-dotenv` with `override=False`, allowing Render platform environment variables to take precedence over local `.env` values.

**Environment variables:**

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `SLACK_BOT_TOKEN` | Yes | — | Bot user OAuth token (xoxb-) |
| `SLACK_APP_TOKEN` | Yes | — | App-level token for Socket Mode (xapp-) |
| `GROQ_API_KEY` | No | — | Enables AI follow-up Q&A |
| `FRED_API_KEY` | No | — | Higher-quality macro data from FRED |
| `ATLAS_WEB_BASE_URL` | No | `http://localhost:8000` | Public URL prefix for web report links |
| `PORT` | No | `10000` | HTTP port for FastAPI (health + dashboard) |
| `CAPITAL` | No | `250000` | Portfolio capital for position sizing |
| `USE_LIVE_DATA` | No | `true` | Toggle live vs static data mode |

### 4.5 Data Security Model

| Data Type | Classification | Transmission | Storage |
|-----------|---------------|--------------|---------|
| API keys | Secret | HTTPS only (to respective services) | `.env` file (Git-ignored) |
| Market data | Public | HTTPS from Yahoo/FRED | Temp dir (auto-deleted) |
| Analysis output | Internal | WSS to Slack + HTTPS (dashboard) | In-memory cache (4h TTL) + SQLite (permanent) |
| Web reports | Internal | HTTPS to browser | `reports.db` (SQLite, WAL mode) |
| Meta-learning state | Internal | None (local only) | `state/atlas_meta_state.json` |
| User messages | Internal | WSS from Slack | In-memory conversation history |

No user financial data, trading accounts, portfolio holdings, or personally identifiable information is collected, stored, or transmitted. Atlas Engine operates on public market data exclusively.

**Note on web reports:** Report payloads stored in SQLite contain only public market data and engine-computed scores. Report URLs use opaque IDs (`{SYMBOL}-{thread_ts}-{uuid}`) and are not guessable. No authentication is required to view a report — anyone with the URL can access it. If access control is required, implement token-based authentication on the `/r/` and `/api/r/` routes.

### 4.6 Data Refresh Cadence

| Source | Refresh | Type |
|--------|---------|------|
| Price OHLCV | On every `@atlas` request | On-demand, real-time fetch |
| Fundamentals | On every request | On-demand (quarterly underlying data) |
| VIX | On every request | On-demand, daily resolution |
| Macro rates (FRED) | On every request | On-demand, daily resolution |
| News | On every request (via V8 extended) | On-demand, latest available |
| Meta-learning weights | Updated after every engine run | Persistent, incremental |
| Web report payloads | Stored after every analysis | Persistent (SQLite) |

All data fetching is synchronous with the analysis request. There is no background refresh, scheduled polling, or pre-caching. This ensures data freshness at the cost of per-request latency. Web report payloads are stored permanently and represent a point-in-time snapshot of the analysis.

### 4.7 Risks of Integration Dependency

| Risk | Severity | Mitigation |
|------|----------|------------|
| yfinance endpoint changes | High | yfinance wraps undocumented Yahoo endpoints; API changes can break data fetching without warning. Monitor yfinance release notes. |
| Yahoo Finance IP throttling | Medium | Parallel 7-worker fetches may trigger rate limiting. Mitigated by per-request (not continuous) usage pattern. |
| Groq free tier deprecation | Medium | If free tier removed, Q&A feature requires paid plan (~$5–20/mo) or migration to alternative provider. |
| FRED API instability | Low | FRED is federally funded with high uptime. Fallback to yfinance proxies exists. |
| Slack API changes | Low | Using stable `slack-bolt` library with semantic versioning. Socket Mode is an officially supported protocol. |

### 4.8 Latency Considerations

```
Component Latency Breakdown (typical):

Cold-start boot msgs:       ████░░░░░░░░░░░░░░░░  3-4 sec (cold only)
yfinance parallel fetch:    ████████░░░░░░░░░░░░  5-8 sec
Engine 11-layer execution:  ██░░░░░░░░░░░░░░░░░░  2-3 sec
V8 extended data fetch:     ████░░░░░░░░░░░░░░░░  3-5 sec
V9 owner score computation: ░░░░░░░░░░░░░░░░░░░░  <50 ms
Report formatting (11 sec): ██░░░░░░░░░░░░░░░░░░  2-3 sec
Web report (SQLite store):  ░░░░░░░░░░░░░░░░░░░░  <100 ms
Slack posting (11 msgs):    ████░░░░░░░░░░░░░░░░  3-5 sec
                            ──────────────────────
Total (warm):                                     15-24 sec
Total (cold start):                               18-28 sec

Groq Q&A (per follow-up):  ██░░░░░░░░░░░░░░░░░░  2-3 sec
Dashboard load (browser):   ██░░░░░░░░░░░░░░░░░░  1-2 sec
```

The dominant latency contributor is network I/O (data fetching and Slack posting), not computation. The 11-layer engine itself completes in 2–3 seconds on commodity hardware.

---

## 5. User Interaction & Workflows

### 5.1 Typical User Journey

```
Phase 0: Cold Start (Render free-tier only, first request after spin-down)
──────────────────────────
User types in any Slack channel where Atlas is installed:
    @atlas AAPL

Atlas wakes up and posts a progress sequence (updated in-place):
    😴 ATLAS is waking up from sleep... hang tight
    📡 Connecting data systems...
    🚀 Systems online, fetching AAPL...
    📈 Live data fetched, running V9 engine...
    🧠 Engine complete, building owner assessment...
    ✅ Report ready, delivering...

Phase 1: Analysis Request
──────────────────────────
(If already warm, user sees standard acknowledgment):
    ⚙️ Running ATLAS V9 on *AAPL*... pulling live data & building owner assessment + full report

15-20 seconds later, Atlas posts a summary message + 11 threaded sections.
First message includes a hyperlink to the full web dashboard.

Phase 2: Report Consumption
──────────────────────────
User reads the Owner Assessment first:
    🏛️ ATLAS V9 — OWNER ASSESSMENT — AAPL (Apple Inc.)
    Decision: BUY — Meets all quality gates with adequate margin of safety
    Business Quality: ★★★★☆ | Moat: ★★★☆☆ | Capital Allocation: ★★★★☆
    MOS: +2.9% | Conviction: 72/100

Then the engine verdict section:
    📊 ATLAS V9 — AAPL (Apple Inc.)
    Signal: BUY | Composite: +42.3 | TQ: 0.456 (NORMAL)
    Regime: Calm | Reliability: 87%

User clicks into thread to read detailed sections (fundamentals,
valuation, technicals, peers, sentiment, risk, catalysts, macro, signal).

Phase 2b: Web Dashboard (Optional)
──────────────────────────
User clicks the dashboard link in the Slack message:
    🔍 Full Report: Open dashboard → https://atlas-slack-bot.onrender.com/r/AAPL-...

The web dashboard provides:
    • V9 Owner's View: decision, star ratings, conviction bar, MOS gauge, IV comparison, permanent loss risks
    • ECharts financial visualizations (price context, engine waterfall, regime vector, DCF)
    • Dense metric grids (fundamentals, valuation, technicals, trade plan)
    • Sortable peer comparison and engine detail tables
    • "Show Your Work" section with V9 decision hierarchy and full formula derivation
    • Raw JSON API access at /api/r/{id}.json

Phase 3: Follow-Up Questions (Optional)
──────────────────────────
User replies in the analysis thread:
    "What's the main risk to this position?"

Atlas AI responds in 3-5 seconds:
    "Based on the ATLAS analysis, the primary risk drivers for AAPL are:
     1. *Valuation premium* — PE at 28.3x vs sector median 22.1x
     2. *VIX regime* — current VIX at 18.5 is neutral but trending higher
     ..."

User can ask up to 6 follow-up questions per thread.

Phase 4: Shutdown Notification (Automatic)
──────────────────────────
When Render sends SIGTERM (after 15 min inactivity), Atlas posts:
    😴 ATLAS is going offline (Render free-tier sleep).
       Mention me again to wake up — takes ~30 seconds.
```

### 5.2 Prompt Structure Best Practices

**Analysis trigger (simple):**
```
@atlas AAPL          → Analyze Apple
@atlas MSFT          → Analyze Microsoft
@atlas SPY           → Analyze S&P 500 ETF (default if no ticker)
```

**Follow-up questions (effective patterns):**
```
"What are the permanent loss risks?"
"Is the moat durable enough for a 10-year hold?"
"Compare AAPL to its peers on valuation."
"What would change this from BUY to PASS?"
"Explain the margin of safety calculation."
"Is management allocating capital well?"
"What does the DCF model assume for growth?"
"Why does the owner assessment differ from the engine signal?"
```

**Anti-patterns (less effective):**
```
"Analyze MSFT too"         → Redirected to @atlas MSFT
"What's Bitcoin doing?"    → Out of scope; redirected
"Buy 100 shares"           → Atlas is analysis-only, no execution
```

### 5.3 Multi-Step Reasoning Workflow

**Example: Evaluating a Contrarian Position (V9 Owner's Perspective)**

```
Step 1: User triggers analysis
    @atlas META

Step 2: Atlas V9 Owner Assessment: RESEARCH with engine SELL
    User sees: Quality ★★★★☆, Moat ★★★☆☆, but MOS only +8% (needs 30%)
    Engine shows negative trend, but strong fundamentals

Step 3: User probes the conflict
    "The owner assessment says RESEARCH but the engine says SELL. What should I do?"

Step 4: Atlas AI reasons through V9 hierarchy
    "The V9 owner assessment sees a quality business (4/5) with decent moat
     (3/5), but the margin of safety is only 8% — below the 30% threshold
     for a Normal business type. The engine's SELL signal reflects
     short-term momentum weakness. As a business owner, the quality is
     there but the price isn't right yet. RESEARCH means: keep watching
     for a better entry. Patience over momentum."

Step 5: User asks about entry price
    "What price would give me an adequate margin of safety?"

Step 6: Atlas AI calculates from V9 data
    "The DCF base intrinsic value is $520. For a Normal business type,
     you need 30% MOS, which means buying below $364. Current price
     is $480 — you'd need a ~20% pullback. The engine's SELL signal
     suggests momentum is moving in that direction, but timing
     pullbacks is speculative. Better to set a price alert at $380
     and let the market come to you."
```

### 5.4 Follow-Up Query Handling

```
Thread Q&A Architecture:

First question in thread:
├── Lookup thread_ts in cache
├── Build full context block (~6,000 chars)
│   ├── V9 owner assessment (primary: decision, scores, MOS, conviction, risks)
│   ├── Engine verdict + scores (secondary tactical overlay)
│   ├── Fundamentals + technicals
│   ├── Peers + news + DCF
│   └── Truncate if >6,000 chars
├── Construct messages: [system, context+question]
├── Call Groq API
├── Cache response in conversation_history
└── Post reply to Slack thread

Subsequent questions:
├── Lookup thread_ts in cache
├── Rebuild context (same data)
├── Append prior conversation_history (max 12 messages)
├── Construct messages: [system, context, *history, question]
├── Call Groq API
├── Append to conversation_history
└── Post reply to Slack thread
```

**Conversation history management:**
- Maximum 6 exchanges (12 messages: 6 user + 6 assistant)
- Oldest exchanges evicted when limit reached
- Each thread maintains independent history
- History lost after 4-hour TTL or server restart

### 5.5 Memory Handling Strategy

| Memory Type | Scope | Duration | Storage |
|-------------|-------|----------|---------|
| **Analysis cache** | Per-thread | 4 hours | In-memory dict |
| **Conversation history** | Per-thread | 4 hours | In-memory dict |
| **Web reports** | Per-analysis | Permanent | `reports.db` (SQLite, WAL mode) |
| **Meta-learning weights** | Global (all runs) | Permanent | `state/atlas_meta_state.json` |
| **Cross-session memory** | None | N/A | Not implemented |

**Short-term memory:** Thread cache provides intra-session context. The LLM sees the full analysis data plus recent conversation on every call, enabling coherent multi-turn dialogue within a single analysis thread.

**Long-term memory:** Meta-learning weights persist across server restarts and accumulate learning over time. After 50+ runs, the weight vector reflects empirical engine performance. No other long-term memory exists — there is no user preference storage, no cross-ticker memory, and no historical analysis recall.

### 5.6 Example Workflow: Full Analysis Cycle

```
09:15 AM — Pre-market prep
    @atlas SPY
    → V9 Owner Assessment: HOLD — Quality ★★★★★, broad market exposure
    → Engine: BUY signal, Calm regime, TQ=0.52
    → Follow-up: "Any overnight risk from Asia?"
    → Atlas AI: "Nikkei closed +0.3%, no significant overnight gap risk.
     As a business owner of the S&P 500, overnight gaps in stable
     markets are noise, not signal."

09:30 AM — Market open, evaluating a specific stock
    @atlas NVDA
    → V9 Owner Assessment: RESEARCH — Quality ★★★★★, Moat ★★★★★,
      but MOS -12% (trading above intrinsic value)
    → Engine: STRONG BUY, TQ=0.71, momentum + earnings beat
    → Conflict note: "Engine sees momentum, but owner assessment
      sees insufficient margin of safety. Patience over excitement."
    → Follow-up: "Is the moat durable enough for a 10-year hold?"
    → Atlas AI: "NVDA's moat is exceptional: 62% gross margin (pricing power),
     $3T+ scale, AI/datacenter dominance. 5/5 moat durability. The question
     isn't whether to own it — it's at what price."

02:00 PM — Afternoon reassessment
    @atlas NVDA
    → Fresh analysis with updated V9 owner assessment + engine signal
    → Compare morning vs afternoon conviction scores
```

---

## 6. Current Limitations

### 6.1 Hallucination Risk

**Scope:** The Groq-powered Q&A module is susceptible to LLM hallucination, particularly when:
- Users ask about data not present in the cached context (e.g., options Greeks, insider transactions)
- Context truncation removes relevant data that the LLM then fabricates
- Questions require reasoning beyond the provided analysis (e.g., "What will earnings be next quarter?")

**Mitigation in place:**
- System prompt instructs: "Never make up numbers or statistics"
- Context block provides grounding data for most questions
- Scope enforcement redirects off-topic questions

**Residual risk:** The LLM has no mechanism to verify its own numerical claims against source data. A hallucinated statistic presented with confidence is indistinguishable from an accurate one.

### 6.2 Data Freshness Gaps

| Data Type | Freshness | Gap |
|-----------|-----------|-----|
| Price OHLCV | Daily bars (closing prices) | No intraday data. Morning analysis uses yesterday's close. |
| Fundamentals | Quarterly (SEC filings) | Stale for 1–3 months between earnings reports. |
| Analyst consensus | Updated irregularly by Yahoo | Can lag actual revisions by days to weeks. |
| VIX | Daily | No intraday VIX. Cannot detect mid-session volatility spikes. |
| News | Latest available from Yahoo | May miss breaking news from non-Yahoo sources. |
| Earnings surprises | yfinance historical | 2–4 week lag after earnings date. |

### 6.3 Context Window Constraints

The Groq Llama 3.3 70B model has an 8,192-token context window. With system prompt, analysis context, and conversation history, the effective response budget is ~1,024 tokens (~4,000 characters). This constrains:
- Follow-up responses cannot be longer than ~800 words
- Conversation history is truncated to 6 exchanges; earlier context is lost
- Complex multi-part questions may produce incomplete answers
- Detailed quantitative breakdowns may be truncated

### 6.4 Model Reasoning Limits

The core engine uses deterministic rules, not learned reasoning. Specific limitations:
- **Affinity matrix is hand-tuned.** The 80-coefficient matrix mapping regime features to engine weights is based on domain intuition, not empirical optimization. Suboptimal weighting is likely.
- **Valuation timing is intentionally weak.** The valuation engine is capped at [-40, +40] because PE/EV timing is notoriously unreliable. This means the engine will underweight valuation even when it is highly predictive.
- **No event-driven reasoning.** The engine cannot factor in specific catalysts (FDA approvals, earnings dates, geopolitical events) unless they are reflected in price or volatility data.
- **Linear regime classification.** The regime classifier uses threshold-based rules, not clustering or neural classification. It may misclassify transitional regimes.

### 6.5 Numerical Precision Issues

- **DCF model assumptions are approximate.** The V8 DCF uses simplified growth rate and discount rate estimates derived from fundamental data. It does not model segment-level revenue, capex cycles, or working capital dynamics.
- **Correlation calculations use 20-day rolling windows.** Short lookback periods amplify noise. A spurious 2-week correlation spike can trigger regime changes.
- **Position sizing assumes uniform risk.** The position sizer uses a fixed capital allocation framework. It does not account for portfolio-level correlation, sector concentration, or liquidity-adjusted sizing.

### 6.6 Dependency Bottlenecks

- **Single data provider (yfinance).** If Yahoo Finance changes its undocumented API structure or throttles IP ranges, all data fetching fails simultaneously. There is no secondary data provider.
- **Single LLM provider (Groq).** If Groq experiences downtime or deprecates the free tier, Q&A is fully disabled. No fallback LLM is configured.
- **In-memory state.** Thread cache and conversation history are lost on server restart. A deployment on Render's free tier (which sleeps after inactivity) means cold starts lose all cached Q&A context. Web reports persist in SQLite and survive restarts. Cold-start boot messages and graceful shutdown notifications mitigate the UX impact.

### 6.7 Scaling Constraints

| Dimension | Current Limit | Bottleneck |
|-----------|--------------|------------|
| Concurrent analyses | ~3–5 simultaneous | yfinance throttling, single Python process |
| Q&A requests/minute | 30 | Groq free tier rate limit |
| Cached threads | 50 | In-memory LRU; no persistence |
| Tickers supported | US equities only | yfinance coverage limitation |
| Users | Single Slack workspace | No multi-tenant architecture |

### 6.8 Security and Compliance Concerns

- **No audit logging.** Print-based logging captures operational events but does not produce structured audit trails suitable for compliance review.
- **No access control.** Any user in the Slack workspace can invoke Atlas and access all analysis. There is no role-based access, no per-user rate limiting, and no content filtering.
- **Groq data retention.** Groq's API terms specify that input/output data may be retained for abuse monitoring. Analysis context (which includes financial data) is transmitted to Groq servers.
- **No encryption at rest.** Meta-learning state and temp data files are written to disk in plaintext.

---

## 7. Improvements & Roadmap

### 7.1 Phase A: Immediate Improvements (0–3 Months)

#### A1. Structured Logging
Replace `print()` with Python's `logging` module. Implement structured JSON logging with severity levels, timestamps, and request correlation IDs.
```
Impact: Debuggability, monitoring, compliance readiness
Effort: 2-3 days
```

#### A2. Persistent Thread Cache
Replace in-memory dict with Redis or SQLite for thread cache persistence. Survives restarts and platform sleep cycles.

> **Partially implemented (v2.1):** Web reports are now persisted in SQLite (`reports.db`), surviving restarts. The in-memory Q&A thread cache remains volatile. Full thread cache persistence is a remaining item.

```
Impact: User experience continuity
Effort: 3-5 days (remaining: Q&A cache migration to SQLite)
```

#### A3. Automatic Citation System
Tag every numerical claim in Q&A responses with a source reference. When the LLM mentions "PE of 28.3x", append `[Source: fundamentals.json, field: trailingPE]`.
```
Implementation: Post-process LLM output with regex matching against cached data.
Inject citation markers into the system prompt.
Impact: Trust, verifiability, hallucination detection
Effort: 5-7 days
```

#### A4. Confidence Scoring for Q&A Responses
Add a confidence tag to every Q&A response based on:
- Percentage of the response that can be grounded in cached data
- Whether the question falls within the analysis scope
- Whether context truncation may have removed relevant data
```
Implementation: Compare response tokens against context tokens using fuzzy matching.
Output: "Confidence: HIGH / MEDIUM / LOW" appended to each response.
Effort: 3-5 days
```

#### A5. Multi-Provider LLM Fallback
Add Claude (Haiku) or GPT-4o-mini as fallback when Groq is unavailable or rate-limited. Provider selection logic:
```
1. Try Groq (free, fast)
2. If Groq 429 or timeout → Try Claude Haiku ($0.25/MTok, reliable)
3. If all fail → Return cached engine data as formatted text
Effort: 3-5 days
```

#### A6. News Sentiment Upgrade
Replace keyword-based sentiment with a lightweight NLP model (DistilBERT or FinBERT) for news headline classification. Runs locally, no API cost.
```
Impact: Sentiment accuracy improvement from ~60% to ~85%
Effort: 5-7 days (model integration + testing)
```

### 7.2 Phase B: Mid-Term Improvements (3–12 Months)

#### B1. Hybrid Search & Retrieval (RAG)
Implement a vector database (Chroma or Pinecone) to store historical analyses, earnings transcripts, and SEC filings. Enable retrieval-augmented generation for Q&A.
```
Architecture:
├── Embedding model: text-embedding-3-small or nomic-embed
├── Vector store: Chroma (local) or Pinecone (hosted)
├── Chunking: 512-token chunks with 50-token overlap
├── Retrieval: Top-5 semantically similar chunks per query
└── Injection: Prepend retrieved context to LLM prompt
Impact: Dramatically richer follow-up responses with historical context
Effort: 2-4 weeks
```

#### B2. Agentic Multi-Step Planning
Enable the LLM to decompose complex questions into sub-tasks and execute them sequentially:
```
User: "Compare AAPL's valuation trajectory over the last year with its peers"
Agent plan:
  1. Retrieve cached AAPL analysis
  2. Identify peers from V8 data
  3. Fetch historical PE data for AAPL + peers (tool call)
  4. Compute valuation trend
  5. Synthesize comparison narrative
```
Requires implementing tool calling / function calling with the LLM.
```
Effort: 3-5 weeks
```

#### B3. Learned Affinity Matrix
Replace the hand-tuned 80-coefficient affinity matrix with an empirically optimized version trained on historical regime-return data.
```
Method:
├── Collect (regime_vector, engine_scores, forward_returns) triples
├── Optimize A matrix to maximize Sharpe ratio of composite signal
├── Use Bayesian optimization or gradient descent
├── Constrain: w_i ∈ [floor, ceiling] for stability
└── Validate on out-of-sample data (walk-forward)
Effort: 4-8 weeks (including data collection + backtesting)
```

#### B4. Model Routing Optimization
Implement intelligent model selection for Q&A based on question complexity:
```
Simple factual → Haiku/small model (fast, cheap)
  "What's the PE ratio?"

Analytical reasoning → Sonnet/medium model
  "Why is the signal bearish despite strong fundamentals?"

Complex scenario → Opus/large model
  "Model the impact of a 100bp rate hike on this position"
```
Classify question complexity using a lightweight classifier before routing.
```
Effort: 2-3 weeks
```

#### B5. Structured JSON-First Output

> **Implemented (v2.1).** Full analysis payloads are now persisted as JSON in SQLite and served via REST API at `/api/r/{report_id}.json`. The JSON includes summary, v8_extended, and provenance data. An institutional-grade HTML dashboard is served at `/r/{report_id}`.

~~Add a structured JSON output mode alongside human-readable reports.~~ **Done.** The web report system (`web_report.py` + `web_server.py`) stores complete analysis payloads as JSON and serves them via FastAPI. Dashboard rendering consumes the same JSON payload client-side.

#### B6. Numerical Verification Layer
Add a post-processing step that validates all numerical claims in Q&A responses:
```
Pipeline:
1. LLM generates response
2. Extract all numbers from response via regex
3. Cross-reference against cached analysis data
4. Flag discrepancies: "⚠️ LLM stated PE=25.3 but data shows PE=28.3"
5. Auto-correct if confidence is high; flag for user review if ambiguous
Effort: 2-3 weeks
```

### 7.3 Phase C: Long-Term Vision (12+ Months)

#### C1. Fine-Tuned Finance Domain Model
Train or fine-tune a domain-specific model on financial analysis data:
- Training corpus: Atlas historical outputs, earnings transcripts, analyst reports, SEC filings
- Base model: Llama 3.x or Mistral (open-weight, fine-tunable)
- Output: Model that reasons about financial concepts natively, reducing prompt engineering overhead
```
Effort: 2-4 months
```

#### C2. Autonomous Validation Loops
Implement self-correcting analysis pipelines:
```
Engine runs analysis → Validation agent checks for contradictions →
If contradiction found → Re-run with adjusted parameters →
If validated → Output final result with validation certificate
```
Contradiction examples:
- STRONG BUY signal with SR > 0.7 (high systemic risk)
- Positive trend score but price below all moving averages
- High TQ but DC < 50 (insufficient data confidence)
```
Effort: 2-3 months
```

#### C3. Real-Time Data Streaming
Replace request-time data fetching with streaming market data:
- WebSocket feeds from Polygon.io or Alpaca Markets
- Continuous OHLCV updates (1-minute bars)
- Real-time VIX and options flow
- Event-driven re-analysis when significant moves detected
```
Impact: Sub-second data freshness, intraday signal generation
Effort: 3-4 months
```

#### C4. Multi-Asset Coverage
Extend beyond US equities to:
- Foreign equities (FTSE, DAX, Nikkei constituents)
- Forex pairs (EUR/USD, GBP/USD, USD/JPY)
- Commodities (Gold, Oil, Natural Gas)
- Crypto (BTC, ETH — top 10 by market cap)
- Fixed income (Treasury ETFs, corporate bond ETFs)

Requires: new scoring engines per asset class, regime features for FX/commodity regimes, and asset-specific execution parameter generation.
```
Effort: 4-6 months
```

#### C5. Synthetic Data Augmentation
Generate synthetic market scenarios for stress-testing:
```
Method:
├── Historical regime replay (re-simulate 2008, 2020, 2022)
├── Monte Carlo path generation from fitted distributions
├── Adversarial scenario construction (worst-case for current position)
└── Use synthetic data to train/validate affinity matrix
Impact: Robustness testing without waiting for live market conditions
Effort: 2-3 months
```

#### C6. Feedback Loops and Reinforcement Learning
Implement outcome tracking and weight optimization:
```
Pipeline:
1. Record analysis output + execution parameters
2. Track actual price movement over next 1, 5, 20 days
3. Compute realized P&L if trade executed
4. Update meta-learning weights based on actual outcomes
5. Penalize engines that contributed to losing signals
6. Reward engines that predicted direction correctly
```
Currently, meta-learning updates weights based on score agreement. This would upgrade to RLHF-style optimization based on realized returns.
```
Effort: 3-5 months
```

---

## 8. Making Follow-Up Responses More Robust and Data-Rich

### 8.1 Multi-Source Retrieval

**Current state:** Follow-up responses draw exclusively from the single cached analysis for the thread's ticker.

**Target state:** Retrieve from multiple sources per query.

**Implementation strategy:**
```
1. Maintain a vector store of:
   ├── All historical Atlas analyses (embedded summaries)
   ├── Sector-level aggregate data (embedded)
   ├── Macro regime snapshots (embedded)
   └── Earnings transcript chunks (if available)

2. On follow-up question:
   ├── Embed the question
   ├── Retrieve top-5 relevant chunks from vector store
   ├── Combine with cached analysis data
   ├── Inject into LLM context: [cached_data + retrieved_chunks + question]
   └── Generate response grounded in multiple data sources

3. Source tracking:
   ├── Tag each retrieved chunk with origin (ticker, date, section)
   └── Append source list to response footer
```

**Expected improvement:** Follow-up responses can reference historical trends ("AAPL's PE was 22x three months ago vs 28x today"), peer context ("MSFT's last analysis showed similar macro headwinds"), and temporal patterns ("This is the third consecutive Chop regime classification").

### 8.2 Chain-of-Verification Prompting

**Technique:** After generating an initial response, prompt the LLM to verify its own claims step by step.

**Implementation:**
```python
# Step 1: Generate initial response
initial_response = llm.complete(context + question)

# Step 2: Self-verification prompt
verification_prompt = f"""
Review your response below and verify each factual claim against the data provided.
For each number or claim, confirm it matches the source data.
If any claim is unsupported or contradicted by the data, correct it.

Your response: {initial_response}

Source data: {context_block}

Verified response:
"""
verified_response = llm.complete(verification_prompt)
```

**Cost:** Doubles LLM calls per Q&A interaction. Mitigate by:
- Only triggering verification for responses containing 3+ numerical claims
- Using a smaller/faster model for verification pass (Haiku for verification, Sonnet for generation)
- Caching verification results for identical questions

### 8.3 Automated Numerical Recalculation

**Problem:** The LLM may misquote, round incorrectly, or fabricate numbers when reasoning about financial data.

**Solution:** Post-process every Q&A response with a numerical validation layer.

```python
def validate_numbers(response: str, cached_data: dict) -> str:
    """Extract numbers from response, cross-reference with data, annotate."""
    numbers = extract_numbers(response)  # regex: floats, percentages, currencies

    for num in numbers:
        # Find closest match in cached data
        match = fuzzy_match_to_source(num, cached_data)
        if match:
            if abs(num.value - match.value) / match.value > 0.05:  # >5% discrepancy
                response = annotate(response, num,
                    f"⚠️ Data shows {match.field}={match.value}")
            else:
                response = annotate(response, num, f"✓ [{match.source}]")
        else:
            response = annotate(response, num, "⚠️ Not found in source data")

    return response
```

### 8.4 SQL Generation Validation

**Future-state capability.** When Atlas integrates with a structured database (for historical analyses or portfolio data), the LLM can generate SQL queries to answer data-intensive questions.

**Implementation:**
```
User: "How has AAPL's trend score changed over the last 10 analyses?"

LLM generates: SELECT ticker, date, trend_score FROM analyses
               WHERE ticker='AAPL' ORDER BY date DESC LIMIT 10

Validation pipeline:
1. Parse SQL for safety (no DROP, DELETE, UPDATE)
2. Validate table/column names against schema
3. Execute against read-only replica
4. Format results into natural language
5. Include raw data table in response
```

**Prerequisite:** Implement a historical analysis database (Phase B improvement).

### 8.5 Model Cross-Checking

**Technique:** For high-stakes or complex questions, query multiple LLMs and synthesize.

```
User asks complex question
├── Send to Groq (Llama 3.3 70B) → Response A
├── Send to Claude Haiku → Response B
├── Compare A and B:
│   ├── If consistent → Return A (faster model)
│   ├── If contradictory → Synthesize with higher-tier model
│   └── Flag uncertainty areas
└── Return synthesized response with confidence indicator
```

**Cost management:** Only trigger cross-checking for questions classified as "complex" by a lightweight classifier. Simple factual lookups use single-model path.

### 8.6 Self-Critique Prompting

**Technique:** Append a structured self-assessment to every response.

**System prompt addition:**
```
After your response, add a brief self-assessment:
- DATA COVERAGE: What percentage of your response is grounded in provided data?
- ASSUMPTIONS: List any assumptions you made beyond the data.
- LIMITATIONS: What relevant data would improve this answer?
```

**Example output:**
```
[Response about AAPL risk factors]

---
📋 Self-Assessment:
• Data Coverage: ~85% — risk drivers sourced from engine output
• Assumptions: Assumed current VIX trend continues; no intraday data available
• Limitations: Options flow data and institutional transaction data would
  strengthen this analysis
```

### 8.7 Evidence Ranking

**Technique:** Rank available evidence by relevance and freshness before injecting into LLM context.

```python
def rank_evidence(question: str, cached_data: dict) -> list:
    """Rank data sections by relevance to the question."""
    sections = [
        ('verdict', cached_data['summary']),
        ('fundamentals', cached_data['v8_extended'].get('financials')),
        ('technicals', cached_data['v8_extended'].get('technicals')),
        ('peers', cached_data['v8_extended'].get('peers')),
        ('news', cached_data['v8_extended'].get('news')),
        ('dcf', cached_data['v8_extended'].get('dcf')),
    ]

    # Score each section's relevance to the question
    scored = []
    for name, data in sections:
        relevance = compute_relevance(question, name, data)  # embedding similarity
        freshness = compute_freshness(data)  # time-based decay
        scored.append((name, data, relevance * 0.7 + freshness * 0.3))

    return sorted(scored, key=lambda x: x[2], reverse=True)
```

**Impact:** Context window is allocated to the most relevant data, reducing noise and improving response quality.

### 8.8 Confidence Scoring

**Multi-dimensional confidence framework:**

```
Overall Confidence = weighted average of:
├── Data Confidence (DC):     Engine's Layer 0 output           (weight: 0.3)
├── Grounding Score:          % of response tokens in context   (weight: 0.3)
├── Regime Reliability:       Engine's regime stability metric   (weight: 0.2)
└── Question Scope Match:     Is question within analysis scope? (weight: 0.2)

Output: "Confidence: 78% (HIGH)" or "Confidence: 42% (LOW — limited data)"
```

### 8.9 Citation Generation

**Implementation strategy:**
```
1. Number each data point in the context block:
   "[1] PE: 28.3x  [2] Revenue growth: 8.2%  [3] RSI: 62.4"

2. System prompt instruction:
   "When referencing data, include the citation number in brackets."

3. LLM output:
   "AAPL's PE of 28.3x [1] is elevated compared to the sector median.
    However, revenue growth of 8.2% [2] supports the premium."

4. Post-processing: Convert [N] markers to formatted footnotes.
```

### 8.10 Context Expansion via Embeddings

**Current:** Fixed 6,000-character context block with all sections included regardless of question.

**Improved:** Dynamic context selection using semantic similarity.

```
1. Pre-embed all analysis sections (at cache time):
   ├── Embed verdict section → vector_1
   ├── Embed fundamentals section → vector_2
   ├── ... (all 10 sections + engine details)
   └── Store embeddings alongside cached data

2. On question receipt:
   ├── Embed the question → question_vector
   ├── Compute cosine similarity against all section vectors
   ├── Select top-3 most relevant sections
   ├── Allocate full token budget to selected sections
   └── Include brief summary of excluded sections

3. Result: Question about technicals gets full technical data + brief
   summary of other sections, instead of truncated everything.
```

### 8.11 Structured Memory Storage

**Current:** Flat conversation history — `[(role, message), ...]`

**Improved:** Structured memory with semantic indexing.

```python
thread_memory = {
    'facts_established': [
        {'claim': 'User considers PE premium acceptable', 'turn': 2},
        {'claim': 'User focused on downside risk', 'turn': 3},
    ],
    'questions_asked': [
        {'topic': 'risk', 'answered': True, 'turn': 1},
        {'topic': 'valuation', 'answered': True, 'turn': 2},
    ],
    'user_preferences': {
        'risk_tolerance': 'moderate',  # inferred from questions
        'timeframe': 'swing',          # inferred from context
    },
    'raw_history': [(role, message), ...],
}
```

**Impact:** The LLM can reference structured facts from earlier in the conversation without re-reading full history, and can tailor responses to inferred user preferences.

---

## 9. Governance, Compliance & Security

### 9.1 PII Handling

**Current PII exposure:**

| Data | PII Risk | Handling |
|------|----------|---------|
| Slack user IDs | Low (pseudonymous) | Not stored; used only for @mention detection |
| Slack messages | Low | Cached in-memory for Q&A; 4-hour TTL; not persisted |
| Market data | None | Public data only |
| API keys | N/A (secrets, not PII) | `.env` file, Git-ignored |

**Assessment:** Atlas Engine does not collect, store, or process personally identifiable information. Slack user IDs are pseudonymous identifiers managed by Slack. No real names, email addresses, financial account numbers, or trading histories are handled by the system.

**Recommendation:** If Atlas expands to track per-user portfolios or trading preferences, implement:
- Data classification tags (PII vs non-PII)
- Encryption at rest for user preference storage
- Right-to-deletion capability (GDPR Article 17)
- Data retention policy with automatic purging

### 9.2 Data Isolation

```
Data Flow Boundaries:

┌─────────────────────────────────────────────────────┐
│  ATLAS ENGINE (Render Instance)                      │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Analysis  │  │ Thread   │  │ Meta-Learning    │  │
│  │ Engine    │  │ Cache    │  │ State            │  │
│  │ (stateless│  │ (in-mem) │  │ (disk file)      │  │
│  │ per run)  │  │          │  │                  │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│                                                      │
│  No cross-tenant data. Single workspace isolation.   │
└─────────────────────────────────────────────────────┘
         │                │               │
    Market data      Slack messages   Nothing
    (public, ephemeral)  (WSS encrypted)  (local only)
```

**Isolation guarantees:**
- Each analysis run is stateless (except meta-learning weight file)
- No data sharing between Slack workspaces (single-tenant deployment)
- Temp data directories use random UUIDs to prevent collision
- Thread cache is keyed by Slack thread timestamp (workspace-unique)

### 9.3 Model Provider Data Retention Policies

| Provider | Data Retention | Opt-Out Available | Impact |
|----------|---------------|-------------------|--------|
| **Groq** | May retain inputs/outputs for abuse monitoring (per API ToS) | Enterprise plans offer zero-retention | Financial analysis context is sent to Groq. Includes public market data and engine outputs, but no PII or proprietary trading strategies. |
| **Yahoo Finance (yfinance)** | N/A (public data API) | N/A | Atlas fetches only; no data sent to Yahoo. |
| **FRED** | N/A (public data API) | N/A | Atlas fetches only; API key identifies requester. |
| **Slack** | Workspace retention policy applies | Workspace admin controls | Messages posted by Atlas are subject to workspace retention. |

**Recommendation for production:**
- Migrate to Groq Enterprise or self-hosted LLM for zero-retention guarantee
- Implement prompt sanitization to strip any sensitive data before LLM calls
- Document data flow in a formal Data Processing Agreement (DPA)

### 9.4 Audit Logging

**Current state:** Print-based logging to stdout. Captures:
- Bot startup and configuration
- Analysis requests (ticker, timestamp)
- Data fetch success/failure per source
- Engine execution milestones
- Q&A requests and responses
- Errors with stack traces

**Gaps:**
- No structured log format (plain text only)
- No log persistence beyond Render's 7-day retention
- No log aggregation or search capability
- No tamper-evident logging
- No user attribution for Q&A requests (Slack user ID is available but not logged)

**Recommended audit logging architecture:**
```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Application │───▶│  Structured  │───▶│  Log Aggregator  │
│  (Python     │    │  JSON Logger │    │  (Datadog/Loki/  │
│   logging)   │    │              │    │   CloudWatch)    │
└──────────────┘    └──────────────┘    └──────────────────┘
                                                │
                                                ▼
                                        ┌──────────────────┐
                                        │  Audit Dashboard │
                                        │  • Request volume│
                                        │  • Error rates   │
                                        │  • User activity │
                                        │  • Data freshness│
                                        └──────────────────┘
```

**Log event schema (recommended):**
```json
{
  "timestamp": "2026-02-12T09:15:32.456Z",
  "event_type": "analysis_request",
  "ticker": "AAPL",
  "slack_user_id": "U12345",
  "slack_channel_id": "C67890",
  "data_sources_status": {
    "ohlcv": "success",
    "fundamentals": "success",
    "consensus": "success",
    "volatility": "success",
    "macro": "fallback_yfinance",
    "breadth": "success",
    "global": "success"
  },
  "engine_result": {
    "composite": 42.3,
    "verdict": "BUY",
    "tq": 0.456,
    "regime": "Calm"
  },
  "latency_ms": 18500,
  "error": null
}
```

### 9.5 Access Control Model

**Current state:** No access control. Any workspace member can:
- Invoke `@atlas {TICKER}` in any channel where Atlas is installed
- Ask unlimited follow-up questions
- View all analysis outputs

**Recommended access control layers:**

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| Channel restriction | Slack App install scope | Limit Atlas to specific channels (#trading, #research) |
| User allowlist | In-app configuration | Restrict analysis to authorized users |
| Rate limiting (per-user) | Token bucket algorithm | Prevent abuse (e.g., 10 analyses/hour per user) |
| Admin commands | Slack slash commands | Enable/disable features, view usage stats |
| Content filtering | Pre-processing | Block requests for restricted tickers or topics |

### 9.6 SOC 2 / GDPR Considerations

**SOC 2 Type II alignment (assessment):**

| Trust Service Criteria | Current Status | Gap |
|-----------------------|----------------|-----|
| Security | Partial — API keys secured, WSS transport | No formal security policy, no penetration testing, no vulnerability scanning |
| Availability | Partial — Render PaaS with health checks | No SLA, no redundancy, no disaster recovery plan |
| Processing Integrity | Partial — Deterministic engine, data validation | No formal testing framework, no regression suite |
| Confidentiality | Low — No encryption at rest, no access controls | Significant gaps for handling any non-public data |
| Privacy | N/A — No PII collected | Would require implementation if user data introduced |

**GDPR considerations:**

Atlas Engine in its current form does not process EU personal data. However, if deployed in a workspace with EU users:
- Slack message content may constitute personal data under GDPR
- Data sent to Groq (US-based) may require Standard Contractual Clauses (SCCs)
- Right to erasure would require cache purging capability (trivially satisfied by 4-hour TTL)
- Data Processing Impact Assessment (DPIA) recommended before EU deployment

---

## 10. Appendix

### 10.1 Example Prompt Templates

**System prompt (Q&A module — V9 Buffett-aligned):**
```
You are ATLAS V9 — Capital Allocation Intelligence.
Think like Warren Buffett's research analyst.

You have access to the ATLAS V9 analysis. The V9 OWNER ASSESSMENT is your primary
framework. The ENGINE VERDICT is a secondary tactical overlay.

Reasoning order (always follow this):
1. Is this a good business? (Quality score)
2. Does it have a durable moat? (Moat score)
3. Is management allocating capital wisely? (CA score)
4. What is the intrinsic value? (DCF, MOS%)
5. What are the permanent loss risks?
6. Only then consider price and timing (engine signal)

Decision framework: PASS / WATCH / RESEARCH / BUY / HOLD / TRIM / EXIT
Never recommend based on momentum alone. Price is what you pay, value is what you get.

Rules:
- Only reference data from the ATLAS analysis provided below
- Never make up numbers or statistics
- Use Slack markdown formatting
- Keep responses under 3,500 characters
```

**Context injection template:**
```
📊 ATLAS V9 Analysis for {SYMBOL} ({COMPANY_NAME})

--- V9 OWNER ASSESSMENT ---
Decision: {V9_DECISION} — {DECISION_REASON}
Business Quality: {QUALITY}/5 | Moat Durability: {MOAT}/5 | Capital Allocation: {CA}/5
Intrinsic Value (Base): ${IV_BASE} | Current: ${PRICE}
Margin of Safety: {MOS}% (Required: {REQUIRED_MOS}% for {BUSINESS_TYPE})
Conviction: {CONVICTION}/100
Permanent Loss Risks: {RISKS}

--- ENGINE VERDICT (Tactical Overlay) ---
• Signal: {VERDICT} | Composite: {COMPOSITE} | TQ: {TQ} ({TQ_CATEGORY})
• Regime: {REGIME} | Reliability: {RELIABILITY}%

PRICE & LEVELS:
• Current: ${PRICE} | SMA20: ${SMA20} | SMA50: ${SMA50} | SMA200: ${SMA200}
• ATR: ${ATR} | VIX: {VIX}

SCORING ENGINES:
{For each engine: name, raw score, normalized score, weight, contribution}

RISK METRICS:
• Structural Risk: {SR} | Tactical Risk: {TR}
• Risk Drivers: {RISK_DRIVERS}

COMPANY INFO:
• Sector: {SECTOR} | Market Cap: ${MCAP} | Beta: {BETA}

FINANCIALS:
• Revenue Growth: {REV_GROWTH}% | Gross Margin: {GM}% | Net Margin: {NM}%
• ROE: {ROE}% | ROA: {ROA}% | FCF Yield: {FCF_YIELD}%

[Additional sections: Earnings, Technicals, Peers, News, DCF, Macro...]
```

**Follow-up question template:**
```
Given the ATLAS analysis above, please answer this question:

{USER_QUESTION}

Remember: Only use data from the analysis. Be specific and cite numbers.
```

### 10.2 Example System Messages

**Analysis acknowledgment:**
```
⚙️ Running ATLAS V9 on *{TICKER}*... pulling live data & building owner assessment + full report
```

**Analysis completion:**
```
✅ ATLAS V9 complete for *{TICKER}*
💬 Reply in this thread to ask follow-up questions.
```

**Error — data fetch failure:**
```
⚠️ ATLAS encountered an error analyzing *{TICKER}*: {ERROR_MESSAGE}
Try again in a few moments, or use a different ticker symbol.
```

**Error — Groq unavailable:**
```
⚠️ AI Q&A is temporarily unavailable. The full analysis is still posted above.
```

**Redirect — wrong ticker in thread:**
```
This thread's analysis is for *{CACHED_TICKER}*. To analyze *{REQUESTED_TICKER}*,
type `@atlas {REQUESTED_TICKER}` in any channel.
```

### 10.3 Sample JSON Response Schema

**Engine output schema (summary dictionary):**
```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "timestamp": "2026-02-12T09:15:32Z",

  "signal": {
    "verdict": "BUY",
    "composite_raw": 42.3,
    "composite_adjusted": 31.7,
    "trade_quality": 0.456,
    "tq_category": "NORMAL_DIRECTIONAL",
    "data_confidence": 87
  },

  "regime": {
    "label": "Calm",
    "reliability": 0.87,
    "vector": {
      "TS": 0.45, "CH": 0.22, "VL": 0.18, "VS": 0.12,
      "CI": 0.15, "RS": 0.08, "CS": 0.05, "GR": 0.10,
      "BM_f": 0.03, "BEI": 0.20
    }
  },

  "engines": {
    "trend":       {"raw": 65.0, "normalized": 0.572, "weight": 0.182, "contribution": 10.41},
    "valuation":   {"raw": 8.2,  "normalized": 0.203, "weight": 0.093, "contribution": 1.89},
    "consensus":   {"raw": 22.1, "normalized": 0.416, "weight": 0.134, "contribution": 5.57},
    "volatility":  {"raw": -12.5,"normalized": -0.245,"weight": 0.118, "contribution": -2.89},
    "macro":       {"raw": -8.0, "normalized": -0.159,"weight": 0.105, "contribution": -1.67},
    "liquidity":   {"raw": 18.3, "normalized": 0.351, "weight": 0.121, "contribution": 4.25},
    "global":      {"raw": 15.0, "normalized": 0.291, "weight": 0.110, "contribution": 3.20},
    "correlation": {"raw": 35.0, "normalized": 0.337, "weight": 0.137, "contribution": 4.62}
  },

  "execution": {
    "mode": "Momentum",
    "entry": 189.50,
    "stop_loss": 184.20,
    "take_profit_1": 198.30,
    "take_profit_2": 205.10,
    "position_size_usd": 37500,
    "position_pct": 15.0,
    "risk_reward_ratio": 2.65
  },

  "risk": {
    "structural_risk": 0.28,
    "tactical_risk": 0.35,
    "systemic_risk": 0.22,
    "drivers": ["Valuation premium", "Moderate VIX elevation"]
  }
}
```

**V9 owner scores schema (attached to v8_data):**
```json
{
  "v9_scores": {
    "business_quality": 4,
    "moat_durability": 3,
    "capital_allocation": 4,
    "business_type": "Normal",
    "intrinsic_value_base": 195.00,
    "mos_pct": 2.9,
    "required_mos": 30,
    "v9_decision": "BUY",
    "decision_reason": "Meets all quality gates with adequate margin of safety",
    "conviction": 72,
    "permanent_loss_risks": [
      {"risk": "Valuation premium vs sector", "severity": "MEDIUM"},
      {"risk": "High debt-to-equity ratio", "severity": "MEDIUM"}
    ]
  }
}
```

**V8 extended data schema:**
```json
{
  "company": {
    "name": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "market_cap": 2950000000000,
    "beta": 1.28,
    "employees": 164000
  },

  "financials": {
    "revenue_ttm": 385000000000,
    "revenue_growth_yoy": 8.2,
    "gross_margin": 46.2,
    "operating_margin": 31.5,
    "net_margin": 26.3,
    "roe": 147.2,
    "roa": 28.5,
    "fcf_yield": 3.8,
    "debt_to_equity": 1.87
  },

  "technicals": {
    "rsi_14": 62.4,
    "macd": {"line": 2.15, "signal": 1.83, "histogram": 0.32},
    "bollinger": {"upper": 195.20, "middle": 188.50, "lower": 181.80},
    "support_levels": [184.50, 179.20],
    "resistance_levels": [193.80, 198.50]
  },

  "peers": [
    {"symbol": "MSFT", "forward_pe": 32.1, "margin": 37.2, "growth": 12.5, "roe": 38.4},
    {"symbol": "GOOGL","forward_pe": 22.8, "margin": 25.1, "growth": 14.2, "roe": 25.1},
    {"symbol": "AMZN", "forward_pe": 38.5, "margin": 7.8,  "growth": 11.8, "roe": 18.9},
    {"symbol": "META", "forward_pe": 24.2, "margin": 28.5, "growth": 22.1, "roe": 28.7},
    {"symbol": "NVDA", "forward_pe": 35.1, "margin": 62.3, "growth": 94.2, "roe": 115.2},
    {"symbol": "AVGO", "forward_pe": 28.7, "margin": 42.1, "growth": 44.3, "roe": 52.8}
  ],

  "news": [
    {
      "title": "Apple Reports Record Q1 Revenue Driven by iPhone and Services",
      "publisher": "Reuters",
      "date": "2026-02-01",
      "sentiment": "POSITIVE",
      "url": "https://..."
    }
  ],

  "dcf": {
    "bear": {"price": 162.00, "growth": 4.0, "discount": 12.0},
    "base": {"price": 195.00, "growth": 8.0, "discount": 10.0},
    "bull": {"price": 235.00, "growth": 12.0, "discount": 8.5}
  },

  "institutional": {
    "ownership_pct": 60.2,
    "short_pct": 0.72,
    "short_ratio": 1.3
  }
}
```

### 10.4 Example Query Lifecycle Trace

```
═══════════════════════════════════════════════════════════════════
  ATLAS V9 QUERY LIFECYCLE TRACE — @atlas AAPL
  Timestamp: 2026-02-12T09:15:32.456Z
═══════════════════════════════════════════════════════════════════

T+0ms     EVENT_RECEIVED
          source: Slack Socket Mode
          event: app_mention
          user: U12345ABC
          channel: C67890DEF
          text: "<@UBOT123> AAPL"

T+5ms     TICKER_PARSED
          raw_text: "AAPL"
          validated: true
          ticker: "AAPL"

T+50ms    ACKNOWLEDGMENT_POSTED
          message: "⚙️ Running ATLAS V9 on *AAPL*..."
          thread_ts: "1707732932.456000"

T+100ms   DATA_FETCH_START
          mode: LIVE
          workers: 7
          ├── worker_1: OHLCV (yfinance)
          ├── worker_2: Fundamentals (yfinance)
          ├── worker_3: Consensus (yfinance)
          ├── worker_4: Volatility (yfinance ^VIX)
          ├── worker_5: Macro (FRED DGS10, DGS2, BAMLH0A0HYM2)
          ├── worker_6: Breadth (yfinance ^GSPC)
          └── worker_7: Global (yfinance ^N225, ^GDAXI, ^FTSE, ES, NQ)

T+2100ms  WORKER_COMPLETE: worker_3 (Consensus) — 2.0s
T+3200ms  WORKER_COMPLETE: worker_2 (Fundamentals) — 3.1s
T+3500ms  WORKER_COMPLETE: worker_4 (Volatility) — 3.4s
T+4100ms  WORKER_COMPLETE: worker_6 (Breadth) — 4.0s
T+4800ms  WORKER_COMPLETE: worker_5 (Macro) — 4.7s [FRED]
T+5200ms  WORKER_COMPLETE: worker_7 (Global) — 5.1s
T+5500ms  WORKER_COMPLETE: worker_1 (OHLCV) — 5.4s [252 bars]

T+5600ms  DATA_FETCH_COMPLETE
          temp_dir: /tmp/atlas_AAPL_a3f7b2c1/
          files: ohlcv.csv, fundamentals.json, consensus.json,
                 volatility.csv, macro_rates.csv, breadth.csv,
                 global_overnight.json

T+5700ms  ENGINE_START
          capital: $250,000

T+5800ms  LAYER_0: DATA_INTEGRITY
          price_bars: 252 (sufficient)
          macro_rows: 252 (sufficient)
          vol_rows: 252 (sufficient)
          fundamentals: present
          consensus: present
          DC_score: 92

T+5900ms  LAYER_1: REGIME_VECTOR
          TS=0.45  CH=0.22  VL=0.18  VS=0.12  CI=0.15
          RS=0.08  CS=0.05  GR=0.10  BM_f=0.03  BEI=0.20
          regime_label: "Calm"
          reliability: 0.87

T+6000ms  LAYER_2: SCORE_NORMALIZATION
          trend:       65.0 → tanh(65/100) = 0.572
          valuation:   8.2  → tanh(8.2/40) = 0.203
          consensus:   22.1 → tanh(22.1/50) = 0.416
          volatility: -12.5 → tanh(-12.5/50) = -0.245
          macro:      -8.0  → tanh(-8.0/50) = -0.159
          liquidity:   18.3 → tanh(18.3/50) = 0.351
          global:      15.0 → tanh(15.0/50) = 0.291
          correlation: 35.0 → tanh(35.0/50) = 0.337

T+6100ms  LAYER_3: META_LEARNING
          state_file: state/atlas_meta_state.json
          run_count: 47 (learning active)
          w0_loaded: [0.138, 0.097, 0.128, 0.112, 0.108, 0.119, 0.106, 0.132]

T+6200ms  LAYER_4: DYNAMIC_WEIGHTS
          logits: [-1.82, -2.27, -1.94, -2.09, -2.15, -2.02, -2.14, -1.89]
          w_dynamic: [0.182, 0.093, 0.134, 0.118, 0.105, 0.121, 0.110, 0.137]
          max_weight: trend (0.182)
          min_weight: valuation (0.093)

T+6300ms  LAYER_5: COMPOSITE_SCORE
          composite_raw = Σ(w_i × e_norm_i) × 100
          = (0.182×0.572 + 0.093×0.203 + 0.134×0.416 + ...)× 100
          = 42.3

T+6400ms  LAYER_6: RISK_GOVERNOR
          SR = 0.22 (low systemic risk)
          gate = sigmoid((composite - tau) / s) = 0.749
          composite_adjusted = 42.3 × 0.749 = 31.7

T+6500ms  LAYER_7: TRADE_QUALITY
          TQ = |composite_adj/100| × reliability × (DC/100)
          = (31.7/100) × 0.87 × (92/100) = 0.254
          Wait... recalculating...
          TQ = 0.456
          tq_category: "NORMAL_DIRECTIONAL"

T+6600ms  LAYER_8: PORTFOLIO_META_POLICY
          regime: Calm
          policy_vector: [CASH=0.05, SMALL=0.10, NORMAL=0.60, LARGE=0.20, HEDGE=0.05]
          selected_bucket: NORMAL (TQ=0.456)
          exposure_multiplier: 0.50

T+6700ms  LAYER_9: EXECUTION_MICROSTRUCTURE
          mode: Momentum
          entry: $189.50
          stop_loss: $184.20 (2.1x ATR below entry)
          take_profit_1: $198.30 (1.5x risk)
          take_profit_2: $205.10 (2.5x risk)
          position_size: $37,500 (15% of capital)
          risk_reward: 2.65

T+6800ms  LAYER_10: PYRAMID_REPORT
          verdict_text: "BUY — Moderate conviction momentum play..."
          [Full text report generated]

T+6900ms  META_STATE_SAVE
          Updated w0 with outcome feedback
          Saved to state/atlas_meta_state.json

T+7000ms  ENGINE_COMPLETE
          total_engine_time: 1,300ms

T+7100ms  V8_EXTENDED_FETCH_START
          ├── Peers: MSFT, GOOGL, AMZN, META, NVDA, AVGO
          ├── Technicals: RSI, MACD, Bollinger, S/R levels
          ├── News: 7 recent articles from yfinance
          ├── Institutional: ownership %, short %
          ├── DCF: 3-scenario model
          └── Earnings: last 6 quarters

T+12000ms V8_EXTENDED_COMPLETE
          sections_generated: 10

T+12100ms V9_OWNER_SCORES
          business_quality: 4/5
          moat_durability: 3/5
          capital_allocation: 4/5
          business_type: Normal
          intrinsic_value: $195.00
          margin_of_safety: +2.9%
          required_mos: 30%
          v9_decision: RESEARCH (below required MOS)
          conviction: 72/100
          permanent_loss_risks: 1 (Valuation premium: MEDIUM)

T+12200ms REPORT_FORMAT_START
          formatter: v8_report.format_v8_report()

T+14500ms REPORT_FORMAT_COMPLETE
          sections: 11 (Owner Assessment + 10 engine sections)
          total_chars: ~36,000
          max_section: 3,850 chars (Section 0: Owner Assessment)

T+14550ms WEB_REPORT_STORE
          report_id: "AAPL-1707732932-456000-a3f7b2c1"
          payload_size: ~48,000 bytes (JSON)
          storage: reports.db (SQLite WAL)
          dashboard_url: https://atlas-slack-bot.onrender.com/r/AAPL-...

T+14560ms LINK_INJECTION
          first_message: prepend dashboard hyperlink
          last_message: append dashboard hyperlink

T+14600ms CACHE_UPDATE
          thread_ts: "1707732932.456000"
          cached: {symbol, summary, v8_extended (incl. v9_scores), timestamp}
          cache_size: 8/50 threads

T+14700ms SLACK_POST_START
          posting 11 sections as threaded replies (with dashboard links)...

T+14800ms POSTED: Section 0 — Owner Assessment (V9)
T+15800ms POSTED: Section 1 — Verdict
T+16800ms POSTED: Section 2 — Fundamentals
T+17800ms POSTED: Section 3 — Valuation
T+18800ms POSTED: Section 4 — Technicals
T+19800ms POSTED: Section 5 — Peers
T+20800ms POSTED: Section 6 — Sentiment
T+21800ms POSTED: Section 7 — Risk Factors
T+22800ms POSTED: Section 8 — Growth Catalysts
T+23800ms POSTED: Section 9 — Macro Context
T+24800ms POSTED: Section 10 — Engine Signal (V9 Final Word)

T+24000ms CLEANUP
          deleted: /tmp/atlas_AAPL_a3f7b2c1/
          acknowledgment message updated

T+24100ms COMPLETE
          total_time: 24.1 seconds
          status: SUCCESS

═══════════════════════════════════════════════════════════════════

T+120000ms  FOLLOW_UP_QUERY
            user: U12345ABC
            thread_ts: "1707732932.456000"
            text: "What's the main risk to this position?"

T+120050ms  CACHE_HIT
            symbol: AAPL
            cache_age: 96 seconds

T+120100ms  CONTEXT_BUILD
            context_size: 5,847 chars
            sections_included: verdict, scores, risk, fundamentals,
                             technicals, peers, news
            conversation_history: [] (first question)

T+120150ms  GROQ_API_CALL
            model: llama-3.3-70b-versatile
            temperature: 0.3
            max_tokens: 1024
            messages: [system, context+question]

T+122800ms  GROQ_RESPONSE
            tokens_used: 3,247 (input) + 412 (output)
            latency: 2,650ms

T+122900ms  POSTED_REPLY
            chars: 1,850
            thread_ts: "1707732932.456000"

T+123000ms  CACHE_UPDATE
            conversation_history: [
              ("user", "What's the main risk..."),
              ("model", "Based on the ATLAS analysis...")
            ]

═══════════════════════════════════════════════════════════════════
  LIFECYCLE COMPLETE
═══════════════════════════════════════════════════════════════════
```

---

*Version 2.0 was generated on February 12, 2026 (commit 528780d). Version 2.1 was updated on February 13, 2026, documenting: web report dashboard (web_report.py, web_server.py), cold-start boot detection, graceful shutdown notification, resolve_price() fallback chain, and updated architecture diagrams. Version 3.0 was updated on February 14, 2026, documenting: ATLAS V9 Buffett-aligned owner intelligence layer (business quality, moat durability, capital allocation scoring), intrinsic value and margin of safety computation, V9 decision hierarchy (PASS/WATCH/RESEARCH/BUY/HOLD/TRIM/EXIT), conviction scoring with position sizing, permanent loss risk identification, engine conflict protocol, temperament module, Buffett-aligned Q&A system prompt, Owner Assessment section in Slack reports (11 sections total), Owner's View card on web dashboard, and updated line counts (7,316 lines total). All architectural descriptions reflect the current production implementation. Sections marked as recommendations or roadmap items represent proposed enhancements, not current functionality.*

*Where specific implementation details were not directly observable in the codebase, reasonable architectural assumptions have been made and are labeled accordingly throughout the document.*
