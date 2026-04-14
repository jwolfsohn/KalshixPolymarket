# Kalshi x Polymarket Arbitrage Bot - Implementation Plan

## Context

Prediction markets on Kalshi and Polymarket frequently price the same real-world events differently. When the combined cost of buying YES on one platform and NO on the other totals less than $1.00, a risk-free profit exists regardless of the outcome. Cross-platform arbitrage traders extracted ~$40M from Polymarket alone between April 2024 and April 2025, and this remains the most consistent edge in 2026. The goal is to build an automated bot that continuously scans both platforms, detects these mispricings, and executes trades to capture the spread.

---

## What You Need to Provide

| Item | Details | How to Get It |
|------|---------|---------------|
| **Kalshi Account** | US-based, KYC-verified account | Sign up at kalshi.com (US residents only) |
| **Kalshi API Key** | RSA key pair for API authentication | Account Settings > API Keys > Generate |
| **Polymarket Wallet** | Ethereum private key for signing orders | Create or use existing Ethereum wallet |
| **USDC on Polygon** | Funds for Polymarket trading | Bridge USDC to Polygon network |
| **Kalshi Funding** | USD balance on Kalshi | ACH deposit (free) or debit card (2% fee) |
| **Starting Capital** | $500 minimum, $2,000+ recommended | Thin spreads (2-4%) need larger size to be meaningful |
| **Node.js** | Required for PMXT sidecar process | `brew install node` if not installed |

---

## How the Arbitrage Works

### Strategy 1: Cross-Platform Arbitrage (Primary)
Same event priced differently across platforms. Example:
- "Will the Fed cut rates in June?" YES costs 42c on Kalshi, NO costs 57c on Polymarket
- Total cost: 99c. Guaranteed payout: $1.00. Profit: 1c/contract
- Observed spreads: 1.5% to 4.5% on high-volume events

### Strategy 2: Bundle Arbitrage (Intra-Platform)
YES + NO on the same platform don't sum to $1.00. Rarer, quickly arbitraged.

### Strategy 3: Multi-Outcome Arbitrage
Markets with N outcomes (e.g., "Who wins the election?") where the sum of all YES prices != $1.00.

### Fee Constraints (Critical)
- **Kalshi**: Taker fee = `roundup(0.07 * P * (1-P))`, max $0.02/contract. Maker: 1/4 of taker rate.
- **Polymarket**: Taker fees 0.75%-1.80% by category (bell curve: max at p=0.50, near-zero at extremes). Makers: free + rebates.
- **Key rule**: Combined fees of ~3-5% mean only spreads above that threshold are profitable. Contracts at extreme prices (90c+) have much lower fees, making those spreads more exploitable.

---

## Architecture

**Core decision**: Use [PMXT](https://github.com/pmxt-dev/pmxt) (`pip install pmxt`) as the unified API layer. It's "CCXT for prediction markets" - one interface for both Kalshi and Polymarket, normalized market/outcome objects, and a free data archive for backtesting. The ~10-50ms sidecar overhead is irrelevant since prediction market opportunities persist for minutes to hours.

---

## Project Structure

```
KalshixPolymarket/
├── pyproject.toml
├── config.yaml                 # All runtime parameters
├── .env                        # API keys (gitignored)
├── .env.example
├── .gitignore
│
├── src/
│   ├── config.py               # YAML + env loading, typed BotConfig dataclass
│   │
│   ├── data/
│   │   ├── fetcher.py          # PMXT market data fetching
│   │   ├── orderbook.py        # Order book depth analysis
│   │   └── archive.py          # PMXT data archive for backtesting
│   │
│   ├── matching/
│   │   ├── fuzzy.py            # Jaccard + Levenshtein similarity
│   │   ├── matcher.py          # 3-stage market pairing (cache → structural → fuzzy)
│   │   └── cache.py            # SQLite-backed match cache
│   │
│   ├── arb/
│   │   ├── models.py           # Opportunity, Spread, FeeEstimate dataclasses
│   │   ├── fees.py             # Platform-specific fee calculators
│   │   ├── cross_platform.py   # Strategy 1
│   │   ├── bundle.py           # Strategy 2
│   │   ├── multi_outcome.py    # Strategy 3
│   │   └── scanner.py          # Unified scanner orchestrating all strategies
│   │
│   ├── execution/
│   │   ├── executor.py         # Dual-leg order placement with rollback
│   │   ├── risk.py             # Position limits, daily loss, kill switch
│   │   └── slippage.py         # Order book depth → max safe size
│   │
│   ├── portfolio/
│   │   ├── tracker.py          # Open/closed position tracking
│   │   ├── pnl.py              # Realized/unrealized P&L
│   │   └── rotation.py         # Exit when spreads close, redeploy capital
│   │
│   ├── backtest/
│   │   ├── engine.py           # Walk-forward backtester
│   │   ├── data_loader.py      # PMXT archive Parquet ingestion
│   │   └── metrics.py          # Sharpe, drawdown, win rate
│   │
│   ├── dashboard/
│   │   ├── server.py           # FastAPI web dashboard
│   │   └── alerts.py           # Discord/email notifications
│   │
│   └── utils/
│       └── logging.py          # Structured logging
│
├── scripts/
│   ├── run_bot.py              # Main entry: scan → detect → execute → track
│   ├── run_scanner.py          # Scan-only (no execution)
│   ├── run_backtest.py         # Historical analysis
│   └── run_dashboard.py        # Dashboard standalone
│
└── tests/
    ├── test_fees.py
    ├── test_matching.py
    └── test_arb_detection.py
```

---

## Key Components

### Fee Calculator (`src/arb/fees.py`)
Accuracy here is the #1 determinant of profitability. If fees are underestimated, every "profitable" trade is actually a loss.
- Kalshi: `min(ceil(0.07 * P * (1-P) * 100) / 100, 0.02)` per contract (taker)
- Polymarket: category-dependent bell curve, 0% for makers
- Always round fees **up** (conservative)

### Market Matcher (`src/matching/matcher.py`)
Three-stage pipeline:
1. **Cache lookup**: SQLite-persisted known pairs (skip re-matching)
2. **Structural pre-filter**: Reject pairs with different settlement dates, outcome counts, or categories
3. **Fuzzy text matching**: `0.6 * jaccard_words + 0.4 * levenshtein_normalized` on titles + outcome labels. Threshold: 0.65.

Outputs `MatchedMarket` with a `settlement_risk` flag ("identical" / "similar" / "different_criteria") since platforms may resolve the same event differently.

### Arbitrage Scanner (`src/arb/scanner.py`)
For each matched pair:
1. Fetch best ask YES on platform A, best ask NO on platform B
2. Check both directions (A→B and B→A)
3. Compute: `gross_profit = 1.00 - (ask_yes + ask_no)`
4. Subtract fees from both legs
5. Filter by `min_profit_after_fees` and order book depth
6. Rank by **annualized return** (a 2% spread settling in 3 days >> 3% spread settling in 6 months)

### Dual-Leg Executor (`src/execution/executor.py`)
- Place the **less liquid leg first** (if it fills, the liquid side will too; if it fails, no exposure)
- Use limit orders with tight timeout (5s)
- On leg-2 failure: automatic rollback (cancel/reverse leg 1)
- Dry-run mode logs everything without placing real orders

### Capital Rotation (`src/portfolio/rotation.py`)
Don't hold to settlement if spreads converge early:
- Exit when spread narrows to <1c (lock in profit without waiting months)
- Redeploy into fresh opportunities with higher annualized returns

---

## Development Phases

### Phase 0: Account Setup (Before Coding)
You need to create accounts on both platforms. Here's a checklist:

**Kalshi:**
1. Sign up at kalshi.com (US residents only, KYC verification required)
2. Fund account via ACH transfer (free) -- start with $250-500
3. Go to Account Settings > API Keys > Generate API Key (creates RSA key pair)
4. Save the API key ID and download the private key file (`.pem`)

**Polymarket:**
1. Create an Ethereum wallet (e.g., MetaMask) or use an existing one
2. Export and securely save the private key
3. Fund with USDC on Polygon network (bridge from Ethereum mainnet or buy directly on Polygon)
4. Start with $250-500 in USDC
5. Visit polymarket.com and connect wallet to enable trading

**Local Environment:**
1. Ensure Python 3.10+ is installed
2. Install Node.js (`brew install node`) -- required for PMXT sidecar
3. Run `npm install -g pmxtjs` to install the PMXT sidecar globally

### Phase 1: Foundation (Days 1-3)
- Project setup: `pyproject.toml`, config, `.env` handling
- PMXT integration: verify both `pmxt.Kalshi()` and `pmxt.Polymarket()` fetch markets
- Fee calculators with unit tests against known examples
- `run_scanner.py` as minimal script: fetch and print all markets from both platforms

### Phase 2: Matching + Detection (Days 4-7)
- Fuzzy matcher with `rapidfuzz` library
- Three-stage matching pipeline with SQLite cache
- All 3 arbitrage strategies with fee-aware profit calculation
- **Milestone**: `run_scanner.py` prints live arbitrage opportunities with fee-adjusted profits

### Phase 3: Backtesting (Days 8-11)
- PMXT archive data loader (hourly order book snapshots in Parquet)
- Walk-forward backtester
- Parameter sweep over `min_edge`, `order_size`, `matching_threshold`
- **Decision gate**: If backtesting shows no opportunities after fees, pivot to alert-only tool

### Phase 4: Execution + Risk (Future -- after validating profitability)
- Only build this after Phase 3 backtesting confirms profitable opportunities exist after fees
- Dual-leg executor with dry-run mode first, then live with 1-5 contracts
- Risk manager: position limits, daily loss ($50 max), kill switch ($100)
- Portfolio tracker with P&L
- FastAPI dashboard + Discord/email alerts
- Capital rotation after initial positions exist

### Phase 5: Statistical Enhancements (Future)
- Apply VECM/cointegration analysis to cross-platform spread time series
- Z-score-based entry signals (enter when spread is statistically wide, not just above a fixed threshold)
- This would be a novel contribution vs. static-threshold bots

---

## Key Dependencies

```
pmxt>=2.30.0           # Unified prediction market API
pyyaml>=6.0            # Configuration
python-dotenv>=1.0     # .env file loading
pandas>=2.0            # DataFrames
numpy>=1.24            # Numerics
rapidfuzz>=3.0         # Fast fuzzy text matching
fastapi>=0.100         # Dashboard
uvicorn>=0.23          # ASGI server
httpx>=0.24            # Async HTTP
aiosqlite>=0.19        # Async SQLite for match cache
pyarrow>=14.0          # Parquet for backtest archive
structlog>=23.0        # Structured logging
```

System: `npm install -g pmxtjs` (PMXT Node.js sidecar)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Matching false positives** | Trade on non-equivalent markets → settlement loss | `settlement_risk` flag; block "different_criteria" matches; manual override for verified pairs |
| **One-legged execution** | Unhedged directional exposure | Place illiquid leg first; timeout + auto-rollback; risk manager tracks one-leg exposure |
| **Fee miscalculation** | Every "profitable" trade is a loss | Conservative rounding; exhaustive unit tests; compare against platform calculators |
| **Capital lockup** | Low annualized returns | Rotation manager exits when spreads close; rank opportunities by annualized return |
| **PMXT sidecar crash** | Lose API access | Health check at each scan; auto-restart; native SDK fallback for critical ops |
| **Rate limits** | Orders rejected | Per-platform rate counters; queue with backoff |

---

## Verification Plan

1. **Unit tests**: Fee calculators against known platform examples, fuzzy matching against curated pairs
2. **Integration test**: `run_scanner.py` fetches live markets and finds at least some matched pairs
3. **Backtest validation**: Historical opportunities exist and survive fee deduction
4. **Dry-run soak test**: 48+ hours with full pipeline running, no real orders, verify no crashes
5. **Live micro-test**: 1-5 contracts on a single high-confidence opportunity with human oversight
6. **Dashboard verification**: All positions, P&L, and opportunities render correctly in browser
