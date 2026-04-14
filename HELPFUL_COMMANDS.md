# Helpful Commands

Quick reference for running the Kalshi x Polymarket arbitrage bot.

## Daily Workflow

### First-Time Setup (once ever)

```bash
cd /Users/jackwolfsohn/KalshixPolymarket
python3 -m venv .venv              # creates .venv/ folder
source .venv/bin/activate
pip install -e ".[dev]"
cp config.example.yaml config.yaml # then edit API keys
```

The `.venv` folder persists on disk — you only create it **once**.
This has already been done, there's a .venv created already, skip to every time you open a Terminal

### Every Time You Open a Terminal

```bash
cd /Users/jackwolfsohn/KalshixPolymarket
source .venv/bin/activate          # activates existing venv
```

Your prompt should now show `(.venv)`. Now `python` and `pytest` point to the venv.

```bash
python scripts/run_scanner.py      # run scanner
pytest tests/                      # run tests
```

### When You're Done

Just close the terminal, or run `deactivate`. **Nothing to clean up.** The venv stays on disk for next time.

### Key Rules

- **Create venv**: once, ever (unless you delete `.venv/`)
- **Activate venv**: every new terminal session
- **Shutting down**: no action needed — venv is just files on disk, not a running process
- **Forgot to activate?** You'll see `ModuleNotFoundError`. Just run `source .venv/bin/activate` and retry.

### Shortcut

If you don't want to activate, call the binary directly every time:
```bash
.venv/bin/python scripts/run_scanner.py
.venv/bin/pytest tests/
```
Same result — skips the activation step.

## Setup

```bash
# Create virtual environment (first time only)
python3 -m venv .venv

# Activate venv (every new shell)
source .venv/bin/activate

# Install project + dev deps
.venv/bin/pip install -e ".[dev]"

# Install optional ML deps (for future calibrated weather model)
.venv/bin/pip install -e ".[ml]"
```

## Configuration

```bash
# Copy example config, then edit
cp config.example.yaml config.yaml

# Edit API keys in .env (create if missing)
# Required: KALSHI_API_KEY, POLYMARKET_API_KEY
# Optional: POLYGONSCAN_API_KEY, VISUAL_CROSSING_API_KEY
```

## Running the Scanner

```bash
# Run the arbitrage + signal scanner once
.venv/bin/python scripts/run_scanner.py

# Run with a specific config
.venv/bin/python scripts/run_scanner.py --config config.yaml
```

## Running the Backtester

```bash
.venv/bin/python scripts/run_backtest.py
```

## Testing

```bash
# Run all tests
.venv/bin/pytest tests/

# Run a single test file
.venv/bin/pytest tests/test_weather_ml.py
.venv/bin/pytest tests/test_smart_money.py

# Run a single test by name
.venv/bin/pytest tests/test_smart_money.py::TestWalletScoring::test_bot_pattern_scores_low

# Verbose output
.venv/bin/pytest tests/ -v

# Stop at first failure
.venv/bin/pytest tests/ -x
```

## Enabling Strategies

Edit `config.yaml`:

```yaml
smart_money:
  enabled: true    # set to true to turn on
  min_wallet_score: 0.7

weather_ml:
  enabled: true
  min_edge: 0.10
```

## Common Dev Tasks

```bash
# Check which Python the venv points to
.venv/bin/python --version

# List installed packages
.venv/bin/pip list

# Reinstall after pyproject.toml changes
.venv/bin/pip install -e ".[dev]"

# Clear SQLite caches if schema changes
rm match_cache.db
rm -f *.db
```

## Troubleshooting

```bash
# "No module named X" → venv not activated or dep missing
source .venv/bin/activate
.venv/bin/pip install -e ".[dev]"

# "python: command not found" → use the venv binary directly
.venv/bin/python scripts/run_scanner.py
```

Your Next Steps
Set up accounts on Kalshi and Polymarket (Phase 0 in the plan)
Add API keys to a .env file (copy from .env.example)
Run continuous scanning: python scripts/run_scanner.py --loop to catch fleeting opportunities
Add more search queries in config.yaml to cover more market categories
Download archive data from archive.pmxt.dev to backtest against real historical order books and validate whether profitable opportunities have existed historically after fees