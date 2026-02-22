# Half-Time Draw Predictor

Finds mispriced half-time draw bets in soccer. Uses a two-model architecture that separates market signal from team fundamentals to identify when bookmakers overprice HT draws in the right direction.

## Quick Start

```bash
# Setup
git clone git@github.com:raulduk3/half-time.git && cd half-time
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Today's picks (with context)
python src/picks.py

# Top 5 value picks only
python src/picks.py --min-edge 0.03 --top 5

# Scan all upcoming fixtures
python src/scan_v4.py

# Single match prediction
python src/predict_v4.py 'Everton' 'Man United' --odds 3.70/3.75/1.91

# Track a bet
python src/tracker.py add --home "Everton" --away "Man United" --league E0 --odds 3.75 --stake 5

# Check results
python src/tracker.py stats
```

## How It Works

**Two models, one signal:**

- **Model A (Market)** — LogisticRegression on B365 log-odds. Mirrors what bookmakers think P(HT draw) is. AUC: 0.557.
- **Model B (Fundamentals)** — XGBoost on 42 non-odds features: rolling team stats (shots, corners, fouls, cards), Dixon-Coles Poisson ratings, Elo strength, referee draw tendencies. AUC: 0.535.

**The Inverted Edge** = Model A − Model B. When fundamentals predict fewer draws than the market prices, actual HT draws historically exceed the market prediction. This counterintuitive signal backtests at 5–13% ROI.

## Daily Workflow

```bash
# 1. Update data (downloads new results, refits team models)
python src/update_data.py

# 2. Check picks with context
python src/picks.py --min-edge 0.03

# 3. Place bets on DraftKings/FanDuel for picks you like

# 4. Track what you bet
python src/tracker.py add --home "Getafe" --away "Sevilla" --league SP1 \
    --odds 2.40 --stake 5 --model-a 0.503 --model-b 0.422 --edge 0.0811 \
    --rating "STRONG VALUE"

# 5. Record results after the first half
python src/tracker.py result --id 1 --outcome win

# 6. Check performance
python src/tracker.py stats
```

## Real HT Draw Odds (optional)

The model uses FT draw odds as a proxy by default. For actual half-time draw lines from DraftKings, FanDuel, Betfair, etc:

```bash
# Get a free API key (500 requests/month) at https://the-odds-api.com
export ODDS_API_KEY=your_key_here

# Scanner automatically enriches with real HT odds
python src/scan_v4.py

# Or fetch HT odds directly
python src/odds_api.py all
python src/odds_api.py odds soccer_epl
```

## Data Pipeline

```bash
# Check data freshness
python src/update_data.py --dry-run

# Incremental update (fast, seconds)
python src/update_data.py

# Force re-download all leagues
python src/update_data.py --force-download

# Full rebuild from scratch (~10 min)
python src/update_data.py --full
```

Data source: [football-data.co.uk](https://www.football-data.co.uk/). 203k+ matches across 22 European leagues + MLS, from 1995–2026.

## JSON Output (for automation)

All tools support `--json` for clean machine-readable output:

```bash
python src/picks.py --json --top 3
python src/scan_v4.py --json
python src/predict_v4.py 'Everton' 'Man United' --odds 3.70/3.75/1.91 --json
```

Diagnostics route to stderr, so JSON pipes cleanly.

## Architecture

```
MODEL A (Market)       — LR on 3 B365 log-odds
MODEL B (Fundamentals) — XGBoost on 42 features:
                         39 rolling stats (5-match window)
                         + Dixon-Coles Poisson draw prob
                         + Elo rating draw prob  
                         + Referee draw-rate adjustment

INVERTED EDGE = A − B
  Positive → market prices draw higher than fundamentals
  → actual draws historically exceed market → value bet
```

### Sub-models

- **Dixon-Coles**: Per-league bivariate Poisson on HT scores. Time-decay weighted.
- **Elo**: Rolling team strength with K=16, home advantage=50. P(draw) via logistic regression on Elo differential.
- **Referee Model**: Bayesian per-referee draw-rate adjustment (chi-squared significance test, p < 0.10).

### Kelly Sizing

Backtest-calibrated hit rates per edge bucket:

| Rating | Edge | Hit Rate | Quarter Kelly @ 3.0 odds |
|--------|------|----------|--------------------------|
| ★★★ STRONG VALUE | ≥ 5% | 47.0% | ~5.1% |
| ★★ VALUE | 3–5% | 46.0% | ~4.5% |
| ★ MARGINAL | 1–3% | 44.0% | ~2.0% |

## Backtest

Test period: Apr 2022 – Feb 2026, 29k matches.

| Edge Threshold | Bets | Win% | Flat ROI |
|---------------|------|------|----------|
| > 1% | ~8,700 | ~44.5% | +0.44 |
| > 3% | ~5,500 | ~46.0% | +0.64 |
| > 5% | ~2,350 | ~47.0% | +0.70 |

Bootstrap (1,000 resamples, edge > 3%): **≥99% profitable**, 95% CI [+1.6%, +11.3%].

> ⚠️ Backtest uses FT draw odds as proxy. Real HT draw market returns may differ.

## Project Structure

```
half-time/
├── src/
│   ├── picks.py             ← Daily picks with context (start here)
│   ├── scan_v4.py           ← Fixture scanner ranked by edge
│   ├── predict_v4.py        ← Single match prediction
│   ├── tracker.py           ← Bet tracking + performance
│   ├── update_data.py       ← Data refresh pipeline
│   ├── odds_api.py          ← Real HT odds from The Odds API
│   ├── train_v4.py          ← Training pipeline (rebuild models)
│   ├── dixon_coles.py       ← Dixon-Coles Poisson model
│   ├── elo.py               ← Elo rating system
│   ├── referee_model.py     ← Referee draw-rate profiles
│   ├── features.py          ← Rolling feature engineering
│   ├── build_mega_dataset.py ← Dataset builder
│   ├── utils.py             ← Team name matching
│   └── archive/             ← Old V1/V2/V3 scripts
├── models/v4/               ← Trained models
├── data/
│   ├── processed/           ← Mega dataset (203k matches)
│   └── bets.json            ← Bet tracker
└── README.md
```

## Limitations

1. **FT odds proxy**: Default backtest uses full-time draw odds. Set `ODDS_API_KEY` for real HT lines.
2. **Data lag**: Football-data.co.uk updates 1–2 days after matches. Form data may be stale by matchday.
3. **No tactical info**: Injuries, suspensions, rotation not modeled.
4. **MLS approximated**: No historical MLS HT scores. Dixon-Coles uses FT×0.45 proxy.
5. **European focus**: Best signal in leagues with deep historical data (EPL, La Liga, Serie A, Bundesliga).

## Training

Rebuild all models from scratch (~15–30 min with Optuna tuning):

```bash
python src/train_v4.py
```
