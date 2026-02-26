# V5 Roadmap — Half-Time Draw Prediction System

**Status:** V4 shipping, V5 planned  
**Goal:** Break through 0.56 AUC ceiling, eliminate FT-odds proxy, full automation  
**Timeline:** 4 phases, ~3 weeks of focused work

---

## Current State (V4)

| Metric | Value |
|---|---|
| Model A (market) test AUC | 0.558 |
| Model B (fundamentals) test AUC | 0.539 |
| Inverted edge STRONG VALUE hit rate | 48.0% (vs 42.3% base) |
| Dataset | 204K matches, 24 leagues, 1995-2026 |
| Features | 7 (Model A) + 42 (Model B) |
| Real HT odds in pipeline | 0 (FT proxy only) |
| xG data | None |
| Lineup data | None |
| Daily automation | None |
| Prospective validation sample | ~60 predictions |

---

## Phase 1: Fix the Foundation (IN PROGRESS)

**Timeline:** Now — Feb 25  
**Owner:** Coding agent (wild-ocean) + HT odds sub-agent

### 1A. Model integrity fixes (coding agent running)
- [x] DC/Elo feature leakage in Model B training
- [x] Elo K-factor tuning (replay_and_predict, not look-ahead)
- [x] Dixon-Coles xi tuning
- [x] DC convergence checks (maxiter 500→2000)
- [x] Unknown team fallback (league-specific, not global 0.42)
- [x] Referee prior strength decoupled from min_matches

### 1B. Historical HT odds (sub-agent running)
- [ ] Pull all available h2h_h1 odds from The Odds API (2023-2026, 5 leagues)
- [ ] Merge with mega dataset
- [ ] Target: 1,000+ real Pinnacle HT draw odds
- API key: new 20K credits active

### 1C. Retrain V4.1 with fixes
- [ ] Run train_v4.py with leakage-free features
- [ ] Compare honest AUC numbers to current (expect Model B train AUC to drop)
- [ ] Verify inverted edge staircase still holds
- [ ] Update white paper with honest numbers

**Exit criteria:** Leakage-free training, 1000+ real HT odds on disk, honest metrics.

---

## Phase 2: New Data Sources

**Timeline:** Feb 25 — Mar 3  
**Goal:** Break the historical-stats ceiling with xG, first-half splits, and lineup data

### 2A. FBref xG scraper
Source: fbref.com (free, no API key, rate-limit 3 req/s)

**Data available:**
- Per-match xG, xGA for every team (top 5 leagues + Championship, 2017-2026)
- First-half / second-half xG splits (the killer feature)
- Per-match lineups with minutes played
- Shooting stats: npxG, shots, SoT, shot distance, FK shots
- Passing: progressive passes, key passes, final third entries
- Defensive: pressures, tackles, interceptions, blocks

**New features to engineer:**
```
# xG-based (rolling 5-match)
home_xg_r5              # team's avg xG per match
home_xga_r5             # team's avg xG against
home_xg_diff_r5         # xG surplus (attacking quality)
away_xg_r5, away_xga_r5, away_xg_diff_r5

# First-half specific (THE breakthrough)
home_1h_xg_r5           # first-half xG (directly predictive of HT outcomes)
home_1h_xga_r5          # first-half xG against
away_1h_xg_r5, away_1h_xga_r5

# xG overperformance (luck/finishing quality)
home_goals_minus_xg_r5  # positive = overperforming (due for regression)
away_goals_minus_xg_r5

# Shot quality
home_xg_per_shot_r5     # shot quality proxy
away_xg_per_shot_r5
```

**Implementation:**
1. Write `src/fbref_scraper.py` — scrape match logs for top 5 leagues + Championship
2. Cache as `data/xg/fbref_{league}_{season}.csv`
3. Merge into mega dataset by (HomeTeam, AwayTeam, Date) fuzzy match
4. Add rolling xG features to `build_mega_dataset.py`

**Expected impact:** xG features alone should push Model B AUC from 0.539 → 0.55+. First-half xG splits are the most directly relevant feature we don't have.

### 2B. Lineup and rotation detection
Source: FBref match reports (same scrape as 2A)

**New features:**
```
home_avg_minutes_r5     # avg minutes played by starters (rotation proxy)
home_lineup_changes     # number of changes from previous match
away_avg_minutes_r5, away_lineup_changes

# Key player availability (derived)
home_top_scorer_available   # binary: is team's top scorer in lineup?
away_top_scorer_available
```

**Implementation:**
1. Extend fbref_scraper to pull lineup tables
2. Build player importance model: minutes played × goals/assists contribution
3. Flag rotation matches (>3 changes = likely resting for UCL/cup)

### 2C. Weather data
Source: Open-Meteo historical API (free, no key)

**New features:**
```
temperature_c           # match-time temperature at stadium
precipitation_mm        # rainfall in prior 3 hours
wind_speed_kmh          # surface wind speed
is_extreme_weather      # binary: rain>5mm OR wind>40kmh OR temp<0°C
```

**Implementation:**
1. Build stadium coordinate lookup (one-time, ~400 stadiums)
2. Write `src/weather.py` — query Open-Meteo for historical weather at (lat, lon, datetime)
3. Cache results, merge into dataset
4. Rain + cold → lower scoring → more 0-0 draws → higher HT draw rate

### 2D. Motivation/context features
Source: Derive from existing data (league tables, fixtures)

**New features:**
```
home_league_position        # current league standing
away_league_position
home_points_to_safety       # relegation proximity (fear → defensive)
away_points_to_safety
home_points_to_title        # title race (pressure → cautious starts)
is_derby                    # local rivalry flag (lookup table)
season_phase                # early/mid/late (months 8-10/11-1/2-5)
home_matches_played         # fixture congestion (>2 in 7 days)
away_matches_played
```

**Implementation:**
1. Compute running league tables from existing results data
2. Calculate points gaps (safety, title, European spots)
3. Derby lookup table (top 5 leagues, ~50 derbies)
4. Congestion: count matches in trailing 7/14 days (already have dates)

---

## Phase 3: Model Architecture V5

**Timeline:** Mar 3 — Mar 10  
**Goal:** Rebuild both models with new data, add ensemble diversity

### 3A. Model A V5 — Real HT odds
Replace FT-odds proxy with actual HT draw odds where available.

**Features:**
```
# Primary (when HT odds available)
log_ht_draw_odds_pinnacle   # sharpest book
log_ht_home_odds_pinnacle
log_ht_away_odds_pinnacle
ht_odds_spread              # cross-book disagreement on HT market

# Fallback (when only FT odds available)
log_ft_draw_odds            # current B365 features
ft_to_ht_draw_ratio         # learned conversion factor
```

**Architecture:** Same LogisticRegression but with HT-native features. Train on matches with real HT odds, apply learned FT→HT conversion for matches without.

**Expected impact:** Model A AUC 0.558 → 0.57+ on matches with real HT odds. Calibration dramatically improves (no more 4x ROI inflation).

### 3B. Model B V5 — xG + context features
Add Tier 2 features to the fundamentals model.

**Feature count:** 42 current + ~20 new = ~62 features  
Run Optuna feature selection (drop features with zero importance after 50 trials).

**Architecture options (evaluate all three):**
1. XGBoost (current) with expanded features
2. CatBoost (handles categoricals natively, better with sparse features)
3. TabNet (attention-based, learns feature interactions)

Pick best by validation AUC. If within 0.002, keep XGBoost (simplicity).

### 3C. Model C — Sequence model (new)
A genuinely different model that sees team history as a time series.

**Architecture:** 
- Input: last 10 matches for each team as a sequence of feature vectors
- Model: 1D-CNN or Transformer encoder (small, 2-layer)
- Output: P(HT draw)
- This captures form trajectories that rolling averages miss (e.g., "improving" vs "declining" teams with the same 5-match average)

**Implementation:**
1. Build sequence dataset: for each match, extract (home_team_last_10, away_team_last_10)
2. Each match in the sequence = [xG, xGA, goals, shots, possession, result]
3. Train PyTorch model with temporal attention
4. Add as third signal to ensemble

### 3D. Ensemble V5
Three independent probability estimates:
- Model A: market signal (odds-based)
- Model B: fundamentals (XGBoost on stats + xG + context)
- Model C: form trajectory (sequence model)

**Edge signal:** Generalize beyond simple A-B.
```python
# V4: edge = model_a - model_b (one-dimensional)
# V5: edge = f(model_a, model_b, model_c, |a-b|, |a-c|, |b-c|, max-min)
# Train a small stacking model on these 7 meta-features
```

**Expected impact:** Model B 0.539 → 0.56+, ensemble edge signal sharper, fewer false positives in STRONG VALUE tier.

---

## Phase 4: Automation and Operations

**Timeline:** Mar 10 — Mar 15  
**Goal:** Fully automated daily pipeline, live tracking, no manual steps

### 4A. Daily pipeline cron
```
08:00 CST — update_data.py (pull latest results from football-data.co.uk)
08:05 CST — fbref_update.py (pull yesterday's xG + lineups)
08:10 CST — odds_fetch.py (pull today's HT odds from The Odds API)
08:15 CST — scan_v5.py (generate predictions for all today's fixtures)
08:20 CST — daily_log.py predict (log predictions with timestamps)
08:25 CST — Discord notification (VALUE+ picks only)
22:00 CST — daily_log.py score (score today's predictions against results)
22:05 CST — tracker update (auto-record bet outcomes)
```

Single OpenClaw cron job, isolated session, runs the chain.

### 4B. Expanded fixture coverage
- Add UCL, UEL, Conference League via The Odds API sport keys
- Add Copa del Rey, FA Cup, DFB-Pokal (domestic cups)
- The Odds API has 48 soccer sport keys. Map all relevant ones.

### 4C. Bet tracker V2
```
data/bets.json → data/bets.db (SQLite)

Fields:
- match_id, date, home, away, league
- model_version, edge_pct, rating, model_a_prob, model_b_prob
- odds_placed, stake, book (DraftKings)
- result (W/L/void), pnl
- ht_score, ft_score
```

Auto-populate results from daily_log scoring. Running PnL chart (matplotlib, auto-generated). Drawdown alerts (>15% from peak → Discord warning).

### 4D. Model versioning
```
models/v5/
  v5_20260310/  ← training run with timestamp
    model_a.pkl
    model_b.json
    model_c.pt
    metrics.json
    predictions/  ← all predictions made with this version
```

Tag each prediction with model version. Compare prospective ROI across versions.

### 4E. Vectorized dataset builder
Rewrite build_mega_dataset.py using pandas groupby + shift for rolling features. Target: <30 seconds for full 200K+ rebuild (currently 10+ minutes).

```python
# Current (O(n²) — iterates every row)
for idx, row in df.iterrows():
    history = df[(df['Date'] < row['Date']) & (df['HomeTeam'] == row['HomeTeam'])].tail(5)

# V5 (O(n log n) — vectorized)
df.groupby('team').rolling(5, on='Date')['goals'].mean()
```

---

## Phase 5: Research and Publication

**Timeline:** Mar 15 onward (ongoing)  
**Goal:** 100+ tracked bets, publishable paper, grad school portfolio piece

### 5A. Prospective validation
- Target: 500 tracked predictions by May 2026
- 100 placed bets with real money outcomes
- Weekly performance report (auto-generated)

### 5B. White paper V2
- Real HT odds throughout (no FT proxy caveats)
- xG features documented
- Three-model ensemble architecture
- Prospective results (not just backtest)
- Title: "Market-Fundamental Divergence in Half-Time Soccer Draw Markets"
- Target journals: Journal of Sports Analytics, Journal of Quantitative Analysis in Sports

### 5C. Open source
- Clean README with reproducible pipeline
- GitHub Actions CI (lint, test, train smoke test)
- Portfolio piece for grad school apps (Northwestern, UChicago, Columbia, NYU)

---

## Success Metrics

| Metric | V4 (current) | V5 target | Stretch |
|---|---|---|---|
| Model A test AUC | 0.558 | 0.575 | 0.59 |
| Model B test AUC | 0.539 | 0.560 | 0.58 |
| STRONG VALUE hit rate | 48% | 50% | 52% |
| Real HT odds in dataset | 141 | 2,000+ | 5,000+ |
| xG coverage (top 5) | 0% | 90%+ | 95%+ |
| Daily automation | manual | full cron | alerting |
| Prospective bets tracked | 3 | 100+ | 500+ |
| Prospective ROI | unknown | +5% | +15% |

---

## Data Sources Summary

| Source | Data | Cost | Coverage | Priority |
|---|---|---|---|---|
| The Odds API | HT odds, FT odds, live odds | $30/mo (20K credits) | 2023-2026, 6+ leagues | ★★★ CRITICAL |
| FBref | xG, lineups, advanced stats | Free (rate limited) | 2017-2026, top 5 + Champ | ★★★ CRITICAL |
| football-data.co.uk | Match results, basic stats, B365 odds | Free | 1995-2026, 22 leagues | ★★★ (already have) |
| Open-Meteo | Historical weather | Free | Global, hourly | ★★ NICE |
| Transfermarkt | Market values, injuries | Free (scrape) | Current season | ★ LATER |
| Understat | xG (alternative to FBref) | Free | 2014-2026, top 5 | ★ BACKUP |

---

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| xG features don't improve Model B | Wasted scraping effort | Test on 1 league first before full scrape |
| The Odds API HT coverage too sparse | Can't build HT-native Model A | Use FT→HT learned conversion as fallback |
| Overfitting with 62 features | Worse out-of-sample | Aggressive regularization + Optuna pruning |
| FBref blocks scraping | No xG data | Switch to Understat API or StatsBomb open data |
| Model C (sequence) underperforms | Wasted complexity | Only add to ensemble if val AUC > 0.53 |
| DraftKings limits HT markets | Can't place bets | Expand to Bovada, BetMGM, FanDuel |
| API credits exhausted mid-pull | Incomplete HT odds | Budget pulls carefully, checkpoint everything |
