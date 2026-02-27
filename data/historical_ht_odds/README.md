# Historical Half-Time Draw Odds Dataset

## Summary

This directory contains real half-time (1st half 3-way) draw odds pulled from The Odds API and related feature data.

## Files

| File | Description |
|------|-------------|
| `ht_draw_odds.csv` | **PRIMARY** — Real HT draw odds from Pinnacle (141 events) |
| `historical_h2h_odds.json` | Full-time (H2H) 3-way odds for 19,954 events (2022–2026) |
| `multi_book_features.parquet` | FT draw odds features (multi-bookmaker aggregates, used as proxy) |
| `pull_checkpoint.json` | H2H pull checkpoint (1,150 sport-date combos done) |
| `ht_pull_checkpoint.json` | HT pull checkpoint (227 events processed) |
| `pull_ht_odds.py` → `../../src/` | Script to pull more HT odds when credits available |

---

## ht_draw_odds.csv

### Coverage
- **Seasons**: 2023-24 only
- **Leagues**: EPL (E0), La Liga (SP1), Serie A (I1), Bundesliga (D1), Ligue 1 (F1)
- **Records**: 141 matches
- **Bookmaker**: Primarily Pinnacle (best sharp line for HT markets)

### Schema
```
sport         - API sport key (e.g. soccer_epl)
league        - Football-data league code (E0, SP1, I1, D1, F1)
season        - Season string (e.g. 2023-24)
event_id      - The Odds API event UUID
home_team     - Home team name (as in Odds API)
away_team     - Away team name
commence_time - Match kickoff (UTC ISO 8601)
snapshot_ts   - Pre-match snapshot timestamp used to fetch odds
ht_home_pinnacle    - Pinnacle H1 home win odds (decimal)
ht_draw_pinnacle    - Pinnacle H1 draw odds (decimal) ← PRIMARY FEATURE
ht_away_pinnacle    - Pinnacle H1 away win odds (decimal)
ht_home_bovada      - Bovada H1 home odds (if available)
ht_draw_bovada      - Bovada H1 draw odds (if available)
ht_away_bovada      - Bovada H1 away odds (if available)
ht_home_best        - Best available H1 home odds
ht_draw_best        - Best available H1 draw odds (highest price)
ht_away_best        - Best available H1 away odds
best_book           - Bookmaker with highest HT draw odds
n_books_with_ht     - Number of bookmakers with H1 data
```

### Odds Distribution (Pinnacle HT Draw)
- Min: 2.00, Max: 4.15, Avg: 2.46
- Range makes sense: HT draws have ~40% base probability; 2.46 ≈ 1/0.40 with margin

---

## API Investigation Notes

### Market: h2h_h1 (NOT h2h_3_way_h1)
- `h2h_3_way_h1` is listed in API docs but returns "INVALID_MARKET" on historical endpoint
- `h2h_h1` **does** work on the historical per-event endpoint
- Returns full 3-way (Home/Draw/Away) outcomes despite the "h2h" name
- Pinnacle started offering historical h2h_h1 snapshots around **August–September 2023**
- Before that date: API returns `HISTORICAL_MARKETS_UNAVAILABLE_AT_DATE`

### Credit Usage
- Plan: 20,000 credits (paid key: `3490018fc5a33ab64916f395675e3b99`)
- Used: ~19,800 credits (investigation + pulls)
- Remaining: ~177 credits (effectively exhausted)
- Each historical event odds call: 1 credit

### Available Events Discovered
- 4,816 unique events from target leagues, 2023-09 onwards
- 7,311 total events from target leagues (all dates in h2h dataset)
- Only ~141/4,816 had h2h_h1 data available (3% hit rate due to Pinnacle coverage gaps)

---

## Extending the Dataset

### Option 1: Buy More API Credits
The Odds API paid plan allows more credits. With 4,816 pending events and ~3% hit rate, need ~5,000 credits minimum to get ~150 more events.

### Option 2: OddsPortal Scraping
OddsPortal has historical HT odds going back to 2010+. Requires browser automation (Playwright/Selenium). The `browser` tool in OpenClaw can be used when available.

Target URLs:
```
https://www.oddsportal.com/football/england/premier-league/results/
# → Navigate to match → odds tab → "1st Half" market
```

### Option 3: Use FT Draw Odds as Proxy
The existing model uses `multi_book_features.parquet` (FT draw odds). The correlation between FT and HT draw probability is strong (both reflect team balance). A calibration model trained on the 141 real HT observations can convert FT odds to HT odds.

### Merge with Match Data
To join `ht_draw_odds.csv` with `data/raw_all/` match data:
```python
import pandas as pd
ht = pd.read_csv('data/historical_ht_odds/ht_draw_odds.csv')
ht['date'] = pd.to_datetime(ht['commence_time']).dt.date

# Fuzzy join on home_team + away_team + date with football-data team names
# Team name normalization needed (API uses full names, FD uses abbreviations)
```

---

## Pull Script

```bash
# Pull more HT odds (requires paid API credits)
cd ~/Dev/half-time
python src/pull_ht_odds.py [--limit N] [--sport soccer_epl]

# The script:
# - Loads events from historical_h2h_odds.json (pre-discovered event IDs)
# - Filters to events from 2023-09-01 onwards (h2h_h1 availability cutoff)
# - Fetches h2h_h1 market from Pinnacle/Bovada
# - Saves incrementally to ht_draw_odds.csv
# - Checkpoints to ht_pull_checkpoint.json
```
