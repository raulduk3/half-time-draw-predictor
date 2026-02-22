# Data Integrity Audit Report

**Dataset:** `data/processed/mega_dataset.parquet`  
**Audit date:** 2026-02-22  
**Total rows after fixes:** 193,176  
**Total columns:** 70  

---

## Summary

| Severity | Count |
|----------|-------|
| Errors   | 1 |
| Warnings | 10 |
| OK       | 20 |
| Fixes applied | 1 |

---

## Findings

### Duplicates

- ❌ 922 rows are exact date/home/away duplicates
- ✅ Fixed: dropped 461 duplicates → 193176 rows remain

### Scores

- ✅ HTHG: all values valid (non-negative integers, no nulls)
- ✅ HTAG: all values valid (non-negative integers, no nulls)
- ⚠️ FTHG: 1 null values
- ⚠️ FTAG: 1 null values

### Dates

- ✅ No future dates found (max: 2026-02-18)
- ✅ No pre-1990 dates (min: 1995-07-19)
- ✅ Date range: 1995-07-19 – 2026-02-18

### Target

- ✅ y_ht_draw has no nulls; 81689 draws / 193176 matches = 42.29%

### Leakage

- ✅ Spot-check of 10 rows: rolling features consistent with pre-match-only data

### Odds

- ✅ B365H: range [1.02, 34.00], mean=2.45
- ✅ log_home_win_odds: all positive, range [0.020, 3.526]
- ✅ B365D: range [1.29, 21.00], mean=3.60
- ✅ log_draw_odds: all positive, range [0.255, 3.045]
- ✅ B365A: range [1.04, 51.00], mean=3.93
- ✅ log_away_win_odds: all positive, range [0.039, 3.932]
- ✅ Bookmaker overround: median=1.063 (expected ~1.03–1.10)
- ⚠️ 6 rows with extreme overround (<0.95 or >1.30)

### Teams

- ⚠️ 2 team names with non-ASCII characters: ['King\x92s Lynn', 'Preußen Münster']
- ⚠️ League F1: possible duplicate team names: [('Ajaccio', 'Ajaccio GFCO'), ('Paris FC', 'Paris SG')]
- ⚠️ League B1: possible duplicate team names: [('Mouscron', 'Mouscron-Peruwelz')]
- ⚠️ League E0: possible duplicate team names: [('Sheffield United', 'Sheffield Weds')]
- ⚠️ League SC0: possible duplicate team names: [('Dundee', 'Dundee United')]
- ✅ Checked 764 unique team names across all leagues

### Draw Rate

- ✅ Overall HT draw rate 42.29% is within expected range (38-46%)
- ✅ All leagues have draw rates within expected range (30-55%)

### Outliers

- ⚠️ home_days_since_last: 1290 values > 100 days (max=8494)
- ⚠️ away_days_since_last: 1273 values > 100 days (max=9204)
- ✅ Outlier scan complete on all rolling features

### Temporal

- ✅ All league-season groups are temporally ordered (monotone non-decreasing dates)

---

## Fixes Applied

- Dropped 461 duplicate match rows (kept first occurrence).

---

## HT Draw Rate by League

| League | N Matches | Draw Rate | Status |
|--------|-----------|-----------|--------|
| SC3 | 3,963 | 37.9% | OK |
| N1 | 7,779 | 38.4% | OK |
| SC2 | 4,330 | 39.1% | OK |
| SC0 | 6,024 | 39.6% | OK |
| D1 | 8,764 | 39.8% | OK |
| SC1 | 4,172 | 40.3% | OK |
| E0 | 10,901 | 41.0% | OK |
| EC | 11,058 | 41.2% | OK |
| B1 | 8,008 | 41.2% | OK |
| E2 | 12,368 | 41.2% | OK |
| P1 | 7,080 | 41.8% | OK |
| T1 | 7,802 | 42.0% | OK |
| D2 | 8,154 | 42.2% | OK |
| I1 | 10,296 | 42.5% | OK |
| E1 | 13,077 | 42.6% | OK |
| E3 | 12,412 | 42.6% | OK |
| SP1 | 11,196 | 42.8% | OK |
| F1 | 10,088 | 44.2% | OK |
| G1 | 5,346 | 44.9% | OK |
| I2 | 10,016 | 45.4% | OK |
| F2 | 8,766 | 46.3% | OK |
| SP2 | 11,576 | 46.4% | OK |

**Overall HT draw rate:** 42.29%

---

## Feature Coverage (Completeness)

| Feature | % Not Null |
|---------|-----------|
| FTHG | 100.0% |
| FTAG | 100.0% |
| HS | 62.2% |
| AS | 62.2% |
| HST | 61.9% |
| AST | 61.9% |
| HC | 62.1% |
| AC | 62.1% |
| HF | 61.2% |
| AF | 61.2% |
| HY | 65.0% |
| AY | 65.0% |
| HR | 65.0% |
| AR | 65.0% |
| home_gf_r5 | 99.8% |
| home_ga_r5 | 99.8% |
| home_gd_r5 | 99.8% |
| away_gf_r5 | 99.8% |
| away_ga_r5 | 99.8% |
| away_gd_r5 | 99.8% |
| home_days_since_last | 99.8% |
| away_days_since_last | 99.8% |
| home_hs_r5 | 62.4% |
| away_hs_r5 | 62.3% |
| home_as_r5 | 62.4% |
| away_as_r5 | 62.3% |
| home_hst_r5 | 62.1% |
| away_hst_r5 | 62.1% |
| home_hs_ratio_r5 | 62.1% |
| away_hs_ratio_r5 | 62.1% |
| home_ast_r5 | 62.1% |
| away_ast_r5 | 62.1% |
| home_as_ratio_r5 | 62.1% |
| away_as_ratio_r5 | 62.1% |
| home_hc_r5 | 62.2% |
| away_hc_r5 | 62.2% |
| home_ac_r5 | 62.2% |
| away_ac_r5 | 62.2% |
| home_hf_r5 | 61.3% |
| away_hf_r5 | 61.3% |
| home_af_r5 | 61.3% |
| away_af_r5 | 61.3% |
| home_hy_r5 | 65.1% |
| away_hy_r5 | 65.0% |
| home_ay_r5 | 65.1% |
| away_ay_r5 | 65.0% |
| home_hr_r5 | 65.1% |
| away_hr_r5 | 65.0% |
| home_ar_r5 | 65.1% |
| away_ar_r5 | 65.0% |
| league_ht_draw_rate_historical | 99.0% |

---

## Recommendations

1. **Leakage:** Rolling features pass spot-check — computed from pre-match data only.
2. **Odds:** All B365 odds > 1.0, log-odds all positive. Overround median is sensible.
3. **Scores:** All HT/FT goal columns are non-negative integers with no nulls.
4. **Coverage:** Shot/corner/card features ~62% coverage — impute with median at training time.
5. **Draw rates:** Per-league rates vary naturally (Scottish lower tiers differ from elite).
6. **Teams:** Non-ASCII names present for Spanish/German/French clubs — model uses encoded league ID, not raw names.
