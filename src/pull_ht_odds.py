#!/usr/bin/env python3
"""
Pull historical h2h_h1 (1st half 3-way) odds from The Odds API.

Uses pre-discovered event IDs from historical_h2h_odds.json to avoid
re-listing events. Fetches Pinnacle/Bovada h2h_h1 market for each event.

Usage:
    python src/pull_ht_odds.py [--dry-run] [--limit N] [--sport SPORT_KEY]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Config ──────────────────────────────────────────────────────────────────
PAID_KEY = "39186881a5ca76e74236220a683a025a"
API_HOST = "https://api.the-odds-api.com"
HT_MARKET = "h2h_h1"  # 1st half 3-way on historical endpoint
REGIONS = "us,uk,eu"
ODDS_FORMAT = "decimal"

TARGET_SPORTS = {
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
}

# h2h_h1 market only available from this date onwards on The Odds API
HT_MARKET_CUTOFF = "2023-09-01T00:00:00Z"

SPORT_TO_LEAGUE = {
    "soccer_epl": "E0",
    "soccer_spain_la_liga": "SP1",
    "soccer_italy_serie_a": "I1",
    "soccer_germany_bundesliga": "D1",
    "soccer_france_ligue_one": "F1",
}

DATA_DIR = Path(__file__).parent.parent / "data" / "historical_ht_odds"
H2H_SOURCE = DATA_DIR / "historical_h2h_odds.json"
CHECKPOINT_FILE = DATA_DIR / "ht_pull_checkpoint.json"
OUTPUT_CSV = DATA_DIR / "ht_draw_odds.csv"

CREDIT_SAFETY_FLOOR = 300  # Stop if credits fall to this level
RATE_LIMIT_SLEEP = 0.12    # Seconds between calls (~8 req/s, safe for paid tier)

CSV_FIELDS = [
    "sport", "league", "season", "event_id",
    "home_team", "away_team", "commence_time", "snapshot_ts",
    "ht_home_pinnacle", "ht_draw_pinnacle", "ht_away_pinnacle",
    "ht_home_bovada", "ht_draw_bovada", "ht_away_bovada",
    "ht_home_best", "ht_draw_best", "ht_away_best", "best_book",
    "n_books_with_ht",
]

# ─── API helpers ─────────────────────────────────────────────────────────────

def _get_historical_event_odds(
    sport: str, event_id: str, snapshot_ts: str, dry_run: bool = False
) -> Tuple[Optional[Dict], int, int]:
    """
    Fetch h2h_h1 odds for a specific event at a specific snapshot time.
    Returns (data_dict, credits_used, credits_remaining).
    """
    if dry_run:
        return None, 1, 9999

    url = (
        f"{API_HOST}/v4/historical/sports/{sport}/events/{event_id}/odds"
        f"?apiKey={PAID_KEY}&markets={HT_MARKET}&regions={REGIONS}"
        f"&oddsFormat={ODDS_FORMAT}&date={snapshot_ts}"
    )
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
            used = int(resp.headers.get("x-requests-used", 0))
            remaining = int(resp.headers.get("x-requests-remaining", 0))
            return data, used, remaining
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  HTTP {e.code}: {body[:200]}", file=sys.stderr)
        if e.code == 422:
            return {"_error": "422", "_body": body}, 0, 0
        return None, 0, 0
    except Exception as ex:
        print(f"  Request error: {ex}", file=sys.stderr)
        return None, 0, 0


def parse_ht_odds(event_data: dict, home: str, away: str) -> Dict:
    """Extract h2h_h1 odds per bookmaker from API response."""
    result = {}
    bookmakers = event_data.get("data", event_data).get("bookmakers", [])
    for bm in bookmakers:
        key = bm.get("key", "")
        for mkt in bm.get("markets", []):
            if mkt.get("key") != HT_MARKET:
                continue
            outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
            ht_home = outcomes.get(home) or outcomes.get("Home")
            ht_draw = outcomes.get("Draw")
            ht_away = outcomes.get(away) or outcomes.get("Away")
            if ht_draw is not None:
                result[key] = {
                    "home": ht_home, "draw": ht_draw, "away": ht_away
                }
    return result


def pick_best_snapshot(snapshots: List[str], commence_time: str) -> str:
    """
    From multiple snapshot timestamps for an event, pick the one closest
    to (but before) the commence time. Falls back to latest if all are after.
    """
    try:
        commence_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    except Exception:
        return snapshots[-1]

    # Target: 2 hours before kickoff
    target = commence_dt - timedelta(hours=2)

    # Pick snapshot closest to target, preferring ones before it
    before = [s for s in snapshots if _ts_to_dt(s) <= commence_dt]
    if before:
        return max(before, key=lambda s: abs((_ts_to_dt(s) - target).total_seconds()))
    return snapshots[-1]


def _ts_to_dt(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def infer_season(commence_time: str) -> str:
    """E.g. '2023-08-13T...' → '2023-24'"""
    try:
        dt = _ts_to_dt(commence_time)
        year = dt.year
        month = dt.month
        if month >= 7:
            return f"{year}-{str(year+1)[2:]}"
        else:
            return f"{year-1}-{str(year)[2:]}"
    except Exception:
        return "unknown"


# ─── Main pull logic ──────────────────────────────────────────────────────────

def load_events_from_h2h(sports: Optional[set] = None) -> Dict[str, Dict]:
    """
    Parse historical_h2h_odds.json to extract unique events with their
    best pre-match snapshot timestamp.

    Filters to events from HT_MARKET_CUTOFF onwards (h2h_h1 not available
    for older dates on The Odds API).

    Returns: {event_id: {sport, league, home, away, commence_time, snapshot_ts}}
    """
    print(f"Loading events from {H2H_SOURCE}...")
    raw = json.loads(H2H_SOURCE.read_text())

    cutoff_dt = _ts_to_dt(HT_MARKET_CUTOFF)

    # Group snapshots by event_id
    from collections import defaultdict
    event_snapshots: Dict[str, Dict] = {}
    event_snap_list: Dict[str, List[str]] = defaultdict(list)
    skipped_old = 0

    for rec in raw:
        sport = rec.get("sport", "")
        if sports and sport not in sports:
            continue
        eid = rec.get("event_id", "")
        if not eid:
            continue

        # Filter: only events from HT_MARKET_CUTOFF onwards
        commence = rec.get("commence_time", "")
        if commence and _ts_to_dt(commence) < cutoff_dt:
            skipped_old += 1
            continue

        snap_ts = rec.get("snapshot_ts", "")
        event_snap_list[eid].append(snap_ts)

        # Store metadata (use first encountered; all snapshots have same static data)
        if eid not in event_snapshots:
            event_snapshots[eid] = {
                "sport": sport,
                "league": SPORT_TO_LEAGUE.get(sport, sport),
                "home_team": rec.get("home_team", ""),
                "away_team": rec.get("away_team", ""),
                "commence_time": commence,
            }

    # Pick best snapshot for each event (closest to but before kickoff)
    events = {}
    for eid, meta in event_snapshots.items():
        snaps = sorted(set(event_snap_list[eid]))
        # Filter to pre-match snapshots only
        commence_dt = _ts_to_dt(meta["commence_time"])
        pre_snaps = [s for s in snaps if _ts_to_dt(s) < commence_dt]
        if not pre_snaps:
            pre_snaps = snaps  # fallback
        best_snap = max(pre_snaps, key=lambda s: _ts_to_dt(s))  # latest pre-match
        events[eid] = {**meta, "snapshot_ts": best_snap}

    print(f"Loaded {len(events)} unique events (skipped {skipped_old} pre-cutoff records)")
    return events


def load_checkpoint() -> set:
    """Load set of event_ids already successfully fetched."""
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        return set(data.get("done", []))
    return set()


def save_checkpoint(done: set) -> None:
    CHECKPOINT_FILE.write_text(json.dumps({"done": sorted(done)}, indent=2))


def append_to_csv(rows: List[Dict]) -> None:
    """Append rows to the output CSV, creating header if needed."""
    exists = OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def run_pull(dry_run: bool = False, limit: Optional[int] = None,
             sport_filter: Optional[str] = None, batch_size: int = 50):
    """Main pull loop."""
    target_sports = {sport_filter} if sport_filter else TARGET_SPORTS

    # Load events
    events = load_events_from_h2h(target_sports)

    # Load checkpoint (already done)
    done = load_checkpoint()
    pending = {eid: meta for eid, meta in events.items() if eid not in done}
    print(f"Events: {len(events)} total, {len(done)} done, {len(pending)} pending")

    if not pending:
        print("✅ All events already fetched!")
        return

    if limit:
        items = list(pending.items())[:limit]
        pending = dict(items)
        print(f"Limited to {limit} events")

    # Sort by commence_time (newest first → get most recent data with best coverage)
    sorted_pending = sorted(
        pending.items(), key=lambda kv: kv[1].get("commence_time", ""), reverse=True
    )

    credits_remaining = 9999
    batch_rows = []
    fetched = 0
    skipped_empty = 0
    errors = 0

    print(f"\nStarting pull... {'[DRY RUN]' if dry_run else ''}")
    print(f"Output: {OUTPUT_CSV}")

    for i, (eid, meta) in enumerate(sorted_pending):
        sport = meta["sport"]
        home = meta["home_team"]
        away = meta["away_team"]
        commence = meta["commence_time"]
        snap_ts = meta["snapshot_ts"]

        if not dry_run and credits_remaining < CREDIT_SAFETY_FLOOR:
            print(f"\n⚠️  Credits low ({credits_remaining}). Stopping early.")
            break

        if i > 0 and i % 100 == 0:
            save_checkpoint(done)
            append_to_csv(batch_rows)
            batch_rows = []
            print(f"  [{i}/{len(sorted_pending)}] fetched={fetched} empty={skipped_empty} "
                  f"credits_left={credits_remaining}")

        resp, used, remaining = _get_historical_event_odds(sport, eid, snap_ts, dry_run)

        if not dry_run:
            if remaining:
                credits_remaining = remaining
            time.sleep(RATE_LIMIT_SLEEP)

        if resp is None or (isinstance(resp, dict) and "_error" in resp):
            errors += 1
            done.add(eid)  # Mark done to avoid infinite retries on bad events
            continue

        # Parse HT odds
        ht_books = parse_ht_odds(resp, home, away)

        if not ht_books:
            skipped_empty += 1
            done.add(eid)
            continue

        fetched += 1
        done.add(eid)

        # Best odds = highest draw price
        best_key = max(ht_books, key=lambda k: ht_books[k].get("draw", 0))
        best = ht_books[best_key]

        pinnacle = ht_books.get("pinnacle", {})
        bovada = ht_books.get("bovada", {})

        season = infer_season(commence)
        row = {
            "sport": sport,
            "league": meta["league"],
            "season": season,
            "event_id": eid,
            "home_team": home,
            "away_team": away,
            "commence_time": commence,
            "snapshot_ts": snap_ts,
            "ht_home_pinnacle": pinnacle.get("home"),
            "ht_draw_pinnacle": pinnacle.get("draw"),
            "ht_away_pinnacle": pinnacle.get("away"),
            "ht_home_bovada": bovada.get("home"),
            "ht_draw_bovada": bovada.get("draw"),
            "ht_away_bovada": bovada.get("away"),
            "ht_home_best": best.get("home"),
            "ht_draw_best": best.get("draw"),
            "ht_away_best": best.get("away"),
            "best_book": best_key,
            "n_books_with_ht": len(ht_books),
        }
        batch_rows.append(row)

    # Final flush
    if batch_rows:
        append_to_csv(batch_rows)
    save_checkpoint(done)

    print(f"\n{'='*60}")
    print(f"Pull complete:")
    print(f"  Fetched with HT odds:  {fetched}")
    print(f"  Empty (no HT market):  {skipped_empty}")
    print(f"  Errors:                {errors}")
    print(f"  Credits remaining:     {credits_remaining}")
    print(f"  Output:                {OUTPUT_CSV}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull historical HT draw odds")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate pull without API calls")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max events to process")
    parser.add_argument("--sport", type=str, default=None,
                        help="Filter to one sport key (e.g. soccer_epl)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Save checkpoint every N events")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    run_pull(
        dry_run=args.dry_run,
        limit=args.limit,
        sport_filter=args.sport,
        batch_size=args.batch_size,
    )
