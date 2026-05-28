import json
import os
import sqlite3
import glob
from pathlib import Path

# ── Setup DB ──────────────────────────────────────────────────────────────────
conn = sqlite3.connect("ipl.db")
conn.executescript("""
CREATE TABLE IF NOT EXISTS matches (
    match_id        TEXT PRIMARY KEY,
    date            TEXT,
    season          TEXT,
    venue           TEXT,
    city            TEXT,
    team1           TEXT,
    team2           TEXT,
    toss_winner     TEXT,
    toss_decision   TEXT,
    winner          TEXT,
    win_by_runs     INTEGER,
    win_by_wickets  INTEGER,
    player_of_match TEXT
);

CREATE TABLE IF NOT EXISTS deliveries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id        TEXT,
    inning          INTEGER,
    over_num        INTEGER,
    ball_num        INTEGER,
    batter          TEXT,
    bowler          TEXT,
    non_striker     TEXT,
    runs_batter     INTEGER,
    runs_extras     INTEGER,
    runs_total      INTEGER,
    extras_type     TEXT,
    is_wicket       INTEGER DEFAULT 0,
    player_out      TEXT,
    dismissal_kind  TEXT,
    fielder         TEXT,
    batting_team    TEXT,
    bowling_team    TEXT
);
""")
conn.commit()

# ── Parse one match file ──────────────────────────────────────────────────────
def parse_match(filepath):
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    match_id = Path(filepath).stem
    info     = data["info"]

    # match metadata
    outcome       = info.get("outcome", {})
    winner        = outcome.get("winner")
    win_by        = outcome.get("by", {})
    win_by_runs   = win_by.get("runs")
    win_by_wkts   = win_by.get("wickets")
    pom           = info.get("player_of_match", [])

    match_row = {
        "match_id":        match_id,
        "date":            info.get("dates", [None])[0],
        "season":          str(info.get("season", "")),
        "venue":           info.get("venue"),
        "city":            info.get("city"),
        "team1":           info["teams"][0] if len(info.get("teams",[])) > 0 else None,
        "team2":           info["teams"][1] if len(info.get("teams",[])) > 1 else None,
        "toss_winner":     info.get("toss", {}).get("winner"),
        "toss_decision":   info.get("toss", {}).get("decision"),
        "winner":          winner,
        "win_by_runs":     win_by_runs,
        "win_by_wickets":  win_by_wkts,
        "player_of_match": pom[0] if pom else None,
    }

    # deliveries
    delivery_rows = []
    teams = info.get("teams", [])

    for inning_idx, inning in enumerate(data.get("innings", []), start=1):
        batting_team = inning.get("team")
        bowling_team = next((t for t in teams if t != batting_team), None)

        for over in inning.get("overs", []):
            over_num = over["over"]
            for ball_idx, delivery in enumerate(over.get("deliveries", []), start=1):
                runs    = delivery.get("runs", {})
                extras  = delivery.get("extras", {})
                wickets = delivery.get("wickets", [])

                extras_type = list(extras.keys())[0] if extras else None

                is_wicket      = 1 if wickets else 0
                player_out     = wickets[0].get("player_out")     if wickets else None
                dismissal_kind = wickets[0].get("kind")           if wickets else None
                fielders       = wickets[0].get("fielders", [])   if wickets else []
                fielder        = fielders[0].get("name") if fielders else None

                delivery_rows.append({
                    "match_id":       match_id,
                    "inning":         inning_idx,
                    "over_num":       over_num,
                    "ball_num":       ball_idx,
                    "batter":         delivery.get("batter"),
                    "bowler":         delivery.get("bowler"),
                    "non_striker":    delivery.get("non_striker"),
                    "runs_batter":    runs.get("batter", 0),
                    "runs_extras":    runs.get("extras", 0),
                    "runs_total":     runs.get("total", 0),
                    "extras_type":    extras_type,
                    "is_wicket":      is_wicket,
                    "player_out":     player_out,
                    "dismissal_kind": dismissal_kind,
                    "fielder":        fielder,
                    "batting_team":   batting_team,
                    "bowling_team":   bowling_team,
                })

    return match_row, delivery_rows

# ── Parse all files ───────────────────────────────────────────────────────────
# Change this path to wherever you extracted the Cricsheet ZIP
JSON_FOLDER = r"C:\Users\HP\ipl project\ipl_male_json"

files = glob.glob(os.path.join(JSON_FOLDER, "*.json"))
print(f"Found {len(files)} match files")

match_count    = 0
delivery_count = 0
errors         = 0

for i, filepath in enumerate(files):
    try:
        match_row, delivery_rows = parse_match(filepath)

        conn.execute("""
            INSERT OR IGNORE INTO matches
            (match_id, date, season, venue, city, team1, team2,
             toss_winner, toss_decision, winner, win_by_runs,
             win_by_wickets, player_of_match)
            VALUES
            (:match_id, :date, :season, :venue, :city, :team1, :team2,
             :toss_winner, :toss_decision, :winner, :win_by_runs,
             :win_by_wickets, :player_of_match)
        """, match_row)

        conn.executemany("""
            INSERT INTO deliveries
            (match_id, inning, over_num, ball_num, batter, bowler,
             non_striker, runs_batter, runs_extras, runs_total,
             extras_type, is_wicket, player_out, dismissal_kind,
             fielder, batting_team, bowling_team)
            VALUES
            (:match_id, :inning, :over_num, :ball_num, :batter, :bowler,
             :non_striker, :runs_batter, :runs_extras, :runs_total,
             :extras_type, :is_wicket, :player_out, :dismissal_kind,
             :fielder, :batting_team, :bowling_team)
        """, delivery_rows)

        conn.commit()
        match_count    += 1
        delivery_count += len(delivery_rows)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(files)} files — {delivery_count:,} deliveries so far")

    except Exception as e:
        errors += 1
        print(f"  Error in {filepath}: {e}")

# ── Create indexes for fast queries ──────────────────────────────────────────
print("\nCreating indexes...")
conn.executescript("""
    CREATE INDEX IF NOT EXISTS idx_deliveries_match   ON deliveries(match_id);
    CREATE INDEX IF NOT EXISTS idx_deliveries_batter  ON deliveries(batter);
    CREATE INDEX IF NOT EXISTS idx_deliveries_bowler  ON deliveries(bowler);
    CREATE INDEX IF NOT EXISTS idx_deliveries_wicket  ON deliveries(is_wicket);
    CREATE INDEX IF NOT EXISTS idx_matches_season     ON matches(season);
    CREATE INDEX IF NOT EXISTS idx_matches_winner     ON matches(winner);
""")
conn.commit()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nDone.")
print(f"  Matches loaded:    {match_count:,}")
print(f"  Deliveries loaded: {delivery_count:,}")
print(f"  Errors:            {errors}")

print("\nSample — first 5 deliveries of match 1527684:")
import pandas as pd
df = pd.read_sql("""
    SELECT over_num, ball_num, batter, bowler,
           runs_total, is_wicket, dismissal_kind
    FROM deliveries
    WHERE match_id = '1527684'
      AND inning = 1
    ORDER BY over_num, ball_num
    LIMIT 10
""", conn)
print(df.to_string(index=False))

print("\nDeliveries per season:")
df2 = pd.read_sql("""
    SELECT m.season, COUNT(d.id) as deliveries, COUNT(DISTINCT d.match_id) as matches
    FROM deliveries d JOIN matches m ON d.match_id = m.match_id
    GROUP BY m.season ORDER BY m.season
""", conn)
print(df2.to_string(index=False))

conn.close()
