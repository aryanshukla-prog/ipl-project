import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import pickle

conn = sqlite3.connect("ipl.db")

# ── Load all deliveries with match context ────────────────────────────────────
print("Loading deliveries...")
df = pd.read_sql("""
    SELECT
        d.*,
        m.winner,
        m.team1,
        m.team2
    FROM deliveries d
    JOIN matches m ON d.match_id = m.match_id
    WHERE m.winner IS NOT NULL
""", conn)
print(f"Loaded {len(df):,} deliveries")

# ── Build cumulative match state at each ball ─────────────────────────────────
print("Computing match state features...")

df = df.sort_values(["match_id", "inning", "over_num", "ball_num"])

# cumulative runs and wickets per inning per match
df["cum_runs"]    = df.groupby(["match_id","inning"])["runs_total"].cumsum()
df["cum_wickets"] = df.groupby(["match_id","inning"])["is_wicket"].cumsum()

# balls bowled (legal only — exclude wides/noballs for ball count)
df["is_legal"]    = (df["extras_type"].isna() | 
                     ~df["extras_type"].isin(["wides","noballs"])).astype(int)
df["cum_balls"]   = df.groupby(["match_id","inning"])["is_legal"].cumsum()
df["cum_overs"]   = df["cum_balls"] / 6

# ── Second innings: build features for win probability ───────────────────────
# For win probability we focus on 2nd innings — chasing team's perspective
inn2 = df[df["inning"] == 2].copy()

# get target from inning 1
inn1_totals = (df[df["inning"] == 1]
               .groupby("match_id")["runs_total"]
               .sum()
               .reset_index()
               .rename(columns={"runs_total": "target"}))
inn1_totals["target"] += 1  # need 1 more than inn1 total

inn2 = inn2.merge(inn1_totals, on="match_id")

# runs needed, balls remaining
inn2["runs_needed"]      = inn2["target"] - inn2["cum_runs"]
inn2["balls_remaining"]  = 120 - inn2["cum_balls"]
inn2["wickets_remaining"] = 10 - inn2["cum_wickets"]
inn2["required_rr"]      = np.where(
    inn2["balls_remaining"] > 0,
    inn2["runs_needed"] / (inn2["balls_remaining"] / 6),
    999
)
inn2["current_rr"] = np.where(
    inn2["cum_overs"] > 0,
    inn2["cum_runs"] / inn2["cum_overs"],
    0
)
inn2["rr_diff"] = inn2["current_rr"] - inn2["required_rr"]

# label: did the batting team (chasing) win?
inn2["batting_won"] = (inn2["batting_team"] == inn2["winner"]).astype(int)

# ── Features and target ───────────────────────────────────────────────────────
features = [
    "cum_runs",
    "runs_needed",
    "balls_remaining",
    "wickets_remaining",
    "required_rr",
    "current_rr",
    "rr_diff",
    "cum_overs",
]

inn2_clean = inn2.dropna(subset=features + ["batting_won"])
inn2_clean = inn2_clean[inn2_clean["balls_remaining"] >= 0]
inn2_clean = inn2_clean[inn2_clean["runs_needed"] >= 0]

X = inn2_clean[features]
y = inn2_clean["batting_won"]

print(f"Training on {len(X):,} delivery states")
print(f"Chase win rate: {y.mean():.1%}")

# ── Train model ───────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Gradient Boosting model...")
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred      = model.predict(X_test)
y_prob      = model.predict_proba(X_test)[:, 1]
acc         = accuracy_score(y_test, y_pred)
ll          = log_loss(y_test, y_prob)

print(f"\nModel performance:")
print(f"  Accuracy:  {acc:.3f}")
print(f"  Log loss:  {ll:.3f}")

# feature importance
print("\nFeature importance:")
for feat, imp in sorted(zip(features, model.feature_importances_),
                        key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {feat:<20} {bar} {imp:.3f}")

# ── Save model ────────────────────────────────────────────────────────────────
with open("win_prob_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel saved to win_prob_model.pkl")

# ── ANALYSIS 1: Wicket Probability Swing Index ────────────────────────────────
print("\n" + "="*60)
print("ANALYSIS 1: Wicket Probability Swing Index")
print("="*60)
print("How much does each dismissal type drop win probability?")

# compute win prob before and after each wicket
wicket_rows = inn2_clean[inn2_clean["is_wicket"] == 1].copy()

# win prob at moment of wicket
wicket_rows["wp_at_wicket"] = model.predict_proba(wicket_rows[features])[:, 1]

# win prob one ball later (after wicket absorbed)
# approximate: add 1 wicket, same everything else
after = wicket_rows[features].copy()
after["wickets_remaining"] = (after["wickets_remaining"] - 1).clip(lower=0)
after["required_rr"]       = np.where(
    after["balls_remaining"] > 0,
    after["runs_needed"] / (after["balls_remaining"] / 6),
    999
)
after["rr_diff"] = after["current_rr"] - after["required_rr"]

wicket_rows["wp_after_wicket"] = model.predict_proba(after)[:, 1]
wicket_rows["wp_swing"]        = wicket_rows["wp_at_wicket"] - wicket_rows["wp_after_wicket"]

swing_by_kind = (wicket_rows.groupby("dismissal_kind")["wp_swing"]
                 .agg(["mean","count"])
                 .rename(columns={"mean":"avg_wp_drop","count":"n_wickets"})
                 .sort_values("avg_wp_drop", ascending=False)
                 .reset_index())
swing_by_kind["avg_wp_drop_pct"] = (swing_by_kind["avg_wp_drop"] * 100).round(1)

print(swing_by_kind[["dismissal_kind","avg_wp_drop_pct","n_wickets"]].to_string(index=False))
swing_by_kind.to_sql("wicket_swing_index", conn, if_exists="replace", index=False)

# ── ANALYSIS 2: Powerplay Leverage Score ─────────────────────────────────────
print("\n" + "="*60)
print("ANALYSIS 2: Powerplay Leverage — does a wicket hurt more early?")
print("="*60)

wicket_rows["phase"] = pd.cut(
    wicket_rows["cum_overs"],
    bins=[0, 6, 15, 20],
    labels=["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"]
)

leverage = (wicket_rows.groupby("phase", observed=True)["wp_swing"]
            .agg(["mean","count"])
            .rename(columns={"mean":"avg_wp_drop","count":"wickets"})
            .reset_index())
leverage["avg_wp_drop_pct"] = (leverage["avg_wp_drop"] * 100).round(1)
print(leverage.to_string(index=False))
leverage.to_sql("powerplay_leverage", conn, if_exists="replace", index=False)

# ── ANALYSIS 3: Choke Detector ────────────────────────────────────────────────
print("\n" + "="*60)
print("ANALYSIS 3: Choke Detector")
print("Which teams had win probability > 80% and still lost?")
print("="*60)

# get max win prob per match for chasing team
inn2_clean["win_prob"] = model.predict_proba(inn2_clean[features])[:, 1]

match_wp = (inn2_clean.groupby(["match_id","batting_team","batting_won"])
            .agg(max_win_prob=("win_prob","max"))
            .reset_index())

# chokes = had >80% win prob but lost
chokes = match_wp[(match_wp["max_win_prob"] >= 0.80) &
                  (match_wp["batting_won"] == 0)]

choke_by_team = (chokes.groupby("batting_team")
                 .agg(choke_count=("match_id","count"))
                 .reset_index()
                 .sort_values("choke_count", ascending=False))

# total matches chased per team (to get choke rate)
total_chases = (match_wp.groupby("batting_team")
                .agg(total_chases=("match_id","count"))
                .reset_index())

choke_by_team = choke_by_team.merge(total_chases, on="batting_team")
choke_by_team["choke_rate_pct"] = (
    choke_by_team["choke_count"] / choke_by_team["total_chases"] * 100
).round(1)

print(choke_by_team.sort_values("choke_rate_pct", ascending=False).to_string(index=False))
choke_by_team.to_sql("choke_detector", conn, if_exists="replace", index=False)

print("\nAll results saved to ipl.db")
print("Tables: wicket_swing_index, powerplay_leverage, choke_detector")
conn.close()
