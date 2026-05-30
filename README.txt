to see the working dashboard
https://ipl-project-5lm7obs2kqen6mrffhc73p.streamlit.app/
IPL Win Probability Analyzer
A ball-by-ball IPL analytics engine that predicts win probability at every moment of a chase, and uses those probabilities to find chokes, measure wicket impact, and quantify powerplay leverage.

What this does
Most cricket stats tell you what happened. This project tries to answer a harder question: at any given ball in a chase, how likely was the batting team to actually win?
Once you have that number for every ball of every match, interesting things fall out naturally:

You can draw a win probability curve for any match and see exactly where momentum shifted
You can find teams that were cruising at 80%+ and still lost (choke detector)
You can measure how much a wicket actually hurt — not just "a wicket fell" but "win probability dropped 18 points on that dismissal"
You can compare whether losing a wicket in the powerplay is more damaging than losing one in the death overs


How it works
Data comes from Cricsheet — one JSON file per IPL match, with every ball recorded. parse_cricsheet.py flattens all of that into a SQLite database (ipl.db) with a matches table and a deliveries table.
Features are engineered per ball for every 2nd innings: cumulative runs, runs needed, balls remaining, wickets remaining, required run rate, current run rate, and the difference between the two. The label is simple — did the chasing team win this match, yes or no.
Model is a Gradient Boosting Classifier (200 trees, depth 4, learning rate 0.05). It learns from thousands of chase situations and outputs a probability between 0 and 1 at each ball. The trained model is saved as win_prob_model.pkl.
Analyses run on top of the model's predictions and write results back to ipl.db:

Win probability curve — the full probability timeline for any match
Choke detector — matches where a team's peak win probability crossed 80% but they still lost, grouped by team
Wicket swing index — for each dismissal, the model is re-run with one fewer wicket to compute the exact probability drop, then grouped by dismissal type
Powerplay leverage — same swing calculation, broken down by phase (powerplay, middle overs, death overs)

Dashboard (dash.py) is a Streamlit app that reads from the database and visualises all of the above interactively.

Project structure
ipl-project/
├── ipl_male_json/          # Raw Cricsheet match files (one JSON per match)
├── parse_cricsheet.py      # Parses JSON → SQLite (matches + deliveries tables)
├── win_probability.py      # Feature engineering, model training, all analyses
├── win_prob_model.pkl       # Trained GradientBoostingClassifier
├── ipl.db                  # SQLite database with all raw and derived data
├── dash.py                 # Streamlit dashboard
├── testipl.py              # Tests
├── requirements.txt        # Python dependencies
├── win_probability_curve.html    # Standalone win probability chart
├── choke_detector.html           # Standalone choke analysis chart
└── powerplay_wicket_impact.html  # Standalone powerplay leverage chart

Running it yourself
bash# Install dependencies
pip install -r requirements.txt

# Parse the raw match files into the database
python parse_cricsheet.py

# Train the model and run all analyses
python win_probability.py

# Launch the dashboard
streamlit run dash.py
The standalone HTML files (win_probability_curve.html, etc.) can be opened directly in a browser without running anything.

Stack

Python, SQLite, scikit-learn, pandas, numpy
Streamlit for the dashboard
Cricsheet for match data (https://cricsheet.org)


Background
This started as a way to learn how gradient boosting works by applying it to something I actually follow. The model isn't trying to be a production betting system — it's trained only on 2nd innings states and treats every ball in a won match as a positive example, which is a simplification. But the probabilities it produces are coherent enough to power the choke and wicket analyses in a way that passes the smell test if you know the matches.
