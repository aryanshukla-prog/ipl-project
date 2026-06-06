@st.cache_resource
def load_model():
    try:
        with open("win_prob_model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        # Retrain on the fly
        import subprocess
        subprocess.run(["python", "win_probability.py"], check=True)
        with open("win_prob_model.pkl", "rb") as f:
            return pickle.load(f)
