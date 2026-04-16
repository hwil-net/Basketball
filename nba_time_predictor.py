import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sqlite3

# --- PAGE CONFIG ---
st.set_page_config(page_title="NBA Game Clock Predictor", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # These files must be in the root of your GitHub repository
    rf_model = joblib.load('nba_rf_model.pkl')
    scaler = joblib.load('nba_scaler.pkl')
    return rf_model, scaler

rf_model, scaler = load_assets()

# Feature list must match the exact order used during model training
FEATURES = [
    'clock_seconds',
    'game_clock_elapsed',
    'scoring_margin',
    'is_close_game',
    'last_four_min',
    'last_two_min',
    'foul_rate_20',
    'is_foul_cum',
    'is_foul_last4_cum',
    'is_timeout_last4_cum',
    'is_ft_last4_cum',
    'is_turnover_last4_cum',
    'is_review_cum',
    'teamId',
    'pace',
    'real_vs_clock_ratio'
]

# --- DATABASE LOGIC ---
def get_game_data(game_id):
    try:
        g_id_int = int(game_id)
        # Identify which of the 8 partition files contains this game via modulo
        partition_index = g_id_int % 8
        db_name = f'nba_part_{partition_index}.db'
        
        conn = sqlite3.connect(db_name)
        # Query the latest state of the game
        query = "SELECT * FROM play_by_play WHERE game_id = ? ORDER BY game_clock_elapsed DESC LIMIT 1"
        df = pd.read_sql(query, conn, params=(g_id_int,))
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# --- UI HEADER ---
st.title("NBA Advanced Game Duration Predictor")
st.markdown("Input a Game ID to auto-fill stats or manually adjust features below to see the prediction update in real-time.")

# --- GAME LOOKUP ---
game_id_input = st.text_input("Enter NBA Game ID (e.g. 22000001)", "")

# Initialize default values with safe types to prevent TypeErrors
defaults = {f: 0.0 for f in FEATURES}
defaults['pace'] = 100.0
defaults['clock_seconds'] = 300.0
defaults['game_clock_elapsed'] = 2580.0
defaults['teamId'] = 0

if game_id_input:
    lookup_df = get_game_data(game_id_input)
    if not lookup_df.empty:
        st.success(f"Game data found for Game ID {game_id_input}")
        for f in FEATURES:
            if f in lookup_df.columns:
                val = lookup_df.iloc[0][f]
                # Only update if the value is not null
                if pd.notnull(val):
                    defaults[f] = val
    else:
        st.warning("Game ID not found in the database partitions. Please enter stats manually.")

# --- FEATURE INPUTS ---
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Time and Pace")
    
    # Fan-friendly time inputs
    fan_mins = st.number_input("Minutes Left in 4th", 0, 12, int(defaults['clock_seconds'] // 60))
    fan_secs = st.number_input("Seconds Left in 4th", 0, 59, int(defaults['clock_seconds'] % 60))
    
    # Internal conversion for model features
    clock_seconds = (fan_mins * 60) + fan_secs
    # NBA Q4 starts at 2160 elapsed seconds (3 quarters * 720s)
    game_clock_elapsed = 2160 + (720 - clock_seconds)
    
    st.info(f"Current Game State: {fan_mins:02d}:{fan_secs:02d} remaining in the 4th Quarter")
    
    pace = st.number_input("Current Pace", 0.0, 250.0, float(defaults['pace']))

with col2:
    st.subheader("Score and Context")
    scoring_margin = st.number_input("Scoring Margin", 0, 100, int(defaults['scoring_margin']))
    
    # Safety check for teamId
    teamId_val = defaults.get('teamId', 0)
    teamId = st.number_input("Team ID", value=int(teamId_val if teamId_val is not None else 0))
    
    st.write("Game State Flags")
    is_close_game = st.checkbox("Is Close Game (Margin ≤ 5)", value=(scoring_margin <= 5))
    last_four_min = st.checkbox("Last 4 Minutes", value=(clock_seconds <= 240))
    last_two_min = st.checkbox("Last 2 Minutes", value=(clock_seconds <= 120))

with col3:
    st.subheader("Fouls and Stoppages")
    foul_rate_20 = st.number_input("Foul Rate (Last 20 Events)", 0.0, 20.0, float(defaults['foul_rate_20']))
    is_foul_cum = st.number_input("Total Cumulative Fouls", 0, 200, int(defaults['is_foul_cum']))
    is_foul_last4_cum = st.number_input("Fouls in Last 4 Min", 0, 100, int(defaults['is_foul_last4_cum']))
    is_timeout_last4_cum = st.number_input("Timeouts in Last 4 Min", 0, 20, int(defaults['is_timeout_last4_cum']))
    is_ft_last4_cum = st.number_input("Free Throws in Last 4 Min", 0, 100, int(defaults['is_ft_last4_cum']))
    is_turnover_last4_cum = st.number_input("Turnovers in Last 4 Min", 0, 50, int(defaults['is_turnover_last4_cum']))
    is_review_cum = st.number_input("Total Reviews/Challenges", 0, 20, int(defaults['is_review_cum']))

# --- LIVE PREDICTION ---
st.divider()

# Consolidate data for the model
input_dict = {
    'clock_seconds': clock_seconds,
    'game_clock_elapsed': game_clock_elapsed,
    'scoring_margin': scoring_margin,
    'is_close_game': int(is_close_game),
    'last_four_min': int(last_four_min),
    'last_two_min': int(last_two_min),
    'foul_rate_20': foul_rate_20,
    'is_foul_cum': is_foul_cum,
    'is_foul_last4_cum': is_foul_last4_cum,
    'is_timeout_last4_cum': is_timeout_last4_cum,
    'is_ft_last4_cum': is_ft_last4_cum,
    'is_turnover_last4_cum': is_turnover_last4_cum,
    'is_review_cum': is_review_cum,
    'teamId': teamId,
    'pace': pace,
    'real_vs_clock_ratio': 2.7  # Using fixed standard ratio for cleaner UI
}

X_input = pd.DataFrame([input_dict])[FEATURES]

# Perform live prediction
prediction = rf_model.predict(X_input)[0]
final_pred = max(0, prediction)
minutes, seconds = divmod(int(final_pred), 60)

st.markdown("### Live Prediction Result")
st.metric(label="Estimated Real-World Time Remaining", value=f"{minutes}m {seconds:02d}s")

with st.expander("View Technical Feature Data"):
    st.write(X_input)
