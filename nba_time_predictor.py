import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="NBA Game Clock Predictor", page_icon="🏀")

# --- LOAD ASSETS ---
@st.cache_resource
def load_models():
    model = joblib.load('nba_rf_model.pkl')
    scaler = joblib.load('nba_scaler.pkl')
    pbp_sample = pd.read_csv('pbp_sample.csv')
    return model, scaler, pbp_sample

rf_model, scaler, pbp = load_models()

# These must match your notebook exactly
FEATURES = [
    'clock_seconds', 'game_clock_elapsed', 'scoring_margin', 'is_close_game',
    'last_four_min', 'last_two_min', 'foul_rate_20', 'is_foul_cum',
    'is_foul_last4_cum', 'is_timeout_last4_cum', 'is_ft_last4_cum',
    'is_turnover_last4_cum', 'is_review_cum', 'teamId', 'pace', 'real_vs_clock_ratio'
]

# --- UI ---
st.title("🏀 NBA Real-Time Duration Predictor")
st.markdown("Predict exactly how much **real-world time** is left in a game.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Game State")
    mins = st.number_input("Q4 Minutes Left", 0, 12, 5)
    secs = st.number_input("Q4 Seconds Left", 0, 59, 0)
    margin = st.number_input("Current Score Margin", 0, 50, 4)
    
with col2:
    st.subheader("Intensity Metrics")
    fouls = st.slider("Fouls in last 4 mins", 0, 15, 2)
    timeouts = st.slider("Timeouts in last 4 mins", 0, 6, 1)
    ratio = st.slider("Real-to-Game Clock Ratio", 1.0, 5.0, 2.7)

# --- PREDICTION LOGIC ---
clock_secs = (mins * 60) + secs
game_clock_elapsed = (3 * 720) + (720 - clock_secs)

# Prepare input row
input_dict = {
    'clock_seconds': clock_secs,
    'game_clock_elapsed': game_clock_elapsed,
    'scoring_margin': margin,
    'is_close_game': 1 if margin <= 5 else 0,
    'last_four_min': 1 if clock_secs <= 240 else 0,
    'last_two_min': 1 if clock_secs <= 120 else 0,
    'foul_rate_20': fouls * 1.5, # Approximation
    'is_foul_cum': 15, 
    'is_foul_last4_cum': fouls,
    'is_timeout_last4_cum': timeouts,
    'is_ft_last4_cum': 2,
    'is_turnover_last4_cum': 1,
    'is_review_cum': 0,
    'pace': 100.0,
    'real_vs_clock_ratio': ratio,
    'teamId': 1610612747 # Default placeholder
}

X_input = pd.DataFrame([input_dict])[FEATURES]

if st.button("Predict Real Time Remaining", type="primary"):
    pred_seconds = rf_model.predict(X_input)[0]
    m, s = divmod(int(pred_seconds), 60)
    
    st.metric(label="Estimated Real Time Left", value=f"{m}m {s:02d}s")
    
    st.info(f"Analysis: Based on a {ratio}x clock ratio and {'clutch' if margin <= 5 else 'blowout'} conditions.")