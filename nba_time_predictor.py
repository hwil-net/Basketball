import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="NBA Game Clock Predictor", page_icon="🏀", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_models():
    # Ensure these files are in your GitHub repo
    model = joblib.load('nba_rf_model.pkl')
    scaler = joblib.load('nba_scaler.pkl')
    return model, scaler

rf_model, scaler = load_models()

# The exact feature order from your notebook
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
    'real_vs_clock_ratio',
]

# --- UI ---
st.title("🏀 NBA Advanced Game Duration Predictor")
st.markdown("Manually adjust every feature used by the Random Forest model to see how it affects real-time duration.")

# Organizing inputs into columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.header("⏱️ Time & Pace")
    clock_seconds = st.number_input("Clock Seconds Remaining (in period)", 0, 720, 120)
    game_clock_elapsed = st.number_input("Total Game Clock Elapsed (sec)", 0, 3600, 2760)
    pace = st.number_input("Game Pace", 0.0, 150.0, 100.0)
    real_vs_clock_ratio = st.number_input("Real vs Clock Ratio", 0.0, 10.0, 2.7)

with col2:
    st.header("📊 Score & Context")
    scoring_margin = st.number_input("Scoring Margin (Abs)", 0, 100, 5)
    teamId = st.number_input("Team ID", value=1610612747)
    
    st.write("**Game Flags**")
    is_close_game = st.checkbox("Is Close Game (Margin ≤ 5)", value=True)
    last_four_min = st.checkbox("In Last 4 Minutes", value=True)
    last_two_min = st.checkbox("In Last 2 Minutes", value=False)

with col3:
    st.header("🚫 Fouls & Stoppages")
    foul_rate_20 = st.number_input("Foul Rate (last 20 events)", 0.0, 20.0, 1.0)
    is_foul_cum = st.number_input("Total Cumulative Fouls", 0, 100, 20)
    is_foul_last4_cum = st.number_input("Fouls in Last 4 Min", 0, 30, 4)
    is_timeout_last4_cum = st.number_input("Timeouts in Last 4 Min", 0, 10, 2)
    is_ft_last4_cum = st.number_input("Free Throws in Last 4 Min", 0, 50, 6)
    is_turnover_last4_cum = st.number_input("Turnovers in Last 4 Min", 0, 20, 1)
    is_review_cum = st.number_input("Total Reviews/Challenges", 0, 10, 0)

# --- PREDICTION ---
st.divider()

# Map inputs to a dictionary
input_data = {
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
    'is_
