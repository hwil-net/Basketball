import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sqlite3

# --- PAGE CONFIG ---
st.set_page_config(page_title="NBA Live Logic Engine", layout="wide")

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
    'clock_seconds', 'game_clock_elapsed', 'scoring_margin', 'is_close_game',
    'last_four_min', 'last_two_min', 'foul_rate_20', 'is_foul_cum',
    'is_foul_last4_cum', 'is_timeout_last4_cum', 'is_ft_last4_cum',
    'is_turnover_last4_cum', 'is_review_cum', 'teamId', 'pace', 'real_vs_clock_ratio'
]

# --- DATABASE LOGIC ---
def get_game_data(game_id):
    try:
        g_id_int = int(game_id)
        partition_index = g_id_int % 8
        db_name = f'nba_part_{partition_index}.db'
        conn = sqlite3.connect(db_name)
        query = "SELECT * FROM play_by_play WHERE game_id = ? ORDER BY game_clock_elapsed DESC LIMIT 1"
        df = pd.read_sql(query, conn, params=(g_id_int,))
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# --- UI HEADER ---
st.title("🏀 NBA Live Logic Engine")
st.markdown("A hybrid ML + Rules-Based Predictor for Game Duration.")

# --- GAME LOOKUP & TEAM LOGO ---
col_search, col_logo = st.columns([3, 1])

defaults = {f: 0.0 for f in FEATURES}
defaults['pace'] = 100.0
defaults['clock_seconds'] = 300.0
defaults['teamId'] = 1610612747 # Default Lakers ID

with col_search:
    game_id_input = st.text_input("Enter NBA Game ID (e.g. 22000001)", "")

if game_id_input:
    lookup_df = get_game_data(game_id_input)
    if not lookup_df.empty:
        st.success(f"Game data synced.")
        for f in FEATURES:
            if f in lookup_df.columns:
                defaults[f] = lookup_df.iloc[0][f]
    else:
        st.warning("Game ID not found. Using manual inputs.")

with col_logo:
    t_id = int(defaults['teamId'])
    # Added cache-busting trick to force logo refresh
    logo_url = f"https://cdn.nba.com/logos/nba/{t_id}/global/L/logo.svg?v={t_id}"
    st.image(logo_url, width=100)

# --- FEATURE INPUTS ---
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("⏱️ Game Clock")
    mins = st.number_input("Minutes Left in 4th", 0, 12, int(defaults['clock_seconds'] // 60))
    secs = st.number_input("Seconds Left in 4th", 0, 59, int(defaults['clock_seconds'] % 60))
    
    clock_seconds = (mins * 60) + secs
    game_clock_elapsed = 2160 + (720 - clock_seconds)
    
    st.markdown(f"""
    <div style="background-color:black; color:orange; padding:10px; border-radius:10px; text-align:center; font-family:monospace; font-size:40px; border: 2px solid #333;">
        4TH {mins:02d}:{secs:02d}
    </div>
    """, unsafe_allow_html=True)
    
    pace = st.number_input("Current Pace", 0.0, 250.0, float(defaults['pace']))

with col2:
    st.subheader("📊 Scoreboard")
    margin = st.number_input("Scoring Margin", 0, 100, int(defaults['scoring_margin']))
    is_close = st.checkbox("Clutch Time (Margin ≤ 5)", value=(margin <= 5))
    
    # --- NBA RULES ENGINE: FOULS & BONUS ---
    st.write("**Foul Tracker**")
    team_fouls = st.number_input("Total Team Fouls", 0, 120, int(defaults['is_foul_cum']))
    quarter_fouls = st.number_input("Fouls this Quarter", 0, 30, int(defaults['is_foul_last4_cum']))
    
    # Bonus Logic
    in_bonus = quarter_fouls >= 5
    if in_bonus:
        st.error("🚨 BONUS ACTIVE: Automatic Free Throws")
    
    # Foul Sanity Check
    if team_fouls > 90:
        st.error("⚠️ Foul count exceeds realistic maximum (90).")

with col3:
    st.subheader("🛑 Stoppages & Logic")
    reviews = st.number_input("Official Reviews", 0, 10, int(defaults['is_review_cum']))
    
    # --- NBA RULES ENGINE: TIMEOUTS ---
    TOTAL_TIMEOUTS_AVAIL = 7
    timeouts_used = st.number_input("Timeouts Used", 0, 14, int(defaults['is_timeout_last4_cum']))
    timeouts_remaining = max(0, TOTAL_TIMEOUTS_AVAIL - timeouts_used)
    
    # Late game restriction: Max 2 in last 2 minutes
    if clock_seconds <= 120:
        timeouts_remaining = min(timeouts_remaining, 2)
        st.warning("⏱️ Late game: Timeouts capped at 2.")
    
    st.metric("Timeouts Remaining", timeouts_remaining)
    
    ft_last_4 = st.number_input("FTs in Last 4 Min", 0, 50, int(defaults['is_ft_last4_cum']))

# --- HYBRID PREDICTION ENGINE ---
st.divider()

input_dict = {
    'clock_seconds': clock_seconds,
    'game_clock_elapsed': game_clock_elapsed,
    'scoring_margin': margin,
    'is_close_game': int(is_close),
    'last_four_min': int(clock_seconds <= 240),
    'last_two_min': int(clock_seconds <= 120),
    'foul_rate_20': float(defaults['foul_rate_20']),
    'is_foul_cum': team_fouls,
    'is_foul_last4_cum': quarter_fouls,
    'is_timeout_last4_cum': timeouts_used,
    'is_ft_last4_cum': ft_last_4,
    'is_turnover_last4_cum': int(defaults['is_turnover_last4_cum']),
    'is_review_cum': reviews,
    'teamId': int(defaults['teamId']),
    'pace': pace,
    'real_vs_clock_ratio': 2.7
}

X_input = pd.DataFrame([input_dict])[FEATURES]

# ML Baseline
prediction = rf_model.predict(X_input)[0]

# --- RULE-BASED ADJUSTMENTS ---
rule_adjustment = 0

# Bonus increases real time due to more free throw stoppages
if in_bonus:
    rule_adjustment += 60  # +1 minute floor

# Timeouts in the clutch significantly extend real-world time
if timeouts_remaining > 0 and clock_seconds < 180:
    rule_adjustment += (timeouts_remaining * 25) # Approx 25 sec per remaining timeout

# Final manual correction for reviews if the model under-estimates
rule_adjustment += (reviews * 120) # 2 minutes per review

final_pred = max(0, prediction + rule_adjustment)
minutes_rem, seconds_rem = divmod(int(final_pred), 60)

# --- DISPLAY RESULTS ---
st.markdown(f"""
<div style="text-align: center;">
    <h3>Estimated Real-World Time Remaining</h3>
    <h1 style="font-size: 80px; color: #FF4B4B;">{minutes_rem}m {seconds_rem:02d}s</h1>
</div>
""", unsafe_allow_html=True)

with st.expander("View Logic Details"):
    st.write("**Base ML Prediction:**", round(prediction / 60, 2), "minutes")
    st.write("**Rule Adjustments Added:**", round(rule_adjustment / 60, 2), "minutes")
    st.write("**Feature Vector:**")
    st.dataframe(X_input)
