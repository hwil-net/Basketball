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
    # Ensure these .pkl files are in your GitHub root
    rf_model = joblib.load('nba_rf_model.pkl')
    scaler = joblib.load('nba_scaler.pkl')
    return rf_model, scaler

rf_model, scaler = load_assets()

# Must match training order exactly
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
st.markdown("Hybrid Machine Learning + Official NBA Rules Adjuster")

# --- DATA SYNC & LOGO ---
col_search, col_logo = st.columns([3, 1])

# Initialize Defaults
defaults = {f: 0.0 for f in FEATURES}
defaults['pace'] = 100.0
defaults['clock_seconds'] = 300.0
defaults['teamId'] = 1610612747 # Default: Lakers

game_id_input = col_search.text_input("Enter NBA Game ID (e.g. 22000001)", "")

if game_id_input:
    lookup_df = get_game_data(game_id_input)
    if not lookup_df.empty:
        col_search.success("Data Synced from Database")
        for f in FEATURES:
            if f in lookup_df.columns:
                val = lookup_df.iloc[0][f]
                # SAFETY: Only update if value is not Null/NaN
                if pd.notnull(val):
                    defaults[f] = val
    else:
        col_search.warning("ID not found. Entering Manual Mode.")

# --- TEAM LOGO (CACHE-BUSTING FIX) ---
raw_t_id = defaults.get('teamId', 1610612747)
# Final safety check before int conversion to prevent TypeError
if raw_t_id is None or (isinstance(raw_t_id, float) and np.isnan(raw_t_id)):
    t_id = 1610612747
else:
    t_id = int(raw_t_id)

logo_url = f"https://cdn.nba.com/logos/nba/{t_id}/global/L/logo.svg?v={t_id}"
col_logo.image(logo_url, width=120)

# --- LIVE INPUTS ---
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("⏱️ Game Clock")
    m = st.number_input("Minutes Left", 0, 12, int(defaults['clock_seconds'] // 60))
    s = st.number_input("Seconds Left", 0, 59, int(defaults['clock_seconds'] % 60))
    
    clock_seconds = (m * 60) + s
    game_clock_elapsed = 2160 + (720 - clock_seconds)
    
    # Visual Clock Graphic
    st.markdown(f"""
    <div style="background-color:black; color:orange; padding:15px; border-radius:10px; text-align:center; font-family:monospace; font-size:45px; border: 3px solid #444;">
        4TH {m:02d}:{s:02d}
    </div>
    """, unsafe_allow_html=True)
    
    pace = st.number_input("Game Pace", 0.0, 250.0, float(defaults['pace']))

with c2:
    st.subheader("📊 Scoreboard")
    margin = st.number_input("Scoring Margin", 0, 100, int(defaults['scoring_margin']))
    is_clutch = st.checkbox("Clutch Mode (Margin ≤ 5)", value=(margin <= 5))
    
    st.write("**Penalty State**")
    total_fouls = st.number_input("Team Fouls (Total)", 0, 150, int(defaults['is_foul_cum']))
    q4_fouls = st.number_input("Fouls this Quarter", 0, 30, int(defaults['is_foul_last4_cum']))
    
    # --- NBA RULE: BONUS ---
    in_bonus = q4_fouls >= 5
    if in_bonus:
        st.error("🚨 BONUS ACTIVE (Penalty)")
    
    # Foul Sanity Check
    if total_fouls > 90:
        st.error("⚠️ Foul count exceeds realistic max (90).")

with c3:
    st.subheader("🛑 Stoppages")
    reviews = st.number_input("Official Reviews", 0, 10, int(defaults['is_review_cum']))
    
    # --- NBA RULE: TIMEOUTS ---
    timeouts_used = st.number_input("Timeouts Used", 0, 14, int(defaults['is_timeout_last4_cum']))
    timeouts_rem = max(0, 7 - timeouts_used)
    
    # Late game restriction: Max 2 in last 2 mins
    if clock_seconds <= 120:
        timeouts_rem = min(timeouts_rem, 2)
        st.warning("⏱️ Timeouts capped at 2 (Last 2m)")
    
    st.metric("Timeouts Remaining", timeouts_rem)
    ft_last4 = st.number_input("Free Throws (Last 4m)", 0, 50, int(defaults['is_ft_last4_cum']))

# --- HYBRID PREDICTION ENGINE ---
st.divider()

input_dict = {
    'clock_seconds': clock_seconds,
    'game_clock_elapsed': game_clock_elapsed,
    'scoring_margin': margin,
    'is_close_game': int(is_clutch),
    'last_four_min': int(clock_seconds <= 240),
    'last_two_min': int(clock_seconds <= 120),
    'foul_rate_20': float(defaults['foul_rate_20']),
    'is_foul_cum': total_fouls,
    'is_foul_last4_cum': q4_fouls,
    'is_timeout_last4_cum': timeouts_used,
    'is_ft_last4_cum': ft_last4,
    'is_turnover_last4_cum': int(defaults['is_turnover_last4_cum']),
    'is_review_cum': reviews,
    'teamId': t_id,
    'pace': pace,
    'real_vs_clock_ratio': 2.7 # Standard ratio (Hidden from UI)
}

X_input = pd.DataFrame([input_dict])[FEATURES]

# 1. Base ML Prediction
prediction = rf_model.predict(X_input)[0]

# 2. Rule-Based Adjustments (Manual Logic)
adj = 0
if in_bonus:
    adj += 60  # +1 minute for FT frequency

if timeouts_rem > 0 and clock_seconds < 180:
    adj += (timeouts_rem * 20) # +20s per remaining timeout in clutch

adj += (reviews * 120) # +2 mins per official review

# Final Combined Result
final_total_seconds = max(0, prediction + adj)
m_rem, s_rem = divmod(int(final_total_seconds), 60)

# --- DISPLAY RESULTS ---
st.markdown(f"""
<div style="text-align: center; background-color: #f8f9fb; padding: 30px; border-radius: 20px; border: 1px solid #ddd;">
    <h3 style="color: #555;">Estimated Real-World Time Remaining</h3>
    <h1 style="font-size: 90px; color: #FF4B4B; margin: 0;">{m_rem}m {s_rem:02d}s</h1>
</div>
""", unsafe_allow_html=True)

with st.expander("Model Logic Breakdown"):
    st.write(f"**Statistical Base:** {round(prediction/60, 2)} minutes")
    st.write(f"**Rule Adjustments:** {round(adj/60, 2)} minutes")
    st.write("**Feature Vector Used:**")
    st.dataframe(X_input)
