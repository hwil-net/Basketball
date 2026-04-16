import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sqlite3

# --- PAGE CONFIG ---
st.set_page_config(page_title="NBA Live Time Predictor", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    rf_model = joblib.load('nba_rf_model.pkl')
    scaler = joblib.load('nba_scaler.pkl')
    return rf_model, scaler

rf_model, scaler = load_assets()

# --- FEATURES ---
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
        query = """
        SELECT * FROM play_by_play
        WHERE game_id = ?
        ORDER BY game_clock_elapsed DESC
        LIMIT 1
        """
        df = pd.read_sql(query, conn, params=(g_id_int,))
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# --- UI HEADER ---
st.title("NBA Advanced Game Duration Predictor")

# --- SEARCH + LOGO ---
col_search, col_logo = st.columns([3, 1])

with col_search:
    game_id_input = st.text_input("Enter NBA Game ID (e.g. 22000001)", "")

# --- DEFAULTS ---
defaults = {f: 0.0 for f in FEATURES}
defaults['pace'] = 100.0
defaults['clock_seconds'] = 300.0
defaults['teamId'] = 1610612747  # Lakers fallback

# --- LOAD GAME DATA ---
if game_id_input:
    lookup_df = get_game_data(game_id_input)
    if not lookup_df.empty:
        st.success("Game data synced.")
        for f in FEATURES:
            if f in lookup_df.columns:
                val = lookup_df.iloc[0][f]
                if pd.notnull(val):
                    defaults[f] = val
    else:
        st.warning("Game ID not found. Using manual inputs.")

# --- SAFE TEAM ID HANDLING ---
t_id_raw = defaults.get('teamId', 0)

try:
    if pd.isna(t_id_raw):
        t_id = 0
    else:
        t_id = int(t_id_raw)
except Exception:
    t_id = 0

# --- TEAM LOGO ---
with col_logo:
    logo_url = f"https://cdn.nba.com/logos/nba/{t_id}/global/L/logo.svg"
    st.image(logo_url, width=100)

# --- INPUT UI ---
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Game Clock")

    mins = st.number_input("Minutes Left in 4th", 0, 12, int(defaults['clock_seconds'] // 60))
    secs = st.number_input("Seconds Left in 4th", 0, 59, int(defaults['clock_seconds'] % 60))

    clock_seconds = (mins * 60) + secs
    game_clock_elapsed = 2160 + (720 - clock_seconds)

    st.markdown(f"""
    <div style="background-color:black; color:orange; padding:10px;
                border-radius:10px; text-align:center;
                font-family:monospace; font-size:40px;
                border: 2px solid #333;">
        4TH {mins:02d}:{secs:02d}
    </div>
    """, unsafe_allow_html=True)

    pace = st.number_input("Current Pace", 0.0, 250.0, float(defaults['pace']))

with col2:
    st.subheader("Scoreboard")

    margin = st.number_input("Scoring Margin", 0, 100, int(defaults['scoring_margin']))

    is_close = st.checkbox("Clutch Time (Margin ≤ 5)", value=(margin <= 5))
    is_last_4 = st.checkbox("Final 4 Minutes", value=(clock_seconds <= 240))
    is_last_2 = st.checkbox("Two Minute Warning", value=(clock_seconds <= 120))

with col3:
    st.subheader("NBA Rules & Stoppages")

    reviews = st.number_input("Official Reviews", 0, 10, int(defaults['is_review_cum']))
    timeouts = st.number_input("Timeouts (Last 4 Min)", 0, 20, int(defaults['is_timeout_last4_cum']))
    fouls_last_4 = st.number_input("Fouls (Last 4 Min)", 0, 30, int(defaults['is_foul_last4_cum']))
    ft_last_4 = st.number_input("Free Throws (Last 4 Min)", 0, 50, int(defaults['is_ft_last4_cum']))

# --- MODEL INPUT ---
input_dict = {
    'clock_seconds': clock_seconds,
    'game_clock_elapsed': game_clock_elapsed,
    'scoring_margin': margin,
    'is_close_game': int(is_close),
    'last_four_min': int(is_last_4),
    'last_two_min': int(is_last_2),
    'foul_rate_20': float(defaults['foul_rate_20']),
    'is_foul_cum': int(defaults['is_foul_cum']),
    'is_foul_last4_cum': fouls_last_4,
    'is_timeout_last4_cum': timeouts,
    'is_ft_last4_cum': ft_last_4,
    'is_turnover_last4_cum': int(defaults['is_turnover_last4_cum']),
    'is_review_cum': reviews,
    'teamId': t_id,
    'pace': pace,
    'real_vs_clock_ratio': 2.7
}

X_input = pd.DataFrame([input_dict])[FEATURES]

# --- PREDICTION ---
prediction = rf_model.predict(X_input)[0]
final_pred = max(0, prediction)
minutes_rem, seconds_rem = divmod(int(final_pred), 60)

# --- OUTPUT ---
st.divider()
col_out1, col_out2 = st.columns([2, 1])

with col_out1:
    st.markdown("### Estimated Real-World Time Remaining")
    st.metric("", f"{minutes_rem}m {seconds_rem:02d}s")

with col_out2:
    st.write("**Model Context:**")
    if reviews > 0:
        st.caption("Review delays included (~2.5 min each).")
    if is_close:
        st.caption("Clutch time → more timeouts expected.")
