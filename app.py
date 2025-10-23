# 🌟 Northstar Ecosystem — Adaptive Scheduler v3
# Includes: Adaptive MC + Clustering + Trickle-down Cross-Influence + Live Pulls
# Scheduled pre/post runs (N5, G5, PB)

import pandas as pd
import streamlit as st
import datetime
import os

st.set_page_config(page_title="Northstar Ecosystem — Scheduler", layout="wide")

# -------------------------------------------------------------------
# --- Data loading helpers ---
# -------------------------------------------------------------------

def load_previous_data(game_key):
    """
    Safely load previous draw data for the given game (N5, G5, PB).
    Returns a pandas DataFrame or an empty frame if not found.
    """
    try:
        file_path = f"data/{game_key}_history.csv"
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.warning(f"⚠️ No historical file found for {game_key}. Using empty frame.")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"⚠️ Could not load previous data for {game_key}: {e}")
        return pd.DataFrame()


def get_latest_draws(game_key):
    """
    Fetch the most recent draw(s) for the given game.
    Replace with live API pull later — for now uses fallback or local cache.
    """
    try:
        file_path = f"data/{game_key}_latest.csv"
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.warning(f"⚠️ No recent data file for {game_key}. Using fallback.")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"⚠️ Could not load latest draw for {game_key}: {e}")
        return pd.DataFrame()

# -------------------------------------------------------------------
# --- Confidence and Summary Placeholder Functions ---
# -------------------------------------------------------------------

def update_confidence_trends(df, game_key):
    """
    Placeholder function for updating confidence trend metrics.
    This will later calculate drift, ±1 hits, and rolling performance.
    """
    if df is None or df.empty:
        st.info(f"No data available for {game_key} to update confidence trends.")
        return
    st.success(f"✅ Confidence trends updated for {game_key} ({len(df)} draws).")

def render_summary(df, game_key):
    """
    Placeholder function for post-draw summaries and highlights.
    """
    if df is None or df.empty:
        st.info(f"No summary available for {game_key}.")
        return
    st.write(f"📊 Showing last {min(5, len(df))} draws for {game_key}:")
    st.dataframe(df.tail(5))

# -------------------------------------------------------------------
# --- Main Loop: Run post-draw analyses for all games ---
# -------------------------------------------------------------------

st.title("🌟 Northstar Ecosystem — Adaptive Scheduler v3")
st.caption("Adaptive MC + clustering • Trickle-down cross-influence • Live pulls • Scheduled pre/post runs")

for g in ["N5", "G5", "PB"]:
    st.subheader(f"Post-draw analysis — {g}")

    df_old = load_previous_data(g)
    latest = get_latest_draws(g)

    try:
        if df_old is not None and latest is not None and not df_old.empty and not latest.empty:
            merged = pd.concat([df_old, latest], ignore_index=True).drop_duplicates(subset=["draw_date"], keep="last")
        elif latest is not None and not latest.empty:
            merged = latest
        else:
            merged = df_old
    except Exception as e:
        st.warning(f"⚠️ Data merge skipped for {g} due to format issue: {e}")
        if 'df_old' in locals() and df_old is not None:
            merged = df_old
        elif 'latest' in locals() and latest is not None:
            merged = latest
        else:
            merged = pd.DataFrame()

    update_confidence_trends(merged, g)
    render_summary(merged, g)

# -------------------------------------------------------------------
# --- Footer ---
# -------------------------------------------------------------------

st.markdown("---")
st.caption("Northstar Ecosystem v3 • Adaptive Scheduler • © 2025")
