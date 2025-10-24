# ==========================================================
# 🌟 Northstar Ecosystem — Adaptive Scheduler v3
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# ==========================================================
# 🎯 Minnesota Lottery Official Sources (Auto Prefill)
# ==========================================================
MN_SOURCES = {
    "N5": "https://www.mnlottery.com/games/draw-games/northstar-cash",
    "G5": "https://www.mnlottery.com/games/draw-games/gopher-5",
    "PB": "https://www.mnlottery.com/games/draw-games/powerball"
}

# ==========================================================
# 🧩 Unified Fetch Function (MN Lottery + JSON fallback)
# ==========================================================
def fetch_draws(url, game_key):
    """
    Pulls and parses draw data from the MN Lottery site.
    Falls back to JSON endpoints if HTML format changes.
    """
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()

        # JSON fallback if available
        if resp.headers.get("Content-Type", "").startswith("application/json"):
            data = resp.json()
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and "draws" in data:
                df = pd.DataFrame(data["draws"])
            else:
                df = pd.DataFrame()
        else:
            # Parse HTML results
            soup = BeautifulSoup(resp.text, "html.parser")
            draw_rows = soup.find_all("li", class_=re.compile("winning-numbers__item|draw-result|draw-date"))
            draws = []
            for row in draw_rows:
                text = re.sub(r"\s+", " ", row.get_text(strip=True))
                nums = re.findall(r"\d+", text)
                if len(nums) >= 5:
                    date_match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}", text)
                    date = date_match.group(0) if date_match else "Unknown"
                    draws.append({
                        "date": date,
                        "n1": int(nums[0]),
                        "n2": int(nums[1]),
                        "n3": int(nums[2]),
                        "n4": int(nums[3]),
                        "n5": int(nums[4]),
                        "game": game_key
                    })
            df = pd.DataFrame(draws)

        if df.empty:
            raise ValueError(f"No draw rows parsed for {game_key}")

        # Cleanup + sort
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date", ascending=False).reset_index(drop=True)

        return df

    except Exception as e:
        print(f"⚠️ Error fetching {game_key}: {e}")
        return pd.DataFrame(columns=["date", "n1", "n2", "n3", "n4", "n5", "game"])

# ==========================================================
# 🧠 Adaptive Core Simulation (placeholder for your ecosystem)
# ==========================================================
def adaptive_simulation(df, game_key):
    """
    Placeholder for your Monte Carlo + clustering analysis.
    Currently computes basic stats and confidence indicators.
    """
    if df.empty:
        return pd.DataFrame(), 0

    latest_five = df.head(5)
    freq = pd.Series(np.concatenate([latest_five[['n1','n2','n3','n4','n5']].values])).value_counts()
    top_nums = freq.head(5).index.tolist()

    confidence = round((len(freq) / 70) * 100, 2)
    result = pd.DataFrame({
        "game": [game_key],
        "predicted_set": [top_nums],
        "confidence_%": [confidence]
    })
    return result, confidence

# ==========================================================
# 🔄 Run and Merge All Draws
# ==========================================================
def run_all_draw_updates():
    combined = {}
    for g in ["N5", "G5", "PB"]:
        url = MN_SOURCES.get(g)
        df = fetch_draws(url, g)
        combined[g] = df
    return combined

# ==========================================================
# 📈 Display and Analysis Logic
# ==========================================================
def display_latest_draws():
    st.header("🎯 Live latest draws")
    results = run_all_draw_updates()
    for g, df in results.items():
        if not df.empty:
            latest = df.iloc[0]
            st.success(f"{g}: {latest['n1']} {latest['n2']} {latest['n3']} {latest['n4']} {latest['n5']} ({latest['date'].strftime('%b %d, %Y')})")
        else:
            st.warning(f"{g}: fallback (no file yet)")
    return results

# ==========================================================
# 🪄 Confidence / Prediction Section
# ==========================================================
def show_predictions(draw_data):
    st.subheader("🔮 Adaptive Predictions")
    for g, df in draw_data.items():
        pred_df, conf = adaptive_simulation(df, g)
        if not pred_df.empty:
            st.info(f"{g}: {pred_df['predicted_set'].iloc[0]}  |  Confidence: {pred_df['confidence_%'].iloc[0]}%")
        else:
            st.warning(f"{g}: Not enough data for analysis yet.")

# ==========================================================
# 🧭 Scheduler View
# ==========================================================
def show_schedule():
    st.subheader("🗓️ Schedule (America/Chicago)")
    st.markdown("""
    **N5** — Daily  
    - Post-draw analysis: 6:30 AM  
    - Pre-draw analysis: 9:00 AM, 11:00 AM, 2:00 PM  
    - Final draw & analysis: 3:30 PM  

    **G5** — Monday, Wednesday, Friday  
    - Same times as N5  

    **Powerball** — Monday, Wednesday, Saturday  
    - Same times as N5  
    """)

# ==========================================================
# 🧩 Streamlit Layout
# ==========================================================
def main():
    st.title("🌟 Northstar Ecosystem — Adaptive Scheduler v3")
    st.caption("Adaptive MC + clustering • Trickle-down cross-influence • Live official pulls • Scheduled pre/post runs")

    with st.expander("Show controls", expanded=True):
        selected_games = st.multiselect("Games", ["N5", "G5", "PB"], default=["N5", "G5", "PB"])
        if st.button("▶️ Run selected phase now"):
            st.session_state["draw_data"] = {g: fetch_draws(MN_SOURCES[g], g) for g in selected_games}
            st.success("Live draw data fetched successfully!")

    st.divider()
    draw_data = st.session_state.get("draw_data", display_latest_draws())
    show_predictions(draw_data)
    show_schedule()

# ==========================================================
# 🚀 Run the Streamlit App
# ==========================================================
if __name__ == "__main__":
    main()
