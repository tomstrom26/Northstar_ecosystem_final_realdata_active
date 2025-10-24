# app.py — Northstar Ecosystem (Adaptive v3, Official-site pulls)
# ----------------------------------------------------------------
import os, json, re, io, time, math, random, datetime as dt
from pathlib import Path
from dateutil import tz
from dateutil.parser import parse as dtparse

import pandas as pd
import numpy as np
import altair as alt
import requests
from bs4 import BeautifulSoup

import streamlit as st


import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

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

        # Try JSON API first if present
        if resp.headers.get("Content-Type", "").startswith("application/json"):
            data = resp.json()
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and "draws" in data:
                df = pd.DataFrame(data["draws"])
            else:
                df = pd.DataFrame()
        else:
            # HTML fallback parse
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

        # Clean and sort
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date", ascending=False).reset_index(drop=True)

        return df

    except Exception as e:
        print(f"⚠️ Error fetching {game_key}: {e}")
        return pd.DataFrame(columns=["date", "n1", "n2", "n3", "n4", "n5", "game"])

# ==========================================================
# 🧠 Automated Update Cycle (integrates sources)
# ==========================================================
def run_all_draw_updates():
    combined = {}
    for g in ["N5", "G5", "PB"]:
        url = MN_SOURCES.get(g)
        df = fetch_draws(url, g)
        combined[g] = df
    return combined

# ==========================================================
# 🪄 Integrate into Streamlit Display
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
# -----------------------------
# Paths & persistence
# -----------------------------
DATA_DIR = Path("./data"); DATA_DIR.mkdir(exist_ok=True)
SOURCES_JSON = DATA_DIR / "sources.json"
CACHE_TTL_SECONDS = 15 * 60  # re-pull at most every 15 minutes

GAMES = {
    "N5": {"name": "North 5", "count": 5, "pool": 34},
    "G5": {"name": "Gopher 5", "count": 5, "pool": 47},
    "PB": {"name": "Powerball", "count": 5, "pool": 69, "bonus_name": "PB", "bonus_pool": 26},
}

TZ = tz.gettz("America/Chicago")

# -----------------------------
# Schedules you requested
# -----------------------------
SCHEDULE = {
    "N5": {
        "days": ["mon","tue","wed","thu","fri","sat","sun"],
        "post":  "06:30",
        "pre":   ["09:00","11:00","14:00"],
        "final": "15:30",
    },
    "G5": {
        "days": ["mon","wed","fri"],
        "post":  "06:30",
        "pre":   ["09:00","11:00","14:00"],
        "final": "15:30",
    },
    "PB": {
        "days": ["mon","wed","sat"],
        "post":  "06:30",
        "pre":   ["09:00","11:00","14:00"],
        "final": "15:30",
    },
}

# -----------------------------
# Helpers
# -----------------------------
def now_ct():
    return dt.datetime.now(TZ)

def load_sources():
    if SOURCES_JSON.exists():
        return json.loads(SOURCES_JSON.read_text())
    return {"N5":"","G5":"","PB":""}

def save_sources(src):
    SOURCES_JSON.write_text(json.dumps(src, indent=2))

def csv_path(game):       return DATA_DIR / f"{game}_historical.csv"
def latest_path(game):    return DATA_DIR / f"{game}_latest.json"
def cache_stamp_path(g):  return DATA_DIR / f"{g}_pulled_at.txt"

def save_latest(game, payload:dict):
    latest_path(game).write_text(json.dumps(payload, indent=2))

def load_latest(game):
    p = latest_path(game)
    if p.exists():
        return json.loads(p.read_text())
    return {}

def recently_pulled(game, ttl=CACHE_TTL_SECONDS):
    p = cache_stamp_path(game)
    if not p.exists(): return False
    try:
        ts = dtparse(p.read_text())
        return (now_ct() - ts).total_seconds() < ttl
    except Exception:
        return False

def set_pulled(game): cache_stamp_path(game).write_text(now_ct().isoformat())

def parse_ints(tokens):
    res=[]
    for t in tokens:
        m = re.findall(r"\d+", str(t))
        res += [int(x) for x in m]
    return res

# -----------------------------
# Official-site scraping
# -----------------------------
def pull_official_table(url:str)->pd.DataFrame:
    """
    Tries to parse a results TABLE from an official page.
    Looks for rows containing date + 5–6 numbers.
    Returns DataFrame with columns:
      date, n1..n5, bonus (optional), source_url
    """
   
# -----------------------------
# Analysis (Adaptive MC + Trickle)
# -----------------------------
def adaptive_mc(game:str, df:pd.DataFrame, trickle_seed:dict|None=None)->dict:
    """
    Produces:
      - top candidates
      - confidence score (0-100)
      - trend metrics
    """
    if df is None or df.empty: 
        return {"candidates":[],"conf":0,"note":"no-data"}

    # recent window
    recent = df.tail(40).copy()

    # frequency weight base
    cols = [c for c in df.columns if re.fullmatch(r"n[1-5]", c)]
    freq = pd.Series(dtype=int)
    for c in cols:
        freq = freq.add(recent[c].value_counts(), fill_value=0)
    freq = freq.fillna(0)

    # cluster bonus: reward pairs/triples that repeat in last 12 draws
    cluster_bonus = pd.Series(0, index=freq.index)
    tail = recent.tail(12)
    for _, row in tail.iterrows():
        s = set(int(row[c]) for c in cols)
        for n in s:
            cluster_bonus.loc[n] = cluster_bonus.get(n,0) + (2 if len(s)>=3 else 1)

    # trickle-down cross influence
    trickle = pd.Series(0, index=freq.index)
    if trickle_seed:
        for n, w in trickle_seed.items():
            n=int(n)
            if n in trickle.index: trickle.loc[n] += w

    # combine weights
    w = (freq*1.0) + (cluster_bonus*0.72) + (trickle*0.28)
    w = w.sort_values(ascending=False)

    # Monte-Carlo sampling with anti-repeat preference
    last = set(int(x) for x in recent.tail(1)[cols].values.flatten().tolist())
    choices = []
    pool = list(w.index)
    probs = (w / w.sum()).reindex(pool).fillna(0.0).values

    rng = np.random.default_rng(42)
    trials=10000
    bucket = {}
    for _ in range(trials):
        picks = set()
        # bias away from immediate repeats
        local_w = w.copy()
        local_w.loc[list(last)] = local_w.loc[list(last)] * 0.6
        while len(picks) < GAMES[game]["count"]:
            pick = rng.choice(local_w.index, p=local_w/local_w.sum())
            if pick not in picks:
                picks.add(pick)
        key = tuple(sorted(int(x) for x in picks))
        bucket[key] = bucket.get(key,0)+1

    ranked = sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)
    candidates = [{"nums":list(k), "score":round(v/trials*100,2)} for k,v in ranked[:5]]
    conf = round(np.mean([c["score"] for c in candidates])*1.35, 2)
    return {"candidates":candidates, "conf":min(100, conf), "note":"ok"}

def build_trickle_seed(base_pick:list)->dict:
    # Seed weights decay across the space
    seed={}
    for n in base_pick:
        seed[n]= seed.get(n,0)+1.0
        # small neighborhood boost
        for d in (-1,1):
            seed[n+d]= seed.get(n+d,0)+0.4
    return seed

# -----------------------------
# Scheduled actions
# -----------------------------
def today_key(): return now_ct().strftime("%a").lower()[:3]

def time_in_list(target_hhmm:str)->bool:
    hh,mm = map(int, target_hhmm.split(":"))
    now = now_ct()
    return now.hour==hh and abs(now.minute-mm)<=2  # 2-minute window

def should_run(game:str, phase:str)->bool:
    day = today_key()
    sc = SCHEDULE[game]
    if day not in sc["days"]: return False
    if phase=="post":  return time_in_list(sc["post"])
    if phase=="final": return time_in_list(sc["final"])
    if phase=="pre":   return any(time_in_list(t) for t in sc["pre"])
    return False

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Northstar — Adaptive v3", layout="wide")
st.title("⭐ Northstar Ecosystem — Adaptive Scheduler v3")
st.caption("Adaptive MC + clustering • Trickle-down cross-influence • Live official pulls • Scheduled pre/post runs")

# Sidebar: source URLs + manual actions
st.sidebar.header("Data sources (official)")
sources = load_sources()

# ✅ Auto-default official URLs if empty
defaults = {
    "N5": "https://www.mnlottery.com/winning-numbers?selectedGames[]=16",
    "G5": "https://www.mnlottery.com/winning-numbers?selectedGames[]=10",
    "PB": "https://www.mnlottery.com/winning-numbers?selectedGames[]=12",
}
for k, v in defaults.items():
    if not sources.get(k):
        sources[k] = v
save_sources(sources)  # persist once

for g in ["N5","G5","PB"]:
    ...
    default = sources.get(g,"")
    sources[g] = st.sidebar.text_input(
        f"{GAMES[g]['name']} results URL", 
        value=default, 
        placeholder="Paste official results page URL"
    )
if st.sidebar.button("💾 Save sources"):
    save_sources(sources)
    st.sidebar.success("Saved!")

col1,col2,col3 = st.columns(3)

# -----------------------------
# Pull & update storage (when needed)
# -----------------------------
pull_errors = {}
for g in ["N5","G5","PB"]:
    url = sources.get(g,"").strip()
    try:
        if url and not recently_pulled(g):
            df_new = pull_official_table(url)
            all_df = upsert_historical(g, df_new)
            set_pulled(g)
            save_latest(g, {"ts":now_ct().isoformat(), "rows":len(df_new)})
    except Exception as e:
        pull_errors[g]= str(e)

# Header cards - live status
with col1:
    st.subheader("🎯 Live latest draws")
    for g in ["N5","G5","PB"]:
        p = csv_path(g)
        if p.exists():
            df = pd.read_csv(p)
            row = df.tail(1).to_dict("records")[0]
            nums = [int(row.get(f"n{i}",0)) for i in range(1,6)]
            label = f"{g}: {', '.join(map(str, nums))}"
            if "bonus" in row and not (pd.isna(row["bonus"])):
                label += f"  |  {GAMES[g].get('bonus_name','B')}: {int(row['bonus'])}"
            st.success(label)
        else:
            st.warning(f"{g}: fallback (no file yet)")

with col2:
    st.subheader("🗓️ Schedule (America/Chicago)")
    for g in ["N5","G5","PB"]:
        sc = SCHEDULE[g]
        st.caption(f"**{g}** — days: {', '.join(sc['days'])} • post {sc['post']} • pre {', '.join(sc['pre'])} • final {sc['final']}")

with col3:
    st.subheader("⚙️ Controls")
    trg = st.selectbox("Run phase", ["post","pre","final"])
    trg_game = st.multiselect("Games", ["N5","G5","PB"], default=["N5","G5","PB"])
    manual_run = st.button("▶️ Run selected phase now")

if pull_errors:
    st.info("Some sources could not be parsed (you can still use fallbacks).")
    with st.expander("Show pull errors"):
        for g, msg in pull_errors.items():
            st.error(f"{g}: {msg}")

st.markdown("---")

# -----------------------------
# Core run: post/pre/final + analysis + trickle
# -----------------------------
def load_hist(game):
    p = csv_path(game)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def run_phase(game, phase, trickle_seed=None):
    st.subheader(f"{phase.capitalize()} analysis — {game}")
    hist = load_hist(game)

    if hist.empty:
        st.warning(f"No historical file found for {game}. Using empty frame.")
        df = pd.DataFrame(columns=["date","n1","n2","n3","n4","n5","bonus"])
    else:
        df = hist.copy()

    # compute analysis
    result = adaptive_mc(game, df, trickle_seed=trickle_seed)
    if result["note"]!="ok":
        st.info("No data available to update confidence trends.")
        st.info("No summary available.")
        return {"seed":None}

    # show candidates
    cand_df = pd.DataFrame([{"pick":"-".join(map(str,c["nums"])), "score":c["score"]} for c in result["candidates"]])
    st.write(cand_df)
    st.metric("Confidence", f"{result['conf']}")

    # save a quick summary for the session
    summary = {
        "ts": now_ct().isoformat(),
        "phase": phase,
        "game": game,
        "top": result["candidates"][0]["nums"] if result["candidates"] else [],
        "conf": result["conf"],
    }
    save_latest(game, summary)

    # trend chart (conf over time from saved latest files if you want; here just show last confidence)
    return {"seed": build_trickle_seed(summary["top"]) if summary["top"] else None}

# Decide what to run
to_run=[]
if manual_run:
    for g in trg_game:
        to_run.append((g, trg))
else:
    # background clock tick (every refresh)
    for g in ["N5","G5","PB"]:
        for phase in ("post","pre","final"):
            if should_run(g, phase):
                to_run.append((g, phase))

# Execute with trickle-down N5 -> G5 -> PB
if to_run:
    # Order them by N5, G5, PB for trickle
    order = {"N5":0,"G5":1,"PB":2}
    to_run = sorted(to_run, key=lambda x: order[x[0]])
    seeds = {"N5":None,"G5":None}
    for g, phase in to_run:
        if g=="N5":
            out = run_phase("N5", phase, trickle_seed=None)
            seeds["N5"]= out["seed"]
        elif g=="G5":
            out = run_phase("G5", phase, trickle_seed=seeds["N5"])
            seeds["G5"]= out["seed"]
        elif g=="PB":
            _ = run_phase("PB", phase, trickle_seed=seeds["G5"])

st.markdown("---")
st.caption("v3 • Official-source pulls with resilient parser • Local persistence • Adaptive MC • Trickle-down cross-influence")
