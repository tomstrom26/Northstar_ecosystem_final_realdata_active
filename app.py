# ==========================================================
# 🌟 Northstar Ecosystem — v4.0 (Autonomous Edition)
# ==========================================================
# • Official MN pulls (robust HTML parser + 30m cache)
# • Schedule windows (CST) + safe auto-tick per refresh
# • Persistent storage (/data) + weekly .zip archives + manifest
# • Confidence & performance logs + charts
# • Trickle-down cross-influence (N5 → G5/PB; adaptive by confidence)
# • Optional GitHub persistence via st.secrets["github"]
# ==========================================================

import os, io, re, json, zipfile, hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytz
import requests
from bs4 import BeautifulSoup

import streamlit as st

# -----------------------------
# Config / paths
# -----------------------------

APP_VER = "4.0"
TZ = pytz.timezone("America/Chicago")
ROOT = Path(".")
DATA = ROOT / "data"
LOGS = DATA / "logs"
ARCH = DATA / "archives"
DATA.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True); ARCH.mkdir(exist_ok=True)

CONF_PATH = DATA / "confidence_trends.csv"
PERF_PATH = DATA / "performance_log.csv"
MANIFEST = DATA / "manifest.json"

GAMES = ["N5","G5","PB"]
MN_SOURCES = {
    "N5": "https://www.mnlottery.com/games/draw-games/northstar-cash",
    "G5": "https://www.mnlottery.com/games/draw-games/gopher-5",
    "PB": "https://www.mnlottery.com/games/draw-games/powerball",
}
HIST_PATH = lambda g: DATA / f"{g}_history.csv"

# Draw schedule windows (CST). We run once if current time within ±2 min of these.
SCHEDULE = {
    "N5": {"days": {0,1,2,3,4,5,6}, "post":"06:30", "pre":["09:00","11:00","14:00"], "final":"15:30"},
    "G5": {"days": {0,2,4},          "post":"06:30", "pre":["09:00","11:00","14:00"], "final":"15:30"},
    "PB": {"days": {0,2,5},          "post":"06:30", "pre":["09:00","11:00","14:00"], "final":"15:30"},
}

# -----------------------------
# GitHub persistence (optional)
# -----------------------------

GH = st.secrets.get("github", {})
GH_TOKEN = GH.get("token","")
GH_OWNER = GH.get("owner","")
GH_REPO  = GH.get("repo","")
GH_BRANCH= GH.get("branch","main")

def _gh_headers():
    return {"Authorization": f"token {GH_TOKEN}", "Accept":"application/vnd.github+json"} if GH_TOKEN else {}

def _gh_url(path:str):
    return f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path}"

def gh_write_text(path:str, text:str, message:str):
    if not GH_TOKEN or not GH_OWNER or not GH_REPO: return False
    try:
        # get current sha (if exists)
        r = requests.get(_gh_url(path), headers=_gh_headers(), params={"ref": GH_BRANCH}, timeout=15)
        sha = r.json().get("sha") if r.status_code==200 else None
        payload = {
            "message": message,
            "content": base64_encode(text),
            "branch": GH_BRANCH
        }
        if sha: payload["sha"] = sha
        r2 = requests.put(_gh_url(path), headers=_gh_headers(), json=payload, timeout=20)
        return r2.status_code in (200,201)
    except Exception:
        return False

def base64_encode(s: str) -> str:
    import base64
    return base64.b64encode(s.encode("utf-8")).decode("utf-8")

# -----------------------------
# Utilities
# -----------------------------

def now_ct():
    return datetime.now(TZ)

def parse_ints(tokens):
    out=[]
    for t in tokens:
        out += [int(x) for x in re.findall(r"\d+", str(t))]
    return out

def log_line(msg:str):
    ts = now_ct().strftime("%Y-%m-%d %H:%M:%S %Z")
    with open(LOGS / "app.log", "a") as f:
        f.write(f"[{ts}] {msg}\n")

def load_manifest():
    if MANIFEST.exists():
        try: return json.loads(MANIFEST.read_text())
        except: pass
    return {"last_update": None, "next_planned": None, "synced": False, "ver": APP_VER}

def save_manifest(m:dict):
    m["ver"] = APP_VER
    MANIFEST.write_text(json.dumps(m, indent=2))

# -----------------------------
# Cache: 30-minute pull cache
# -----------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_pull(url: str) -> str:
    r = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0 (Northstar)"})
    r.raise_for_status()
    return r.text

# -----------------------------
# Robust MN HTML parse
# -----------------------------

import os
import requests
import pandas as pd
import streamlit as st


    # Try GitHub, then proxy
    data = fetch_json(github_urls[game])
    if data is None:
        st.info(f"{game}: Retrying via Jina proxy…")
        data = fetch_json(proxy_urls[game])

    # If a raw list comes back, wrap it
    if isinstance(data, list):
        data = {"draws": data}

    # Fallback to local file if both remotes fail
    if data is None:
        st.error(f"{game}: ❌ All remote sources failed.")
        if os.path.exists(filename):
            st.warning(f"{game}: Using cached local file.")
            try:
                return pd.read_csv(filename)
            except Exception:
                return pd.DataFrame(columns=["date","n1","n2","n3","n4","n5","game"])
        return pd.DataFrame(columns=["date","n1","n2","n3","n4","n5","game"])

    # Normalize + save + return
    df = normalize_and_save_draws(game, data, filename)
    return df if df is not None else pd.DataFrame(columns=["date","n1","n2","n3","n4","n5","game"])

# --------------------------------------------------------------------
# Fetch JSON helper
# --------------------------------------------------------------------

def fetch_json(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

# --------------------------------------------------------------------
# Pull official MN data (GitHub → proxy → local fallback)
# --------------------------------------------------------------------
def pull_official(game):
    """
    Pulls MN Lottery results from GitHub (primary) or proxy (backup),
    then normalizes and persists to ./data/{game}_history.csv
    """
    github_urls = {
        "N5": "https://raw.githubusercontent.com/Minnesota-Lottery/history/main/northstar_cash.json",
        "G5": "https://raw.githubusercontent.com/Minnesota-Lottery/history/main/gopher5.json",
        "PB": "https://raw.githubusercontent.com/Minnesota-Lottery/history/main/powerball.json"
    }

    proxy_urls = {
        "N5": "https://r.jina.ai/https://raw.githubusercontent.com/Minnesota-Lottery/history/main/northstar_cash.json",
        "G5": "https://r.jina.ai/https://raw.githubusercontent.com/Minnesota-Lottery/history/main/gopher5.json",
        "PB": "https://r.jina.ai/https://raw.githubusercontent.com/Minnesota-Lottery/history/main/powerball.json"
    }

    if game not in github_urls:
        st.error(f"{game}: No source mapping found.")
        return pd.DataFrame()

    folder = "./data"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{game}_history.csv")

    # Try GitHub, then proxy
    data = fetch_json(github_urls[game])
    if data is None:
        st.info(f"{game}: Retrying via proxy...")
        data = fetch_json(proxy_urls[game])

    # If a raw list comes back, wrap it
    if isinstance(data, list):
        data = {"draws": data}

    # Fallback to local if both fail
    if data is None:
        st.error(f"{game}: ❌ All remote sources failed.")
        if os.path.exists(filename):
            st.warning(f"{game}: Using cached local file.")
            try:
                return pd.read_csv(filename)
            except Exception:
                return pd.DataFrame(columns=["date", "numbers"])
        return pd.DataFrame(columns=["date", "numbers"])

    # Normalize + save + return
    df = normalize_and_save_draws(game, data, filename)
    return df if df is not None else pd.DataFrame(columns=["date", "numbers"])

def normalize_and_save_draws(game, data, filename):
    try:
        # Normalize JSON into rows
        draw_rows = []
        items = data.get("draws", data)
        for draw in items:
            draw_date = draw.get("draw_date")
            numbers = draw.get("numbers") or draw.get("winning_numbers")
            if draw_date and numbers:
                if isinstance(numbers, list):
                    numbers = ",".join(map(str, numbers))
                draw_rows.append({"date": draw_date, "numbers": numbers})

        if not draw_rows:
            st.warning(f"{game}: No valid data found in JSON.")
            return None

        # ✅ Build new DataFrame and merge/save
        df_new = pd.DataFrame(draw_rows)
        df_new["game"] = game

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if os.path.exists(filename):
            old_df = pd.read_csv(filename)
            merged = pd.concat([old_df, df_new], ignore_index=True)
        else:
            merged = df_new

        merged.to_csv(filename, index=False)
        st.success(f"{game}: ✅ Pulled & saved successfully.")
        return merged

    except Exception as e:
        # ✅ Safety check and debug printout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        st.error(f"{game}: Failed to build or save data: {e}")
        st.warning(f"{game}: Check if data format or URL changed.")
        st.write("DEBUG — Incoming data sample:", data)
        return None
        
        df_new = pd.DataFrame(draw_rows)
        df_new["game"] = game

        # Ensure /data directory exists before saving
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Merge with any previous file
        if os.path.exists(filename):
            old_df = pd.read_csv(filename)
            merged = pd.concat([old_df, df_new]).drop_duplicates(subset=["date"], keep="last")
        else:
            merged = df_new

        merged.to_csv(filename, index=False)
        st.success(f"{game}: ✅ Pulled & saved {len(df_new)} draws successfully.")
        return merged

    except Exception as e:
        st.error(f"{game}: Failed to build or save history file. {e}")
        return None
    
# -----------------------------
# Persistence: histories / logs
# -----------------------------

def save_history(game:str, df_new:pd.DataFrame) -> pd.DataFrame:
    p = HIST_PATH(game)
    if p.exists():
        old = pd.read_csv(p)
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
    else:
        old = pd.DataFrame(columns=["date","n1","n2","n3","n4","n5","game"])
    merged = pd.concat([old, df_new], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date"])
    merged = merged.drop_duplicates(subset=["date","n1","n2","n3","n4","n5"]).sort_values("date", ascending=False)
    merged.to_csv(p, index=False)
    # Optional: push to GitHub
    if GH_TOKEN:
        gh_write_text(f"data/{p.name}", merged.to_csv(index=False), f"Update {game} history")
    return merged

def load_history(game:str) -> pd.DataFrame:
    p = HIST_PATH(game)
    if p.exists():
        df = pd.read_csv(p)
        df["date"]=pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"]).sort_values("date", ascending=False).reset_index(drop=True)
    return pd.DataFrame(columns=["date","n1","n2","n3","n4","n5","game"])

def log_confidence(game:str, conf:float):
    row = pd.DataFrame([[now_ct(), game, conf]], columns=["timestamp","game","confidence"])
    if CONF_PATH.exists():
        old = pd.read_csv(CONF_PATH)
        df = pd.concat([old, row], ignore_index=True)
        if len(df) > 500: df = df.tail(500)
    else:
        df = row
    df.to_csv(CONF_PATH, index=False)
    if GH_TOKEN:
        gh_write_text("data/confidence_trends.csv", df.to_csv(index=False), "Update confidence_trends")

def score_performance(game:str, predicted:List[int], actual:List[int]):
    aset = set(actual)
    exact = len(set(predicted) & aset)
    pm1 = sum(1 for p in predicted if any(abs(p-a)==1 for a in aset))
    row = pd.DataFrame([[now_ct(), game, exact, pm1]], columns=["timestamp","game","exact","plusminus1"])
    if PERF_PATH.exists():
        old = pd.read_csv(PERF_PATH)
        df = pd.concat([old, row], ignore_index=True)
        if len(df) > 2000: df = df.tail(2000)
    else:
        df = row
    df.to_csv(PERF_PATH, index=False)
    if GH_TOKEN:
        gh_write_text("data/performance_log.csv", df.to_csv(index=False), "Update performance_log")

# -----------------------------
# Weekly archive + manifest
# -----------------------------

    def weekly_archive_if_needed():
    # Make a zip each Sunday ~16:00 CST
    now = now_ct()
    if now.weekday()==6 and 16 <= now.hour < 17:  # Sunday hour window
        tag = now.strftime("%Y%m%d_%H%M")
        zpath = ARCH / f"northstar_archive_{tag}.zip"
        if not zpath.exists():
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
                for f in DATA.glob("*.csv"):
                    z.write(f, f.name)
            if GH_TOKEN:
                gh_write_text(f"data/archives/{zpath.name}", zpath.read_bytes().decode("latin1"), "Archive zip")  # store as bytes? (optional)

def update_manifest(synced=False):
    m = load_manifest()
    m["last_update"] = now_ct().isoformat()
    m["next_planned"] = (now_ct()+timedelta(minutes=30)).isoformat()
    m["synced"] = bool(synced)
    save_manifest(m)
    if GH_TOKEN:
        gh_write_text("data/manifest.json", json.dumps(m, indent=2), "Update manifest")

# -----------------------------
# Trickle-down seed
# -----------------------------

def build_trickle_seed(n5_hist:pd.DataFrame, window:int=20) -> Dict[int,float]:
    if n5_hist is None or n5_hist.empty:
        return {}
    cols = ["n1","n2","n3","n4","n5"]
    vals = np.concatenate(n5_hist.head(window)[cols].values)
    freq = pd.Series(vals).value_counts()
    seed={}
    for n,f in freq.items():
        n=int(n); seed[n]=float(f)
        for d in (-1,1):
            seed[n+d]=seed.get(n+d,0.0)+0.35
    return seed

# -----------------------------
# Adaptive simulation (with dynamic trickle weighting)
# -----------------------------

    def adaptive_simulation(df:pd.DataFrame, game:str, trickle:Dict[int,float]|None=None) -> (List[int], float):
    if df is None or df.empty: return [], 0.0
    cols=["n1","n2","n3","n4","n5"]
    recent = df.head(40).copy()
    vals = np.concatenate(recent[cols].values)
    freq = pd.Series(vals).value_counts()

    # cluster bonus (last 12 draws)
    bonus = pd.Series(0.0, index=freq.index)
    last12 = recent.head(12)
    for _, row in last12.iterrows():
        s=set(int(row[c]) for c in cols)
        for n in s:
            bonus.loc[n] = bonus.get(n,0.0) + (2.0 if len(s)>=3 else 1.0)

    # trickle factor (if provided)
    tr = pd.Series(0.0, index=freq.index)
    if trickle:
        for n,w in trickle.items():
            n=int(n)
            if n in tr.index: tr.loc[n]+=float(w)

    # volatility (variance of last 10 draws)
    last10 = recent.head(10)
    vol = last10[cols].std().mean() if not last10.empty else 1.0
    tighten = 0.85 if vol < 8.0 else 1.0

    w = (freq.astype(float)*1.0 + bonus*0.72 + tr*0.28) * tighten
    w = w[w>0].sort_values(ascending=False)
    if w.empty: return [], 0.0

    # Monte Carlo (random seed per run for variety)
    rng = np.random.default_rng(int(datetime.now().timestamp()))
    trials = 9000
    bucket={}
    base = w.copy()

    last = set(int(x) for x in recent.iloc[0][cols].tolist())
    base.loc[list(set(base.index) & last)] = base.loc[list(set(base.index) & last)] * 0.65

    for _ in range(trials):
        picks=set()
        local=base.copy()
        while len(picks)<5 and not local.empty:
            pick = rng.choice(local.index, p=(local/local.sum()).values)
            if pick not in picks:
                picks.add(int(pick))
                local = local.drop(index=pick, errors="ignore")
        if len(picks)==5:
            key=tuple(sorted(picks))
            bucket[key]=bucket.get(key,0)+1

    ranked = sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)
    top = list(ranked[0][0])
    conf = round(min(100.0, (ranked[0][1]/trials)*100*1.35), 2)
    return top, conf

# -----------------------------
# Schedule tick (safe, idempotent)
# -----------------------------

def within_window(hhmm:str, slack_min:int=2) -> bool:
    hh,mm = map(int, hhmm.split(":"))
    now = now_ct()
    t0 = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return abs((now - t0).total_seconds()) <= slack_min*60

def should_run(game:str, phase:str) -> bool:
    cfg = SCHEDULE[game]
    if now_ct().weekday() not in cfg["days"]: return False
    if phase=="post":  return within_window(cfg["post"])
    if phase=="final": return within_window(cfg["final"])
    if phase=="pre":   return any(within_window(t) for t in cfg["pre"])
    return False

def scheduler_tick(auto:bool=True):
    # run any due phases once per window (tracked in session_state)
    if "last_run_keys" not in st.session_state: st.session_state.last_run_keys=set()
    due=[]
    for g in GAMES:
        for phase in ("post","pre","final"):
            if should_run(g, phase):
                key=f"{now_ct().date()}_{g}_{phase}"
                if key not in st.session_state.last_run_keys:
                    due.append((g, phase, key))
    if not auto: return []

    for g, phase, key in sorted(due, key=lambda x: x[0]):
        run_phase(g, phase)
        st.session_state.last_run_keys.add(key)

    return due

# -----------------------------
# Single phase run (+ trickle)
# -----------------------------

def run_phase(game:str, phase:str):
    st.info(f"Running **{phase}** for **{game}** …")
    # Pull + save history
    df_new = pull_official(game, MN_SOURCES[game])
    if not df_new.empty:
        hist = save_history(game, df_new)
    else:
        hist = load_history(game)

    # Build trickle from N5
    seed = build_trickle_seed(load_history("N5"), window=20)
    trickle = seed if game in ("G5","PB") else None

    pick, conf = adaptive_simulation(hist, game, trickle)
    if pick:
        st.success(f"{game} → pick {pick}  |  confidence {conf}%")
        log_confidence(game, conf)

        # score vs latest actual if available
        if not hist.empty:
            latest = hist.iloc[0]
            actual = [int(latest[f"n{k}"]) for k in range(1,6)]
            score_performance(game, pick, actual)
    else:
        st.warning(f"{game}: not enough data.")

    weekly_archive_if_needed()
    update_manifest(synced=bool(GH_TOKEN))

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title=f"Northstar v{APP_VER}", page_icon="🌟", layout="wide")
st.title(f"🌟 Northstar Ecosystem — v{APP_VER}")
st.caption("Live official pulls • Auto storage • Trickle-down influence • Confidence & performance logging • CST schedules")

# Health widget
m = load_manifest()
colH1, colH2, colH3 = st.columns(3)
colH1.metric("Last update", (m.get("last_update") or "—"))
colH2.metric("Next planned", (m.get("next_planned") or "—"))
colH3.metric("Git sync", "✅ enabled" if GH_TOKEN else "—")

# Controls
with st.expander("Controls", expanded=True):
    auto_tick = st.toggle("Enable auto scheduler tick (recommended)", value=True)

    st.markdown("#### 🟢 Manual Control")
    if st.button("🚀 Run System Now"):
        _cached_pull.clear()  # force fresh data
        st.info("Running full Northstar update… please wait ⏳")
        seed = build_trickle_seed(load_history("N5"), window=20)
def load_previous_data(game):
    """
    Loads the most recent stored data file for the specified game (N5, G5, PB).
    Falls back to an empty DataFrame if no file is found.
    """
    import os
    import pandas as pd
    import streamlit as st

    folder = "./data"
    filename = f"{folder}/{game}_history.csv"

    try:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            st.info(f"{game}: Loaded previous data ({len(df)} rows).")
            return df
        else:
            st.warning(f"{game}: No previous file found — using empty DataFrame.")
            return pd.DataFrame(columns=["date", "n1", "n2", "n3", "n4", "n5", "game"])
    except Exception as e:
        st.error(f"{game}: Error loading previous data: {e}")
        return pd.DataFrame(columns=["date", "n1", "n2", "n3", "n4", "n5", "game"])
for g in GAMES:
    st.subheader(f"Running phase for {g}...")

    # Step 1: Pull official or fallback data
    df_new = pull_official(g)
    if df_new is None or df_new.empty:
        st.warning(f"{g}: no new data available — using fallback file.")
        df_new = load_previous_data(g)

    # Step 2: Store or merge history
    hist = save_history(g, df_new)
    if hist is None or hist.empty:
        st.error(f"{g}: failed to build or save history file.")
        continue

    # Step 3: Establish trickle-down seed for N5/G5 (cross-pull logic)
    trickle = build_trickle_seed(hist) if g in ["N5", "G5"] else None

    # Step 4: Run adaptive Monte Carlo + clustering
    pick, conf = adaptive_scheduler(hist, trickle)

    # Step 5: Update confidence tracking and logs
    if pick is not None:
        update_confidence_trends(hist, g)
        st.success(f"{g}: ✅ Updated confidence trends — {conf:.1f}% confidence.")
    else:
        st.warning(f"{g}: No viable pick returned from adaptive scheduler.")

        weekly_archive_if_needed()
        update_manifest(synced=bool(GH_TOKEN))
        st.success(f"✅ System run complete — {now_ct().strftime('%I:%M %p %Z')}")

# Auto tick on each refresh
due = scheduler_tick(auto=auto_tick)
if due:
    st.success(f"Ran: {', '.join([f'{g}:{p}' for g,p,_ in due])}")

# Live cards
st.subheader("🎯 Live latest draws")
cols = st.columns(3)
for i,g in enumerate(GAMES):
    with cols[i]:
        dfh = load_history(g)
        if not dfh.empty:
            latest = dfh.iloc[0]
            nums = [int(latest[f"n{k}"]) for k in range(1,6)]
            st.success(f"{g}: {' '.join(map(str, nums))}  —  {latest['date'].strftime('%b %d, %Y')}")
        else:
            st.warning(f"{g}: fallback (no file yet)")

st.divider()

# Predictions + logs
st.subheader("🔮 Adaptive predictions (with trickle-down)")
seed = build_trickle_seed(load_history("N5"), window=20)
for g in GAMES:
    hist = load_history(g)
    trickle = seed if g in ("G5","PB") else None
    pick, conf = adaptive_simulation(hist, g, trickle)
    if pick:
        st.info(f"{g}: {pick}  |  {conf}%")
        # log preview only (scheduler/phase already logs)
    else:
        st.caption(f"{g}: insufficient data yet.")

st.divider()

# Charts
st.subheader("📈 Confidence trend")
if CONF_PATH.exists():
    cdf = pd.read_csv(CONF_PATH)
    if not cdf.empty:
        cdf["timestamp"]=pd.to_datetime(cdf["timestamp"])
        cdf = cdf.sort_values("timestamp").tail(300)
        st.line_chart(cdf.pivot(index="timestamp", columns="game", values="confidence"))
    else:
        st.caption("No confidence data yet.")
else:
    st.caption("No confidence data yet.")

st.subheader("🏁 Performance (Exact / ±1)")
if PERF_PATH.exists():
    pdf = pd.read_csv(PERF_PATH)
    if not pdf.empty:
        pdf["timestamp"]=pd.to_datetime(pdf["timestamp"])
        st.dataframe(pdf.tail(50), use_container_width=True)
    else:
        st.caption("No performance entries yet.")
else:
    st.caption("No performance entries yet.")

st.divider()

# Schedule view
st.subheader("🗓️ Schedule (CST)")
for g,cfg in SCHEDULE.items():
    days_map = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    days_txt = ", ".join(days_map[d] for d in sorted(cfg["days"]))
    st.write(f"**{g}** — days: {days_txt} • post {cfg['post']} • pre {', '.join(cfg['pre'])} • final {cfg['final']}")

st.caption(f"Northstar v{APP_VER} • © 2025")
