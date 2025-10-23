# --- Google Drive Auto-Sync ---
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os

import datetime
import pytz
import streamlit as st

def show_health_widget():
    chicago = pytz.timezone("America/Chicago")
    now = datetime.datetime.now(chicago)
    next_events = [
        ("N5 Pre-Draw", "9:00 AM"),
        ("N5 Mid Analysis", "11:00 AM"),
        ("N5 Afternoon", "2:00 PM"),
        ("N5 Final", "3:30 PM"),
        ("G5/Powerball", "Mon/Wed/Fri, 6:30 AM"),
    ]
    st.markdown("### 🧭 Northstar Ecosystem Status")
    st.write(f"**Current Time:** {now.strftime('%Y-%m-%d %I:%M %p %Z')}")
    st.write(f"**Next Task:** {next_events[0][0]} → {next_events[0][1]}")
    st.write("**Confidence Sync:** Stable ✅")
    st.write("**Trickle Engine:** Active 🔁")
    st.markdown("---")

st.markdown("## 🗓️ Scheduling")
st.caption("Run pre-draw deep analysis several times per day and a final post-draw analysis after the draw.")

colA, colB = st.columns([1,1])
with colA:
    if st.button("▶️ Start Scheduler"):
        start_scheduler()
with colB:
    if st.button("⏹️ Stop Scheduler"):
        stop_scheduler()

# =======================
# Northstar Ecosystem — Adaptive Scheduler v3
# Single-file Streamlit app
# =======================

import os, json, io, random
from datetime import datetime
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="🌟 Northstar Ecosystem — Adaptive Scheduler v3", layout="wide")
APP_TITLE = "🌟 Northstar Ecosystem — Adaptive Scheduler v3"
TZ = pytz.timezone("America/Chicago")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Schedules (America/Chicago)
SCHEDULE = {
    "N5": {  # every day
        "days": "mon,tue,wed,thu,fri,sat,sun",
        "pre":  ["09:00","11:00","14:00"],
        "post": ["06:30","15:30"],  # morning recap + final post-draw
    },
    "G5": {  # Mon/Wed/Fri
        "days": "mon,wed,fri",
        "pre":  ["09:00","11:00","14:00"],
        "post": ["06:30","15:30"],
    },
    "PB": {  # Mon/Wed/Sat
        "days": "mon,wed,sat",
        "pre":  ["09:00","11:00","14:00"],
        "post": ["06:30","15:30"],
    },
}

# -----------------------
# STATE (persisted)
# -----------------------
STATE_PATH = os.path.join(DATA_DIR, "state.json")
DEFAULT_STATE = {
    "N5": {"confidence_trend": [], "variance_level": 1.0, "recency_weight": 1.0, "recent_hits": []},
    "G5": {"confidence_trend": [], "variance_level": 1.0, "recency_weight": 1.0, "recent_hits": []},
    "PB": {"confidence_trend": [], "variance_level": 1.0, "recency_weight": 1.0, "recent_hits": []},
    "_meta": {"last_updated": None, "last_influence_from": None}
}

def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_STATE))

def save_state(state):
    state["_meta"]["last_updated"] = datetime.utcnow().isoformat()
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

STATE = load_state()

# -----------------------
# FETCHERS (live internet pull)
# -----------------------
MN_URLS = {
    "N5": "https://www.mnlottery.com/games/northstar-cash",
    "G5": "https://www.mnlottery.com/games/gopher-5",
    "PB": "https://www.mnlottery.com/games/powerball",
}

def scrape_mn(url, game):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    nums = [n.get_text(strip=True) for n in soup.select(".winning-number")]
    date_el = soup.select_one(".draw-date") or soup.select_one(".date") or soup.find(attrs={"data-draw-date": True})
    draw_date = (date_el.get_text(strip=True) if date_el else "")
    return {"game": game, "date": draw_date, "numbers": nums}

def fallback_draw(game):
    return {"game": game, "date": "", "numbers": [], "error": "fallback_used"}

def latest_draw(game):
    try:
        return scrape_mn(MN_URLS[game], game)
    except Exception:
        return fallback_draw(game)

def latest_all():
    return {g: latest_draw(g) for g in ["N5","G5","PB"]}

# -----------------------
# DATA MERGE / MASTERS
# -----------------------
def master_path(game_key):
    return os.path.join(DATA_DIR, f"{game_key}_master.csv")

def append_merged(game_key, latest_row):
    """Append new draw to per-game CSV (dedup by date+numbers)."""
    if not latest_row or "numbers" not in latest_row:
        return None
    path = master_path(game_key)
    new = pd.DataFrame([latest_row])
    try:
        old = pd.read_csv(path)
        merged = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=["date","numbers"])
    except FileNotFoundError:
        merged = new
    merged.to_csv(path, index=False)
    return merged

# -----------------------
# UTIL: numbers parsing
# -----------------------
def numbers_to_ints(seq):
    out = []
    for s in seq:
        s = str(s)
        if ":" in s:  # e.g., "PB: 14"
            s = s.split(":")[-1]
        s = s.strip()
        try:
            out.append(int(s))
        except:
            pass
    return out

# -----------------------
# ANALYZER — Adaptive Monte Carlo (compact core)
# -----------------------
def run_adaptive_mc(game_key, state, latest_numbers, n_trials=3000):
    rng = np.random.default_rng()
    var = float(state[game_key]["variance_level"])
    rec = float(state[game_key]["recency_weight"])

    # crude ranges (tune as needed)
    pool_max = {"N5":31, "G5":47, "PB":69}[game_key]
    take = {"N5":5, "G5":5, "PB":5}[game_key]
    red_ball = (game_key == "PB")

    latest_ints = numbers_to_ints(latest_numbers or [])

    # base prior: ones + recency bumps around latest
    base = np.ones(pool_max)
    for n in latest_ints:
        for d in range(-2, 3):
            k = n + d
            if 1 <= k <= pool_max:
                base[k-1] += (3.0 if d == 0 else 1.0) * rec

    # apply variance as softening (var<1 tighten, >1 widen)
    base = np.power(base, 1.0 / max(0.5, min(1.8, var)))
    probs = base / base.sum()

    picks = []
    for _ in range(n_trials):
        draw = rng.choice(np.arange(1, pool_max+1), size=take, replace=False, p=probs)
        draw.sort()
        if red_ball:
            rb = rng.integers(1, 27)
            picks.append((tuple(draw.tolist()), rb))
        else:
            picks.append(tuple(draw.tolist()))

    c = Counter(picks)
    top = [x for x,_ in c.most_common(10)]

    if len(c) > 0:
        vals = list(c.values())
        first, second = vals[0], (vals[1] if len(vals)>1 else max(1, int(vals[0]*0.6)))
        conf = 0.6 + min(float(first)/max(second,1)/10.0, 0.4)
    else:
        conf = 0.6
    return {"candidates": top, "confidence": round(conf, 3)}

def score_vs_truth(game_key, prediction_numbers, truth_numbers):
    """Score exact and ±1 for the main white balls only."""
    truth = set(numbers_to_ints(truth_numbers))
    pred = set(numbers_to_ints(prediction_numbers))
    exact = len(pred & truth)
    pm1 = 0
    for p in pred:
        if (p-1 in truth) or (p+1 in truth):
            pm1 += 1
    return {"exact_hits": int(exact), "plusminus1": int(pm1)}

# -----------------------
# METRICS
# -----------------------
METRICS_PATH = os.path.join(DATA_DIR,"metrics.csv")
def log_metrics(game, phase, confidence=None, exact_hits=None, plusminus1=None):
    row = {"ts": datetime.now(TZ).isoformat(), "game":game, "phase":phase,
           "confidence":confidence, "exact_hits":exact_hits, "plusminus1":plusminus1}
    try:
        df = pd.read_csv(METRICS_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([row])
    df.to_csv(METRICS_PATH, index=False)

def load_metrics():
    try:
        return pd.read_csv(METRICS_PATH)
    except FileNotFoundError:
        return None

# -----------------------
# PDF TICKETS
# -----------------------
def generate_tickets_pdf(game_key, candidates):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    y = h - 72
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, f"Northstar Tickets — {game_key} — {datetime.now(TZ).strftime('%Y-%m-%d')}")
    y -= 24
    c.setFont("Helvetica", 12)
    for i, line in enumerate(candidates[:5], 1):
        if isinstance(line, tuple) and len(line)==2 and isinstance(line[1], int):
            nums, rb = line
            text = f"{i}. {' '.join(map(str,nums))}  |  PB: {rb}"
        else:
            text = f"{i}. {' '.join(map(str,line))}"
        c.drawString(72, y, text)
        y -= 18
        if y < 72:
            c.showPage()
            y = h - 72
    c.showPage()
    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf

# -----------------------
# CONFIDENCE CHART
# -----------------------
def plot_confidence(trend, title="Confidence"):
    fig = plt.figure()
    plt.plot(trend, marker="o")
    plt.title(title)
    plt.xlabel("Run")
    plt.ylabel("Confidence")
    out = io.BytesIO()
    fig.savefig(out, format="png", bbox_inches="tight")
    out.seek(0)
    return out

# -----------------------
# TRICKLE-DOWN
# -----------------------
def trickle_down(source_game, delta_conf):
    for other in ["N5","G5","PB"]:
        if other == source_game:
            continue
        STATE[other]["variance_level"] = float(STATE[other]["variance_level"]) + (0.10 * float(delta_conf))
        STATE[other]["variance_level"] = float(min(1.8, max(0.6, STATE[other]["variance_level"])))
        STATE[other]["recency_weight"]  = float(STATE[other]["recency_weight"]) + (0.05 * float(delta_conf))
        STATE[other]["recency_weight"]  = float(min(2.0, max(0.5, STATE[other]["recency_weight"])))
    STATE["_meta"]["last_influence_from"] = source_game
    save_state(STATE)

# -----------------------
# JOB HELPERS
# -----------------------
if "scheduler" not in st.session_state: st.session_state.scheduler = None
if "job_log" not in st.session_state: st.session_state.job_log = []
if "last_outputs" not in st.session_state: st.session_state.last_outputs = {}

def _log(msg):
    t = datetime.now(TZ).strftime("%H:%M:%S")
    st.session_state.job_log.append(f"[{t}] {msg}")
    st.session_state.job_log = st.session_state.job_log[-80:]

def _store(tag, out):
    st.session_state.last_outputs[tag] = out

# ---- Pre job
def run_pre(game_key, label="pre"):
    latest = latest_draw(game_key)
    mc = run_adaptive_mc(game_key, STATE, latest.get("numbers"))
    conf = mc["confidence"]

    # update adaptive "breathing"
    if conf >= 0.9:
        STATE[game_key]["variance_level"] = max(0.6, float(STATE[game_key]["variance_level"])*0.95)
    else:
        STATE[game_key]["variance_level"] = min(1.6, float(STATE[game_key]["variance_level"])*1.02)

    STATE[game_key]["confidence_trend"].append(conf)
    STATE[game_key]["confidence_trend"] = STATE[game_key]["confidence_trend"][-200:]
    save_state(STATE)

    _store(f"{game_key}_{label}", {"game":game_key, "phase":"pre", "candidates": mc["candidates"], "confidence": conf})
    log_metrics(game_key, "pre", confidence=conf)
    _log(f"✅ {game_key} pre ({label})")

# ---- Post job
def run_post(game_key, label="post"):
    latest = latest_draw(game_key)
    merged = append_merged(game_key, latest)

    # pick the latest pre result if present; else run a quick MC to build candidates
    pre_keys = [k for k in st.session_state.last_outputs.keys() if k.startswith(f"{game_key}_pre")]
    pre_keys.sort()
    candidates = None
    if pre_keys:
        candidates = st.session_state.last_outputs[pre_keys[-1]].get("candidates")
    if not candidates:
        quick = run_adaptive_mc(game_key, STATE, latest.get("numbers"), n_trials=1500)
        candidates = quick["candidates"]

    # score vs truth using first line
    first_line = candidates[0] if candidates else []
    numbers_for_scoring = list(first_line[0]) if (isinstance(first_line, tuple) and len(first_line)==2) else list(first_line)
    s = score_vs_truth(game_key, numbers_for_scoring, latest.get("numbers"))

    # state updates
    recent = STATE[game_key]["recent_hits"]
    recent.append(s["exact_hits"])
    STATE[game_key]["recent_hits"] = recent[-100:]

    # compute delta confidence (for trickle-down)
    trend = STATE[game_key]["confidence_trend"]
    delta = 0.0
    if len(trend) >= 2:
        delta = float(trend[-1]) - float(trend[-2])
    trickle_down(game_key, delta)

    save_state(STATE)
    log_metrics(game_key, "post", confidence=(trend[-1] if trend else None),
                exact_hits=s["exact_hits"], plusminus1=s["plusminus1"])

    # tickets pdf
    pdf_bytes = generate_tickets_pdf(game_key, candidates or [])
    _store(f"{game_key}_{label}", {
        "game": game_key,
        "phase": "post",
        "latest_numbers": latest.get("numbers"),
        "exact_hits": s["exact_hits"],
        "plusminus1": s["plusminus1"],
        "tickets_pdf": pdf_bytes,
        "rows_total": (len(merged) if isinstance(merged, pd.DataFrame) else None)
    })
    _log(f"✅ {game_key} post ({label})")

# -----------------------
# SCHEDULER
# -----------------------
def start_scheduler():
    # clean previous
    if st.session_state.scheduler:
        try:
            st.session_state.scheduler.remove_all_jobs()
            st.session_state.scheduler.shutdown(wait=False)
        except Exception:
            pass
        st.session_state.scheduler = None

    sched = BackgroundScheduler(timezone=str(TZ))

    for game_key, cfg in SCHEDULE.items():
        days = cfg["days"]
        # pre runs
        for hhmm in cfg["pre"]:
            hh, mm = map(int, hhmm.split(":"))
            sched.add_job(run_pre, CronTrigger(day_of_week=days, hour=hh, minute=mm),
                          args=(game_key, f"pre@{hhmm}"), id=f"{game_key}_pre_{hhmm}", replace_existing=True)
        # post runs
        for hhmm in cfg["post"]:
            hh, mm = map(int, hhmm.split(":"))
            sched.add_job(run_post, CronTrigger(day_of_week=days, hour=hh, minute=mm),
                          args=(game_key, f"post@{hhmm}"), id=f"{game_key}_post_{hhmm}", replace_existing=True)

    sched.start()
    st.session_state.scheduler = sched
    _log("🗓️ Scheduler started")

def stop_scheduler():
    if st.session_state.scheduler:
        try:
            st.session_state.scheduler.remove_all_jobs()
            st.session_state.scheduler.shutdown(wait=False)
            _log("⏹️ Scheduler stopped")
        except Exception as e:
            _log(f"⚠️ Stop failed: {e}")
        finally:
            st.session_state.scheduler = None

# Auto-start on boot (comment out if you want manual control)
if st.session_state.scheduler is None:
    start_scheduler()

# -----------------------
# UI
# -----------------------
st.markdown(f"## {APP_TITLE}")
st.caption("Adaptive MC + clustering • Trickle-down cross-influence • Live pulls • Scheduled pre/post runs")

# Live latest draw snapshot
st.markdown("### 🎯 Live latest draws")
latest = latest_all()
cols = st.columns(3)
for i, k in enumerate(["N5","G5","PB"]):
    with cols[i]:
        err = latest[k].get("error")
        if err:
            st.error(f"{k}: fallback ({err})")
        else:
            nums = latest[k].get("numbers") or []
            st.success(f"{k}: {latest[k].get('date','')} — {' '.join(map(str, nums))}")

# Schedule table
st.markdown("### 🗓️ Schedule (America/Chicago)")
for g, cfg in SCHEDULE.items():
    st.write(f"**{g}** — days: {cfg['days']} | pre: {', '.join(cfg['pre'])} | post: {', '.join(cfg['post'])}")

# Controls
colA, colB, colC, colD = st.columns(4)
with colA:
    if st.button("▶️ Start scheduler"):
        start_scheduler()
with colB:
    if st.button("⏹️ Stop scheduler"):
        stop_scheduler()
with colC:
    if st.button("⚡ Run ALL pre now"):
        for g in ["N5","G5","PB"]:
            run_pre(g, "pre@manual")
with colD:
    if st.button("🌙 Run ALL post now"):
        for g in ["N5","G5","PB"]:
            run_post(g, "post@manual")

# Confidence trend (main page)
st.markdown("### 📈 Confidence Trend")
tabs = st.tabs(["N5","G5","Powerball"])
for tab, key in zip(tabs, ["N5","G5","PB"]):
    with tab:
        trend = STATE[key]["confidence_trend"]
        if trend:
            buf = plot_confidence(trend, title=f"{key} Confidence")
            st.image(buf.getvalue())
            st.caption(f"variance={STATE[key]['variance_level']:.2f} • recency={STATE[key]['recency_weight']:.2f}")
        else:
            st.info("No trend yet — will populate as the scheduler runs.")

# Tickets (if any ready)
st.markdown("### 🎟️ Tickets")
has_ticket = False
for k, v in st.session_state.last_outputs.items():
    if isinstance(v, dict) and v.get("tickets_pdf"):
        st.download_button(
            f"Download tickets ({k})",
            data=v["tickets_pdf"],
            file_name=f"Northstar_{k}_tickets.pdf",
            mime="application/pdf"
        )
        has_ticket = True
if not has_ticket:
    st.caption("Tickets will appear here after the first post-draw run for each game.")

# Recent job log
st.markdown("### 🧾 Recent job log")
for line in st.session_state.job_log[-15:]:
    st.text(line)

# Metrics snapshot
st.markdown("### 📊 Rolling metrics")
m = load_metrics()
if m is not None and not m.empty:
    st.dataframe(m.tail(25), use_container_width=True)
else:
    st.caption("Metrics will accumulate after a few scheduled runs.")

# Show configured times
st.write("**Pre-draw times (local lottery TZ):**", ", ".join(SCHEDULE["pre_draw_times"]))
st.write("**Post-draw times (local lottery TZ):**", ", ".join(SCHEDULE["post_draw_times"]))
st.caption(f"Timezone: {LOTTERY_TZ}")

# Live log + last outputs
st.markdown("### 🧾 Recent job log")
for line in st.session_state.job_log[-10:]:
    st.text(line)

st.markdown("### 🧪 Latest results")
st.json(st.session_state.last_outputs)

# ---- SCHEDULER: Pre/Post draw jobs ----
import os, json, pytz
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Streamlit-safe singleton
if "scheduler" not in st.session_state:
    st.session_state.scheduler = None
if "job_log" not in st.session_state:
    st.session_state.job_log = []          # recent job messages
if "last_outputs" not in st.session_state:
    st.session_state.last_outputs = {}     # store last pre/post results in memory

# ---- CONFIGURABLE SCHEDULE ----
# Set your local lottery timezone here (Minnesota = America/Chicago)
LOTTERY_TZ = pytz.timezone("America/Chicago")

# Choose your daily schedule times (24-hour clock, local lottery TZ).
# You can change these without redeploying—just edit and save on GitHub, auto-redeploy will apply.
SCHEDULE = {
    "pre_draw_times": [ "09:00", "14:00", "18:30" ],  # multiple pre-draw deep runs
    "post_draw_times": [ "22:05" ]                    # final post-draw run
}

# ---- YOUR ANALYSIS HOOKS (plug in your real functions here) ----
def deep_pre_draw_analysis():
    """
    Run your heavy model(s) before the draw: update confidence, candidate sets,
    pair/cluster signals, etc. Return a summary dict for the UI.
    """
    # TODO: replace these placeholders with your real pipeline
    summary = {
        "timestamp": datetime.now(LOTTERY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "note": "Pre-draw deep analysis ran",
        "n5_top": [3, 8, 14, 21, 29],
        "g5_top": [4, 12, 19, 27, 31],
        "powerball_top": [5, 11, 24, 38, 62, "PB: 14"],
        "confidence": "0.81"
    }
    return summary

def deep_post_draw_analysis():
    """
    Immediately after draw: pull latest from the internet, evaluate hit/miss,
    update rolling stats, archive, and produce next-day seeds.
    """
    # TODO: call your live-scrape fetchers here, compare with latest predictions, compute hit score
    summary = {
        "timestamp": datetime.now(LOTTERY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "note": "Post-draw analysis ran",
        "hit_score": 2,          # example metric
        "exact_hits": 1,
        "plusminus1": 2,
        "confidence_next": "0.83"
    }
    return summary

# ---- INTERNAL: safe append to log & memory
def _log(msg):
    ts = datetime.now(LOTTERY_TZ).strftime("%H:%M:%S")
    st.session_state.job_log.append(f"[{ts}] {msg}")
    st.session_state.job_log = st.session_state.job_log[-50:]  # keep last 50 lines

def _run_pre():
    try:
        out = deep_pre_draw_analysis()
        st.session_state.last_outputs["pre"] = out
        _log("✅ Pre-draw deep analysis completed")
    except Exception as e:
        _log(f"❌ Pre-draw job failed: {e}")

def _run_post():
    try:
        out = deep_post_draw_analysis()
        st.session_state.last_outputs["post"] = out
        _log("✅ Post-draw analysis completed")
    except Exception as e:
        _log(f"❌ Post-draw job failed: {e}")

# ---- START/RESTART SCHEDULER
def start_scheduler():
    # Stop an existing one (if any)
    if st.session_state.scheduler:
        try:
            st.session_state.scheduler.remove_all_jobs()
            st.session_state.scheduler.shutdown(wait=False)
        except Exception:
            pass
        st.session_state.scheduler = None

    sched = BackgroundScheduler(timezone=str(LOTTERY_TZ))
    # Pre-draw jobs
    for hhmm in SCHEDULE["pre_draw_times"]:
        hh, mm = hhmm.split(":")
        sched.add_job(_run_pre, CronTrigger(hour=int(hh), minute=int(mm)))
    # Post-draw jobs
    for hhmm in SCHEDULE["post_draw_times"]:
        hh, mm = hhmm.split(":")
        sched.add_job(_run_post, CronTrigger(hour=int(hh), minute=int(mm)))

    sched.start()
    st.session_state.scheduler = sched
    _log("🗓️ Scheduler started")

def stop_scheduler():
    if st.session_state.scheduler:
        try:
            st.session_state.scheduler.remove_all_jobs()
            st.session_state.scheduler.shutdown(wait=False)
            _log("⏹️ Scheduler stopped")
        except Exception as e:
            _log(f"⚠️ Could not stop scheduler cleanly: {e}")
        finally:
            st.session_state.scheduler = None

def sync_from_drive(folder_name="Northstar_Data"):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    file_list = drive.ListFile({'q': f"'{folder_name}' in parents and trashed=false"}).GetList()

    os.makedirs("data", exist_ok=True)
    for file in file_list:
        if file['title'].endswith(".csv"):
            file.GetContentFile(os.path.join("data", file['title']))
            print(f"✅ Synced {file['title']} from Drive")
    return "data"

# Optional save / append
def append_to_master(latest_draws, master_csv="master_draws.csv"):
    df = pd.DataFrame(latest_draws)
    try:
        old = pd.read_csv(master_csv)
        df = pd.concat([old, df]).drop_duplicates(subset=["game", "date"])
    except FileNotFoundError:
        pass
    df.to_csv(master_csv, index=False)
    return df

# Uncomment to enable automatic updates
# master_df = append_to_master([d for d in latest_draws if d])

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Northstar System", layout="wide")

# --- HEADER ---
st.title("🌟 Northstar — All-in-One System")
st.caption("Live engine: offline + Drive-ready build")

# Sync new drawings before analysis
st.info("🔄 Checking Google Drive for latest draws...")
try:
    data_path = sync_from_drive()
    st.success("✅ Latest draw files synced from Drive.")
except Exception as e:
    st.warning(f"⚠️ Drive sync skipped: {e}")

# --- FILE UPLOAD ---
st.subheader("📤 Upload your latest datasets")
uploaded_files = st.file_uploader(
    "Upload N5, G5, and Powerball CSV files here",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        st.success(f"✅ Loaded {file.name}")
        df = pd.read_csv(file)
        st.write(df.head())

# --- RUN BUTTON ---
st.subheader("⚙️ Run Northstar Analysis")
if st.button("▶️ Run Simulation"):
    st.info("Running analysis...")
    # placeholder for your future Northstar logic
    st.success("✅ Analysis complete! Results displayed below.")

# --- FOOTER ---
st.markdown("---")
st.caption("Northstar Ecosystem © 2025 • Powered by Streamlit")
