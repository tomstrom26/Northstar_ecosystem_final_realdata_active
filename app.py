# =======================
# Northstar Ecosystem — Adaptive Scheduler v3 (Trickle-Down + GitHub persistence)
# =======================

import os, io, json, base64
from datetime import datetime
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# -----------------------
# PAGE / GLOBALS
# -----------------------
st.set_page_config(page_title="🌟 Northstar Ecosystem — Adaptive Scheduler v3", page_icon="🌟", layout="wide")
APP_TITLE = "🌟 Northstar Ecosystem — Adaptive Scheduler v3"
TZ = pytz.timezone("America/Chicago")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# GitHub persistence (Option A)
GITHUB = st.secrets.get("github", {})
GH_TOKEN = GITHUB.get("token", "")
GH_OWNER = GITHUB.get("owner", "tomstrom26")
GH_REPO  = GITHUB.get("repo", "northstar_ecosystem_final_realdata_active")
GH_BRANCH = GITHUB.get("branch", "main")

def _gh_headers():
    return {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github+json"}

def _gh_contents_url(path):
    return f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path}"

def gh_read_text(path, default=""):
    if not GH_TOKEN:
        return default
    resp = requests.get(_gh_contents_url(path), headers=_gh_headers(), params={"ref": GH_BRANCH})
    if resp.status_code == 200:
        j = resp.json()
        content = base64.b64decode(j["content"]).decode("utf-8")
        return content
    return default

def gh_write_text(path, text, message="Update file via app", sha=None):
    if not GH_TOKEN:
        return False
    url = _gh_contents_url(path)
    # get current sha if not supplied
    if sha is None:
        r = requests.get(url, headers=_gh_headers(), params={"ref": GH_BRANCH})
        sha = r.json().get("sha") if r.status_code==200 else None
    payload = {
        "message": message,
        "content": base64.b64encode(text.encode("utf-8")).decode("utf-8"),
        "branch": GH_BRANCH
    }
    if sha: payload["sha"] = sha
    r = requests.put(url, headers=_gh_headers(), json=payload)
    return r.status_code in (200,201)

# -----------------------
# SCHEDULE (America/Chicago)
# -----------------------
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
# STATE (persisted to /data + GitHub if token present)
# -----------------------
STATE_PATH = os.path.join(DATA_DIR, "state.json")
DEFAULT_STATE = {
    "N5": {"confidence_trend": [], "variance_level": 1.0, "recency_weight": 1.0, "recent_hits": []},
    "G5": {"confidence_trend": [], "variance_level": 1.0, "recency_weight": 1.0, "recent_hits": []},
    "PB": {"confidence_trend": [], "variance_level": 1.0, "recency_weight": 1.0, "recent_hits": []},
    "_meta": {"last_updated": None, "last_influence_from": None}
}

def load_state():
    # Try GitHub first
    if GH_TOKEN:
        txt = gh_read_text("data/state.json", "")
        if txt:
            try:
                s = json.loads(txt)
                return s
            except Exception:
                pass
    # Fallback local
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_STATE))

def save_state(state):
    state["_meta"]["last_updated"] = datetime.utcnow().isoformat()
    # local
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    # GitHub
    if GH_TOKEN:
        gh_write_text("data/state.json", json.dumps(state, indent=2), "Update state.json via app")

STATE = load_state()

# -----------------------
# METRICS
# -----------------------
METRICS_PATH = os.path.join(DATA_DIR,"metrics.csv")

def load_metrics():
    # GitHub first
    if GH_TOKEN:
        txt = gh_read_text("data/metrics.csv", "")
        if txt:
            try:
                from io import StringIO
                return pd.read_csv(StringIO(txt))
            except Exception:
                pass
    # local
    try:
        return pd.read_csv(METRICS_PATH)
    except FileNotFoundError:
        return pd.DataFrame(columns=["ts","game","phase","confidence","exact_hits","plusminus1"])

def save_metrics(df: pd.DataFrame):
    df.to_csv(METRICS_PATH, index=False)
    if GH_TOKEN:
        gh_write_text("data/metrics.csv", df.to_csv(index=False), "Update metrics.csv via app")

METRICS = load_metrics()

def log_metrics(game, phase, confidence=None, exact_hits=None, plusminus1=None):
    global METRICS
    row = {"ts": datetime.now(TZ).isoformat(), "game":game, "phase":phase,
           "confidence":confidence, "exact_hits":exact_hits, "plusminus1":plusminus1}
    METRICS = pd.concat([METRICS, pd.DataFrame([row])], ignore_index=True)
    save_metrics(METRICS)

# -----------------------
# FETCHERS (live internet pull + fallback)
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
# MASTER CSV per game (persist to GitHub if token)
# -----------------------
def _master_path(game_key): return os.path.join(DATA_DIR, f"{game_key}_master.csv")

def _read_master(game_key):
    path = _master_path(game_key)
    # GitHub first
    if GH_TOKEN:
        txt = gh_read_text(f"data/{game_key}_master.csv", "")
        if txt:
            from io import StringIO
            try: return pd.read_csv(StringIO(txt))
            except Exception: pass
    # local
    if os.path.exists(path):
        try: return pd.read_csv(path)
        except Exception: pass
    return pd.DataFrame(columns=["game","date","numbers"])

def _save_master(game_key, df: pd.DataFrame):
    path = _master_path(game_key)
    df.to_csv(path, index=False)
    if GH_TOKEN:
        gh_write_text(f"data/{game_key}_master.csv", df.to_csv(index=False), f"Update {game_key}_master.csv via app")

def append_merged(game_key, latest_row):
    if not latest_row or "numbers" not in latest_row: return None
    df_old = _read_master(game_key)
    new = pd.DataFrame([latest_row])
    if df_old is not None and new is not None and not df_old.empty and not new.empty:
        merged = pd.concat([df_old, new], ignore_index=True).drop_duplicates(subset=["draw_date"], keep="last")
    elif new is not None and not new.empty:
        merged = new
    else:
        merged = df_old
except Exception as e:
    st.warning(f"⚠️ Data merge skipped due to format issue: {e}")
    merged = df_old if 'df_old' in locals() else new
    _save_master(game_key, merged)
    return merged

# -----------------------
# UTIL
# -----------------------
def numbers_to_ints(seq):
    out = []
    for s in seq or []:
        s = str(s)
        if ":" in s: s = s.split(":")[-1]
        s = s.strip()
        try: out.append(int(s))
        except: pass
    return out

# -----------------------
# ANALYZER — Adaptive Monte Carlo + simple cluster bias
# -----------------------
def run_adaptive_mc(game_key, state, latest_numbers, n_trials=3000):
    rng = np.random.default_rng()
    var = float(state[game_key]["variance_level"])
    rec = float(state[game_key]["recency_weight"])

    pool_max = {"N5":31, "G5":47, "PB":69}[game_key]
    take = {"N5":5, "G5":5, "PB":5}[game_key]
    red_ball = (game_key == "PB")

    latest_ints = numbers_to_ints(latest_numbers)

    base = np.ones(pool_max)
    # recency bumps: ±2 around most recent numbers
    for n in latest_ints:
        for d in range(-2, 3):
            k = n + d
            if 1 <= k <= pool_max:
                base[k-1] += (3.0 if d == 0 else 1.0) * rec

    # variance breathing (var < 1 tighten; >1 widen)
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
    truth = set(numbers_to_ints(truth_numbers))
    pred = set(numbers_to_ints(prediction_numbers))
    exact = len(pred & truth)
    pm1 = sum(1 for p in pred if (p-1 in truth) or (p+1 in truth))
    return {"exact_hits": int(exact), "plusminus1": int(pm1)}

# -----------------------
# TRICKLE-DOWN (cross-influence after post)
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
# PDF tickets
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
# JOB RUNNERS
# -----------------------
if "scheduler" not in st.session_state: st.session_state.scheduler = None
if "job_log" not in st.session_state: st.session_state.job_log = []
if "last_outputs" not in st.session_state: st.session_state.last_outputs = {}

def _log(msg):
    t = datetime.now(TZ).strftime("%H:%M:%S")
    st.session_state.job_log.append(f"[{t}] {msg}")
    st.session_state.job_log = st.session_state.job_log[-120:]

def _store(tag, out): st.session_state.last_outputs[tag] = out

def run_pre(game_key, label="pre"):
    latest = latest_draw(game_key)
    mc = run_adaptive_mc(game_key, STATE, latest.get("numbers"))
    conf = mc["confidence"]

    # adaptive breathing
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

def run_post(game_key, label="post"):
    latest = latest_draw(game_key)
    merged = append_merged(game_key, latest)

    # pick candidates from most recent pre; else quick run
    pre_keys = [k for k in st.session_state.last_outputs.keys() if k.startswith(f"{game_key}_pre")]
    pre_keys.sort()
    candidates = None
    if pre_keys:
        candidates = st.session_state.last_outputs[pre_keys[-1]].get("candidates")
    if not candidates:
        quick = run_adaptive_mc(game_key, STATE, latest.get("numbers"), n_trials=1500)
        candidates = quick["candidates"]

    # score vs truth
    first_line = candidates[0] if candidates else []
    nums_for_score = list(first_line[0]) if (isinstance(first_line, tuple) and len(first_line)==2) else list(first_line)
    s = score_vs_truth(game_key, nums_for_score, latest.get("numbers"))

    # update state + trickle
    recent = STATE[game_key]["recent_hits"]; recent.append(s["exact_hits"])
    STATE[game_key]["recent_hits"] = recent[-100:]
    trend = STATE[game_key]["confidence_trend"]; delta = 0.0
    if len(trend)>=2: delta = float(trend[-1]) - float(trend[-2])
    trickle_down(game_key, delta)

    save_state(STATE)
    log_metrics(game_key, "post", confidence=(trend[-1] if trend else None),
                exact_hits=s["exact_hits"], plusminus1=s["plusminus1"])

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
        for hhmm in cfg["pre"]:
            hh, mm = map(int, hhmm.split(":"))
            sched.add_job(run_pre, CronTrigger(day_of_week=days, hour=hh, minute=mm),
                          args=(game_key, f"pre@{hhmm}"), id=f"{game_key}_pre_{hhmm}", replace_existing=True)
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

# Auto-start
if st.session_state.scheduler is None:
    start_scheduler()

# -----------------------
# UI — HEALTH STRIP
# -----------------------
def show_health_widget():
    now = datetime.now(TZ)
    st.markdown("### 🧭 Northstar Ecosystem Status")
    st.write(f"**Current Time:** {now.strftime('%Y-%m-%d %I:%M %p %Z')}")
    st.write("**Confidence Sync:** Stable ✅")
    st.write("**Trickle Engine:** Active 🔁")
    st.markdown("---")

def get_wake_status():
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/actions/runs?per_page=1"
        )
        if resp.status_code == 200:
            data = resp.json()
            if data["workflow_runs"]:
                last = data["workflow_runs"][0]
                status = last["conclusion"]
                ts = datetime.strptime(last["created_at"], "%Y-%m-%dT%H:%M:%SZ")
                ts = pytz.utc.localize(ts).astimezone(TZ)
                if status == "success":
                    st.markdown(f"🟢 **Last Wake Ping:** {ts.strftime('%I:%M %p')} — OK")
                else:
                    st.markdown(f"🔴 **Last Wake Ping:** {ts.strftime('%I:%M %p')} — Failed")
            else:
                st.caption("No wake runs found yet.")
        else:
            st.caption("Could not fetch wake status.")
    except Exception as e:
        st.caption(f"Wake status error: {e}")

# -----------------------
# PAGE LAYOUT
# -----------------------
st.markdown(f"## {APP_TITLE}")
show_health_widget()
get_wake_status()

# Live latest snapshot
st.markdown("### 🎯 Live latest draws")
_latest = latest_all()
cols = st.columns(3)
for i, k in enumerate(["N5","G5","PB"]):
    with cols[i]:
        err = _latest[k].get("error")
        if err:
            st.error(f"{k}: fallback ({err})")
        else:
            nums = _latest[k].get("numbers") or []
            st.success(f"{k}: {_latest[k].get('date','')} — {' '.join(map(str, nums))}")

# Schedule table
st.markdown("### 🗓️ Schedule (America/Chicago)")
for g, cfg in SCHEDULE.items():
    st.write(f"**{g}** — days: {cfg['days']} | pre: {', '.join(cfg['pre'])} | post: {', '.join(cfg['post'])}")

# Controls
colA, colB, colC, colD = st.columns(4)
with colA:
    if st.button("▶️ Start scheduler"): start_scheduler()
with colB:
    if st.button("⏹️ Stop scheduler"):  stop_scheduler()
with colC:
    if st.button("⚡ Run ALL pre now"):
        for g in ["N5","G5","PB"]: run_pre(g, "pre@manual")
with colD:
    if st.button("🌙 Run ALL post now"):
        for g in ["N5","G5","PB"]: run_post(g, "post@manual")

# Confidence trend
st.markdown("### 📈 Confidence Trend")
tabs = st.tabs(["N5","G5","Powerball"])
for tab, key in zip(tabs, ["N5","G5","PB"]):
    with tab:
        trend = STATE[key]["confidence_trend"]
        if trend:
            fig = plt.figure()
            plt.plot(trend, marker="o")
            plt.title(f"{key} Confidence")
            plt.xlabel("Run")
            plt.ylabel("Confidence")
            st.pyplot(fig)
            st.caption(f"variance={STATE[key]['variance_level']:.2f} • recency={STATE[key]['recency_weight']:.2f}")
        else:
            st.info("No trend yet — will populate as the scheduler runs.")

# Tickets (if available)
st.markdown("### 🎟️ Tickets")
shown = False
for k, v in st.session_state.last_outputs.items():
    if isinstance(v, dict) and v.get("tickets_pdf"):
        st.download_button(
            f"Download tickets ({k})",
            data=v["tickets_pdf"],
            file_name=f"Northstar_{k}_tickets.pdf",
            mime="application/pdf"
        )
        shown = True
if not shown:
    st.caption("Tickets will appear here after the first post-draw run for each game.")

# Recent job log
st.markdown("### 🧾 Recent job log")
for line in st.session_state.job_log[-18:]:
    st.text(line)

# Metrics snapshot
st.markdown("### 📊 Rolling metrics")
m = METRICS
if m is not None and not m.empty:
    st.dataframe(m.tail(25), use_container_width=True)
else:
    st.caption("Metrics will accumulate after a few scheduled runs.")
