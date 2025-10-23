# --- Google Drive Auto-Sync ---
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os

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
