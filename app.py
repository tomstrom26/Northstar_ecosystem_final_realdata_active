# ==========================================================
# 🌟 Northstar Ecosystem — v5.7-Unified (All-in-One)
# ==========================================================
# • Schedulers (CST): 5 AM pull (+ first-open fallback) • 7 AM post • 11 AM & 1 PM pre • 4 PM final
# • "Run Now" orchestrator: runs EVERYTHING for ALL games (ignores weekday) with progress + errors
# • Option A: Seed from /data/N5_source.zip (.xlsx/.csv) + continue daily MN pulls
# • Robust MN HTML parser (date + numbers; PB supports red if available)
# • Monte Carlo v2.7-M + trickle-down (N5→G5/PB)
# • Confidence & performance logs
# • GitHub: daily CSV uploads + weekly archive upload (optional email)
# • Tools: seed ZIP, test GH/email, view logs
# ==========================================================

import os, re, json, zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
from bs4 import BeautifulSoup

# ------------------ App config / paths ------------------
APP_VER = "5.7-Unified"
TZ      = pytz.timezone("America/Chicago")
ROOT    = Path(".")
DATA    = ROOT / "data"
LOGS    = DATA / "logs"
ARCH    = DATA / "archives"
for d in (DATA, LOGS, ARCH): d.mkdir(parents=True, exist_ok=True)

CONF_PATH = DATA / "confidence_trends.csv"
PERF_PATH = DATA / "performance_log.csv"
MANIFEST  = DATA / "manifest.json"
HIST      = lambda g: DATA / f"{g}_history.csv"
LAST      = lambda g: DATA / f"{g}_next.csv"
LOG_FILE  = LOGS / "app.log"

GAMES = ["N5","G5","PB"]

# Live sources (HTML pages parsed for latest draw)
MN_SRC = {
    "N5": "https://www.mnlottery.com/games/north-5",
    "G5": "https://www.mnlottery.com/games/gopher-5",
    "PB": "https://www.mnlottery.com/games/powerball",
}

# Draw calendars: Mon=0 ... Sun=6
DRAW_DAYS = {"N5": {0,1,2,3,4,5,6}, "G5": {0,2,4}, "PB": {0,2,5}}
HOURS     = {"daily":5, "post":7, "mid1":11, "mid2":13, "final":16}

# ------------------ GitHub + Email (Option A: Enabled) ------------------
# Add to Streamlit Secrets (App settings → Secrets) or .streamlit/secrets.toml:
# [github]
# token="ghp_xxx"; owner="YourGitHubUser"; repo="YourRepo"; branch="main"
# [email]
# smtp_host="smtp.gmail.com"; smtp_port=587; smtp_user="you@example.com"; smtp_pass="app-pass"
# to="you@example.com"; from_addr="Northstar <you@example.com>"
GH = st.secrets.get("github", {})
GH_TOKEN  = GH.get("token","")
GH_OWNER  = GH.get("owner","")
GH_REPO   = GH.get("repo","")
GH_BRANCH = GH.get("branch","main")
EM = st.secrets.get("email", {})

def _gh_headers():
    return {"Authorization": f"token {GH_TOKEN}", "Accept":"application/vnd.github+json"} if GH_TOKEN else {}
def _gh_url(path:str): return f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path}"
def _b64_bytes(b:bytes)->str:
    import base64; return base64.b64encode(b).decode("utf-8")

def gh_upload_bytes(repo_path:str, content_bytes:bytes, message:str)->bool:
    if not (GH_TOKEN and GH_OWNER and GH_REPO): return False
    try:
        r = requests.get(_gh_url(repo_path), headers=_gh_headers(), params={"ref": GH_BRANCH}, timeout=20)
        sha = r.json().get("sha") if r.status_code == 200 else None
        payload = {"message":message, "content":_b64_bytes(content_bytes), "branch":GH_BRANCH}
        if sha: payload["sha"] = sha
        put = requests.put(_gh_url(repo_path), headers=_gh_headers(), json=payload, timeout=30)
        return put.status_code in (200,201)
    except Exception as e:
        st.warning(f"GitHub upload failed: {e}")
        return False

def gh_upload_text(repo_path:str, text:str, message:str)->bool:
    return gh_upload_bytes(repo_path, text.encode("utf-8"), message)

def email_send(subject:str, body:str, attachment:Path|None=None)->bool:
    try:
        host=EM.get("smtp_host"); port=int(EM.get("smtp_port",587))
        user=EM.get("smtp_user"); pw=EM.get("smtp_pass")
        to=EM.get("to"); from_addr=EM.get("from_addr", user)
        if not all([host,port,user,pw,to,from_addr]): return False

        import smtplib
        from email.mime.base import MIMEBase
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email import encoders

        msg=MIMEMultipart()
        msg["From"]=from_addr; msg["To"]=to; msg["Subject"]=subject
        msg.attach(MIMEText(body,"plain"))
        if attachment and Path(attachment).exists():
            part=MIMEBase("application","octet-stream")
            with open(attachment,"rb") as f: part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{Path(attachment).name}"')
            msg.attach(part)

        with smtplib.SMTP(host,port) as s:
            s.starttls(); s.login(user,pw); s.sendmail(from_addr,[to], msg.as_string())
        return True
    except Exception as e:
        st.warning(f"Email send failed: {e}"); return False

# ------------------ helpers / logging ------------------
def now(): return datetime.now(TZ)

def log_line(msg:str):
    ts = now().strftime("%Y-%m-%d %H:%M:%S %Z")
    LOG_FILE.write_text(LOG_FILE.read_text()+f"[{ts}] {msg}\n") if LOG_FILE.exists() else LOG_FILE.write_text(f"[{ts}] {msg}\n")

def manifest():
    if MANIFEST.exists():
        try: return json.loads(MANIFEST.read_text())
        except: pass
    return {}
def save_manifest(m): m["ver"]=APP_VER; MANIFEST.write_text(json.dumps(m,indent=2))
def mark(key): m=manifest(); m[key]=now().strftime("%Y-%m-%d"); save_manifest(m)
def not_today(key): return manifest().get(key) != now().strftime("%Y-%m-%d")
def in_window(h): return h <= now().hour < (h+1)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch(url:str)->str:
    try: r=requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0 (Northstar)"})
    except Exception: return ""
    if r.status_code != 200: return ""
    return r.text

DATE_RE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}", re.I)
NUM_RE  = re.compile(r"\b\d{1,2}\b")

def parse_html_latest(game:str, html:str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Return (date_iso, numbers_string, red_int_or_None)
    PB: attempts to parse 6 numbers (5 whites + red). If only 5 found, red=None.
    """
    if not html: return None, None, None
    soup=BeautifulSoup(html,"html.parser")
    text=soup.get_text(" ", strip=True)

    m = DATE_RE.search(text)
    date_iso=None
    if m:
        try: date_iso = datetime.strptime(m.group(0), "%B %d, %Y").date().isoformat()
        except: pass

    nums = [int(x) for x in NUM_RE.findall(text)]
    if len(nums) < 5: return date_iso, None, None

    if game == "PB":
        if len(nums) >= 6:
            whites = nums[:5]
            red    = nums[5]
            return date_iso, ",".join(map(str, whites)), int(red)
        else:
            whites = nums[:5]
            return date_iso, ",".join(map(str, whites)), None
    else:
        return date_iso, ",".join(map(str, nums[:5])), None

def pull_latest(game:str) -> pd.DataFrame:
    """Fetch latest draw from MN and return as DF with columns: date,numbers,red,game."""
    html = fetch(MN_SRC[game])
    dt, numbers, red = parse_html_latest(game, html)
    if not numbers:
        return pd.DataFrame(columns=["date","numbers","red","game"])
    return pd.DataFrame([{
        "date": dt or now().date().isoformat(),
        "numbers": numbers,
        "red": red if game=="PB" else None,
        "game": game
    }])

def _upload_csv_daily(g:str, df:pd.DataFrame):
    if df is None or df.empty: return
    if GH_TOKEN and GH_OWNER and GH_REPO:
        ok = gh_upload_text(f"data/{g}_history.csv", df.to_csv(index=False), f"Daily update {g} {now().isoformat()}")
        if ok: st.toast(f"📤 Pushed {g}_history.csv to GitHub")

def save_hist(g:str, df_new:pd.DataFrame) -> pd.DataFrame:
    """Merge & save with strong de-dup (date,numbers,red)."""
    p = HIST(g)
    old = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["date","numbers","red","game"])
    merged = pd.concat([old, df_new], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["numbers"] = merged["numbers"].astype(str)
    if "red" not in merged.columns:
        merged["red"] = None
    # hash for de-dup
    merged["_h"] = merged["date"].astype(str) + "|" + merged["numbers"].astype(str) + "|" + merged["red"].astype(str)
    merged = merged.dropna(subset=["date"]).drop_duplicates("_h", keep="last").drop(columns=["_h"]).sort_values("date")
    merged.to_csv(p, index=False)
    log_line(f"save_hist {g}: rows={len(merged)}")
    _upload_csv_daily(g, merged)
    return merged

def load_hist(g:str) -> pd.DataFrame:
    p = HIST(g)
    if p.exists():
        df = pd.read_csv(p)
        if "red" not in df.columns: df["red"] = None
        df["date"]=pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"]).sort_values("date")
    return pd.DataFrame(columns=["date","numbers","red","game"])

# ------------------ seeding from /data/N5_source.zip ------------------
def _norm_from_grid(df:pd.DataFrame, game:str) -> pd.DataFrame:
    """Accepts grid: Date, N1..N5, (optional Red). Builds date,numbers,red,game."""
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    # find date column
    dcol = next((c for c in cols if "date" in c.lower()), cols[0])
    # candidate numeric columns excluding date
    rest = [c for c in cols if c != dcol]
    # try to identify red by name for PB
    red_col = None
    if game=="PB":
        for key in ("red","powerball","pb","r","power ball"):
            red_col = next((c for c in rest if key in c.lower()), None)
            if red_col: break
    # take first five numeric-like columns for whites
    whites_candidates = [c for c in rest if c != red_col]
    if len(whites_candidates) >= 5:
        whites_candidates = whites_candidates[:5]
    numbers = df[whites_candidates].astype(str).agg(",".join, axis=1)
    red = None
    if game=="PB":
        if red_col and red_col in df.columns:
            # try to coerce numeric, else None
            try:
                red_vals = pd.to_numeric(df[red_col], errors="coerce").astype("Int64")
            except:
                red_vals = pd.Series([None]*len(df))
            red = red_vals
        else:
            red = pd.Series([None]*len(df))

    out = pd.DataFrame({
        "date": pd.to_datetime(df[dcol], errors="coerce"),
        "numbers": numbers,
        "red": (red if isinstance(red, pd.Series) else pd.Series([None]*len(df))),
        "game": game
    })
    out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
    return out[["date","numbers","red","game"]]

def seed_zip():
    """Seed histories from /data/N5_source.zip (supports .xlsx/.csv inside the zip). Idempotent."""
    marker = DATA/".seeded"
    if marker.exists(): return
    src = DATA/"N5_source.zip"
    if not src.exists(): return
    try:
        with zipfile.ZipFile(src) as z: z.extractall(DATA/"_seed")
        tmp = DATA/"_seed"
        xlsx = list(tmp.rglob("*.xlsx"))
        csvs = list(tmp.rglob("*.csv"))

        def handle(path:Path, game:str):
            df = pd.read_excel(path) if path.suffix.lower()==".xlsx" else pd.read_csv(path)
            norm = _norm_from_grid(df, game)
            norm.to_csv(HIST(game), index=False)
            st.success(f"{game}: seeded {len(norm)} rows from {path.name}")
            log_line(f"seeded {game} from {path.name} rows={len(norm)}")

        seeded=False
        for f in xlsx+csvs:
            n=f.name.lower()
            if any(k in n for k in ["n5","north-5","northstar"]):
                handle(f,"N5"); seeded=True
            elif any(k in n for k in ["g5","gopher"]):
                handle(f,"G5"); seeded=True
            elif any(k in n for k in ["powerball","pb"]):
                handle(f,"PB"); seeded=True

        marker.write_text(now().isoformat())
        if not seeded:
            st.warning("ZIP extracted, but no usable XLSX/CSV detected. Export from Numbers to CSV/XLSX and re-zip.")
    except Exception as e:
        st.error(f"Seed failed: {e}")
        log_line(f"seed_zip error: {e}")

# ------------------ analytics ------------------
def _explode_whites(df:pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for s in df["numbers"].astype(str):
        vals=[int(x) for x in re.findall(r"\d+", s)][:5]
        if len(vals)==5: rows.append(vals)
    return pd.DataFrame(rows, columns=[f"n{i}" for i in range(1,6)]) if rows else pd.DataFrame()

def trickle_from_n5(n5_hist:pd.DataFrame, window:int=20) -> Dict[int,float]:
    if n5_hist is None or n5_hist.empty: return {}
    vals=[]
    for s in n5_hist.tail(window)["numbers"]:
        vals += [int(x) for x in re.findall(r"\d+", str(s))]
    ser=pd.Series(vals).value_counts(); w={}
    for n,f in ser.items():
        n=int(n); w[n]=float(f)
        for d in (-1,1): w[n+d]=w.get(n+d,0)+0.35
    return w

def _weighted_choice_k(rng, w:pd.Series, k:int=5) -> List[int]:
    picks=set(); local=w.copy()
    while len(picks)<k and not local.empty:
        p = rng.choice(local.index, p=(local/local.sum()).values)
        picks.add(int(p)); local = local.drop(p, errors="ignore")
    return sorted(picks)

def monte_whites(hist:pd.DataFrame, trickle:Optional[Dict[int,float]]=None) -> Tuple[List[int], float]:
    if hist.empty: return [], 0.0
    nd=_explode_whites(hist.tail(80))
    if nd.empty: return [], 0.0
    freq=pd.Series(nd.values.reshape(-1,)).value_counts().astype(float)

    # recency bonus (last 12 draws)
    bonus=pd.Series(0.0, index=freq.index)
    last12=_explode_whites(hist.tail(12))
    for _, r in last12.iterrows():
        for n in r.values:
            bonus.loc[int(n)] = bonus.get(int(n),0.0) + 1.0

    if trickle:
        for n,w in trickle.items():
            freq.loc[n] = freq.get(n,0.0) + float(w)*0.18

    w = (freq + bonus*0.65)
    w = w[w>0].sort_values(ascending=False)
    if w.empty: return [], 0.0

    rng=np.random.default_rng(int(datetime.now().timestamp()))
    trials=12000; bucket={}
    for _ in range(trials):
        picks=_weighted_choice_k(rng, w, k=5)
        if picks:
            t=tuple(picks); bucket[t]=bucket.get(t,0)+1
    top,cnt=max(bucket.items(), key=lambda kv: kv[1])
    conf=round(min(99, (cnt/trials)*130),2)
    return list(top), conf

def monte_red(hist:pd.DataFrame) -> Tuple[Optional[int], float]:
    """Powerball red model: empirical frequency over available red values; fallback uniform if missing."""
    if hist.empty or "red" not in hist.columns: return None, 0.0
    reds = hist["red"].dropna().astype(int)
    if reds.empty: return None, 0.0
    freq = reds.value_counts().astype(float)
    w = (freq / freq.sum())
    rng = np.random.default_rng(int(datetime.now().timestamp()))
    pick = int(rng.choice(w.index, p=w.values))
    conf = round(min(99, (w.loc[pick]*100)*1.2), 2)
    return pick, conf

def log_confidence(g:str, conf:float):
    row=pd.DataFrame([[now(), g, conf]], columns=["timestamp","game","confidence"])
    base=pd.read_csv(CONF_PATH) if CONF_PATH.exists() else pd.DataFrame(columns=row.columns)
    out=pd.concat([base,row], ignore_index=True)
    out.to_csv(CONF_PATH, index=False)
    # Optional GitHub push for logs (uncomment to enable)
    # gh_upload_text("data/confidence_trends.csv", out.to_csv(index=False), f"Confidence {now().isoformat()}")

def score_performance(g:str, predicted:List[int], actual:List[int]):
    aset=set(actual); exact=len(set(predicted)&aset)
    pm1=sum(1 for p in predicted if any(abs(p-a)==1 for a in aset))
    row=pd.DataFrame([[now(),g,exact,pm1]], columns=["timestamp","game","exact","plusminus1"])
    base=pd.read_csv(PERF_PATH) if PERF_PATH.exists() else pd.DataFrame(columns=row.columns)
    out=pd.concat([base,row], ignore_index=True)
    out.to_csv(PERF_PATH, index=False)
    # Optional GitHub push for logs (uncomment to enable)
    # gh_upload_text("data/performance_log.csv", out.to_csv(index=False), f"Performance {now().isoformat()}")

# ------------------ phase runners ------------------
def run_post(game:str):
    df = pull_latest(game); hist = save_hist(game, df)
    if hist is None or hist.empty: return
    # score against latest
    whites = [int(x) for x in re.findall(r"\d+", str(hist.iloc[-1]["numbers"]))][:5]
    pred, _ = monte_whites(hist)
    if pred: score_performance(game, pred, whites)

def run_pre(game:str):
    hist = load_hist(game)
    seed = trickle_from_n5(load_hist("N5"), 20)
    use_trickle = seed if game in ("G5","PB") else None
    whites, conf_w = monte_whites(hist, use_trickle)
    red, conf_r = (monte_red(hist) if game=="PB" else (None, 0.0))
    out = {
        "primary": " ".join(map(str, whites)) if whites else "",
        "red": (str(red) if red is not None else ""),
        "confidence": round(conf_w if game!="PB" else (0.7*conf_w + 0.3*conf_r), 2)
    }
    pd.DataFrame([out]).to_csv(LAST(game), index=False)
    log_confidence(game, out["confidence"])

# ------------------ archive (weekly) ------------------
def archive():
    n=now()
    if n.weekday()==6 and 16<=n.hour<17:
        tag=n.strftime("%Y%m%d_%H%M")
        zp=ARCH/f"northstar_{tag}.zip"
        if not zp.exists():
            with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as z:
                for f in DATA.glob("*.csv"): z.write(f, f.name)
        if GH_TOKEN and GH_OWNER and GH_REPO:
            ok = gh_upload_bytes(f"archives/{zp.name}", zp.read_bytes(), f"Archive {zp.name}")
            if ok: st.toast("📤 Weekly archive uploaded to GitHub")
        if EM:
            email_send(f"Northstar weekly archive {tag}", "Attached archive.", zp)

# ------------------ daily & window schedulers ------------------
def daily_pull():
    for g in GAMES:
        save_hist(g, pull_latest(g))

def scheduled_post_draws():
    for g in GAMES:
        if now().weekday() in DRAW_DAYS[g]:
            run_post(g)

def scheduled_pre_draws():
    for g in GAMES:
        if now().weekday() in DRAW_DAYS[g]:
            run_pre(g)

def sched():
    # Seed once (from /data/N5_source.zip)
    seed_zip()

    # 5 AM window OR first open fallback: ensure one daily pull per day
    if not_today("daily") and in_window(HOURS["daily"]):
        daily_pull(); mark("daily")
    if not_today("daily"):   # fallback: first open of the day
        daily_pull(); mark("daily")

    # 7 AM post
    if not_today("post") and in_window(HOURS["post"]):
        scheduled_post_draws(); mark("post")

    # 11 AM and 1 PM pre
    if not_today("mid1") and in_window(HOURS["mid1"]):
        scheduled_pre_draws(); mark("mid1")
    if not_today("mid2") and in_window(HOURS["mid2"]):
        scheduled_pre_draws(); mark("mid2")

    # 4 PM final (respect calendars)
    if not_today("final") and in_window(HOURS["final"]):
        for g in GAMES:
            if now().weekday() in DRAW_DAYS[g]:
                run_pre(g)
        mark("final")

    archive()

# ------------------ orchestrator (Run Now) ------------------
def run_full_cycle(respect_draw_calendar: bool = False) -> dict:
    """
    Run the entire pipeline NOW with progress + error surfacing.
    respect_draw_calendar:
      False => run every phase for ALL games regardless of weekday  (requested)
      True  => respect DRAW_DAYS
    """
    results = {"seeded": False, "pull": {}, "post": [], "pre_11": [], "pre_13": [], "final": [], "archive": None, "errors": []}

    # clear cache to force fresh pulls
    try:
        fetch.clear()
    except Exception as e:
        results["errors"].append(f"cache_clear: {e}")

    # 1) seed
    try:
        seed_zip(); results["seeded"] = True
    except Exception as e:
        results["errors"].append(f"seed_zip: {e}")

    # 2) pull
    try:
        for g in GAMES:
            df = pull_latest(g)
            hist = save_hist(g, df)
            results["pull"][g] = int(0 if hist is None else len(hist))
    except Exception as e:
        results["errors"].append(f"daily_pull: {e}")

    # 3) post
    try:
        for g in GAMES:
            if (not respect_draw_calendar) or (now().weekday() in DRAW_DAYS[g]):
                run_post(g); results["post"].append(g)
    except Exception as e:
        results["errors"].append(f"post_draw: {e}")

    # 4) pre windows (run twice)
    try:
        for g in GAMES:
            if (not respect_draw_calendar) or (now().weekday() in DRAW_DAYS[g]):
                run_pre(g); results["pre_11"].append(g)
        for g in GAMES:
            if (not respect_draw_calendar) or (now().weekday() in DRAW_DAYS[g]):
                run_pre(g); results["pre_13"].append(g)
    except Exception as e:
        results["errors"].append(f"pre_windows: {e}")

    # 5) final
    try:
        for g in GAMES:
            if (not respect_draw_calendar) or (now().weekday() in DRAW_DAYS[g]):
                run_pre(g); results["final"].append(g)
    except Exception as e:
        results["errors"].append(f"final_pred: {e}")

    # 6) archive (safe anytime)
    try:
        archive(); results["archive"] = "ok"
    except Exception as e:
        results["archive"] = "error"; results["errors"].append(f"archive: {e}")

    return results

# ------------------ UI ------------------
st.set_page_config(page_title=f"Northstar {APP_VER}", page_icon="🌟", layout="wide")
st.title(f"🌟 Northstar Ecosystem — {APP_VER}")
st.caption("Auto ZIP seed • Live MN pulls • Monte Carlo v2.7-M • 5/7/11/1/4 CST • Daily GitHub CSV + Weekly archive GitHub/email")

# kick scheduled tick on load
sched()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Status","Predictions","Confidence","History","Tools"])

with tab1:
    m=manifest()
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Daily (5 AM)", m.get("daily") or "—")
    c2.metric("Post (7 AM)",  m.get("post")  or "—")
    c3.metric("Pre 11 AM",    m.get("mid1")  or "—")
    c4.metric("Pre 1 PM",     m.get("mid2")  or "—")
    c5.metric("Final 4 PM",   m.get("final") or "—")

    st.subheader("Manual controls")
    b1,b2,b3,b4 = st.columns(4)
    if b1.button("🔄 Pull now"):
        with st.spinner("Pulling latest draws…"):
            daily_pull()
        st.success("Pulled & pushed CSVs (if GitHub is configured).")
    if b2.button("🧪 Post-draw now"):
        with st.spinner("Running post-draw analysis…"):
            scheduled_post_draws()
        st.success("Post-draw complete.")
    if b3.button("🔮 Pre-draw now"):
        with st.spinner("Running pre-draw predictions…"):
            scheduled_pre_draws()
        st.success("Pre-draw complete.")
    if b4.button("📦 Archive now"):
        with st.spinner("Archiving…"):
            archive()
        st.success("Archive step done (GitHub/email if configured).")

    st.divider()
    if st.button("🚀 Run Entire System Now (ignore weekday)", use_container_width=True):
        ph = st.empty()
        ph.info("Running: Seed → Pull → Post → Pre(11/1) → Final → Archive …")
        outcome = run_full_cycle(respect_draw_calendar=False)  # <- IGNORE weekday as requested
        if outcome.get("errors"):
            st.error("Some steps failed. See details below.")
        else:
            st.success(f"✅ Completed — {now().strftime('%I:%M %p %Z')}")
        st.json(outcome)

with tab2:
    st.subheader("Next Predictions")
    cols=st.columns(3)
    for i,g in enumerate(GAMES):
        with cols[i]:
            p = LAST(g)
            if p.exists(): st.write(f"**{g}**"); st.dataframe(pd.read_csv(p).tail(1), use_container_width=True)
            else: st.write(f"**{g}** — no prediction yet.")

with tab3:
    st.subheader("Confidence trend")
    if CONF_PATH.exists():
        df=pd.read_csv(CONF_PATH)
        if not df.empty:
            df["timestamp"]=pd.to_datetime(df["timestamp"], errors="coerce")
            df=df.dropna(subset=["timestamp"]).tail(300)
            st.line_chart(df.pivot(index="timestamp", columns="game", values="confidence"))
    st.subheader("Performance (tail)")
    if PERF_PATH.exists(): st.dataframe(pd.read_csv(PERF_PATH).tail(50), use_container_width=True)

with tab4:
    st.subheader("Latest histories")
    cols=st.columns(3)
    for i,g in enumerate(GAMES):
        with cols[i]:
            df=load_hist(g)
            st.write(f"**{g} ({len(df)} rows)**")
            st.dataframe(df.tail(20), use_container_width=True, height=320)

with tab5:
    st.write("**Data folder**"); st.code(str(DATA))
    c1,c2,c3,c4 = st.columns(4)
    if c1.button("📥 Seed ZIP Now"):
        seed_zip(); st.success("Seeding attempted. Check History tab.")
    if c2.button("🧪 Test GitHub upload"):
        ok = gh_upload_text("archives/dummy.txt", f"Northstar test {now().isoformat()}", "Test upload")
        st.success("GitHub OK") if ok else st.warning("GitHub not configured/failed.")
    if c3.button("✉️ Send test email"):
        ok = email_send("Northstar test email", "This is a test from Northstar.", None)
        st.success("Email OK") if ok else st.warning("Email not configured/failed.")
    if c4.button("🔎 Show latest log"):
        if LOG_FILE.exists():
            st.code("\n".join(LOG_FILE.read_text().splitlines()[-200:]))
        else:
            st.info("No log yet.")

st.caption(f"© Northstar — {APP_VER} • All times America/Chicago")
# --- DIAGNOSTICS / VERIFICATION ---------------------------------------------
diag = st.tabs(["Diagnostics"])[0]
with diag:
    st.subheader("Northstar Diagnostics")

    # 1) Secrets check
    st.markdown("### 1) Secrets loaded?")
    try:
        gh = st.secrets.get("github", {})
        em = st.secrets.get("email", {})
        ok_gh = all(bool(gh.get(k)) for k in ("token","owner","repo","branch"))
        ok_em = all(bool(em.get(k)) for k in ("smtp_host","smtp_port","smtp_user","smtp_pass","to"))
        masked = (gh.get("token","")[:6] + "…" + gh.get("token","")[-4:]) if gh.get("token") else ""
        c1,c2,c3 = st.columns(3)
        c1.metric("GitHub secrets", "OK" if ok_gh else "Missing")
        c2.metric("Email secrets", "OK" if ok_em else "Missing")
        c3.code(f"token={masked}")
        st.caption("If any show 'Missing', fix .streamlit/secrets.toml.")
    except Exception as e:
        st.error(f"Secrets access error: {e}")

    st.divider()

    # 2) GitHub upload test
    st.markdown("### 2) GitHub upload test")
    if st.button("📤 Upload diagnostics file to GitHub"):
        try:
            content = f"Northstar diagnostics {now().isoformat()} (OK)"
            ok = gh_upload_text("diagnostics/hello.txt", content, "Diagnostics test upload")
            st.success("GitHub upload OK ✔") if ok else st.warning("GitHub upload failed or not configured.")
        except Exception as e:
            st.error(f"GitHub test error: {e}")

    # 3) Email test
    st.markdown("### 3) Email test")
    if st.button("✉️ Send test email"):
        try:
            ok = email_send("Northstar diagnostics", "This is a test email from your app.", None)
            st.success("Email send OK ✔") if ok else st.warning("Email send failed or not configured.")
        except Exception as e:
            st.error(f"Email test error: {e}")

    st.divider()

    # 4) MN Lottery connectivity
    st.markdown("### 4) MN Lottery connectivity")
    colA, colB, colC = st.columns(3)
    def _check(url: str) -> str:
        try:
            html = fetch(url)
            return "OK" if (html and len(html) > 500) else "Thin/Blocked"
        except Exception:
            return "Error"
    colA.metric("N5 page", _check(MN_SRC["N5"]))
    colB.metric("G5 page", _check(MN_SRC["G5"]))
    colC.metric("PB page", _check(MN_SRC["PB"]))
    st.caption("If ‘Thin/Blocked’, your host may rate-limit MN Lottery. Seeding still works.")

    st.divider()

    # 5) Local data & seeding
    st.markdown("### 5) Local histories present?")
    cols = st.columns(3)
    for i,g in enumerate(GAMES):
        with cols[i]:
            df = load_hist(g)
            st.write(f"**{g}** rows:", len(df))
            if len(df)==0:
                st.warning("Empty. Use Tools → Seed ZIP Now.")
            else:
                st.dataframe(df.tail(5), use_container_width=True)

    # 6) Read/Write test
    st.markdown("### 6) Read/Write test")
    try:
        testp = DATA / "diagnostics_rw.txt"
        testp.write_text(f"rw-ok {now().isoformat()}")
        st.success(f"Write OK ✔  ({testp})")
        _ = testp.read_text()
        st.success("Read OK ✔")
    except Exception as e:
        st.error(f"Read/Write error: {e}")

    st.divider()

    # 7) Cache & clock
    st.markdown("### 7) Cache & clock")
    col1, col2 = st.columns(2)
    if col1.button("🧹 Clear network cache"):
        try:
            fetch.clear()
            st.success("Cache cleared ✔")
        except Exception as e:
            st.error(f"Cache clear error: {e}")
    col2.metric("Now (CST)", now().strftime("%Y-%m-%d %I:%M:%S %p %Z"))

    st.divider()

    # 8) Prediction smoke test (local only, no network)
    st.markdown("### 8) Prediction engine smoke test (local histories)")
    if st.button("🔮 Run local-only prediction smoke test"):
        try:
            out = {}
            for g in GAMES:
                hist = load_hist(g)
                if hist is None or hist.empty:
                    out[g] = {"error": "no history"}
                    continue
                seed = trickle_from_n5(load_hist("N5"), 20)
                use_tr = seed if g in ("G5","PB") else None
                whites, conf_w = monte_whites(hist, use_tr)
                red, conf_r = (monte_red(hist) if g=="PB" else (None, 0.0))
                out[g] = {
                    "whites": whites,
                    "red": red,
                    "confidence": round(conf_w if g!="PB" else (0.7*conf_w+0.3*conf_r), 2)
                }
            st.json(out)
            st.success("Engine OK ✔ (see results above)")
        except Exception as e:
            st.error(f"Smoke test error: {e}")

    st.divider()

    # 9) Full orchestrator (IGNORE weekday)
    st.markdown("### 9) Full ‘Run Now’ (all phases, all games, ignores weekday)")
    if st.button("🚀 Run Entire System NOW (Diagnostics)"):
        with st.spinner("Seed → Pull → Post → Pre(11/1) → Final → Archive"):
            try:
                outcome = run_full_cycle(respect_draw_calendar=False)
                if outcome.get("errors"):
                    st.error("Some steps failed. See JSON below.")
                else:
                    st.success(f"Completed — {now().strftime('%I:%M %p %Z')}")
                st.json(outcome)
            except Exception as e:
                st.error(f"Orchestrator error: {e}")
