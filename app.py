# ==========================================================
# 🌟 Northstar Ecosystem — v5.5-Sync (Scheduler + Daily GitHub)
# ==========================================================
# • Schedulers (CST): 5 AM pull • 7 AM post • 11 AM & 1 PM pre • 4 PM final
# • Monte Carlo v2.7-M + trickle-down (N5→G5/PB)
# • Auto-seed from /data/N5_source.zip (.xlsx/.csv; PB = five mains only)
# • Confidence & performance logs
# • Weekly archive (Sun 4–5 PM) → GitHub upload + optional email
# • NEW: Daily CSV uploads to GitHub after saves (N5/G5/PB)
# • One-click: “🚀 Run Entire System Now”
# ==========================================================

import os, re, json, zipfile, base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import numpy as np, pandas as pd, pytz, requests, streamlit as st
from bs4 import BeautifulSoup

APP_VER = "5.5-Sync"
TZ = pytz.timezone("America/Chicago")
ROOT = Path(".")
DATA = ROOT / "data"
LOGS = DATA / "logs"
ARCH = DATA / "archives"
for d in (DATA, LOGS, ARCH): d.mkdir(parents=True, exist_ok=True)

CONF_PATH = DATA / "confidence_trends.csv"
PERF_PATH = DATA / "performance_log.csv"
MANIFEST  = DATA / "manifest.json"
HIST      = lambda g: DATA / f"{g}_history.csv"
LAST      = lambda g: DATA / f"{g}_next.csv"

GAMES = ["N5","G5","PB"]

MN_SRC = {
    "N5": "https://www.mnlottery.com/games/north-5",
    "G5": "https://www.mnlottery.com/games/gopher-5",
    "PB": "https://www.mnlottery.com/games/powerball",
}

# Draw calendars: Mon=0 ... Sun=6
DRAW_DAYS = {"N5":{0,1,2,3,4,5,6}, "G5":{0,2,4}, "PB":{0,2,5}}
HOURS     = {"daily":5, "post":7, "mid1":11, "mid2":13, "final":16}

# ---------- GitHub + Email (secrets) ----------
# Add these in .streamlit/secrets.toml or Streamlit Cloud app secrets:
# [github]
# token="ghp_xxx" ; owner="YourUser"; repo="YourRepo"; branch="main"
# [email]
# smtp_host="smtp.gmail.com"; smtp_port=587; smtp_user="you@x.com"; smtp_pass="app-pass"
# to="you@x.com"; from_addr="Northstar <you@x.com>"
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
    """Create/update file by bytes via GitHub contents API."""
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

# ---------- helpers ----------
def now(): return datetime.now(TZ)
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
    try: r=requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0 (Northstar)"}); r.raise_for_status(); return r.text
    except: return ""

DATE_RE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}", re.I)
NUM_RE  = re.compile(r"\b\d{1,2}\b")

def parse_html(game, html):
    soup=BeautifulSoup(html,"html.parser"); text=soup.get_text(" ", strip=True)
    d=DATE_RE.search(text); dt=None
    if d:
        try: dt=datetime.strptime(d.group(0), "%B %d, %Y").date().isoformat()
        except: pass
    nums=NUM_RE.findall(text)
    if len(nums)<5: return None, None
    take = 6 if (game=="PB" and len(nums)>=6) else 5
    return dt, ",".join(nums[:take])

def pull(game):
    html=fetch(MN_SRC[game]); dt, nums = parse_html(game, html)
    if not nums: return pd.DataFrame(columns=["date","numbers","game"])
    return pd.DataFrame([{"date": dt or now().date().isoformat(), "numbers": nums, "game": game}])

def _upload_csv_daily(g:str, df:pd.DataFrame):
    """Upload the current CSV for a game to GitHub after each save."""
    if df is None or df.empty: return
    if GH_TOKEN and GH_OWNER and GH_REPO:
        try:
            repo_path = f"data/{g}_history.csv"
            ok = gh_upload_text(repo_path, df.to_csv(index=False), f"Daily update {g} {now().isoformat()}")
            if ok: st.info(f"📤 Pushed {g}_history.csv to GitHub")
        except Exception as e:
            st.warning(f"Daily GitHub push failed for {g}: {e}")

def save_hist(g, df_new):
    p = HIST(g)
    old = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["date","numbers","game"])
    merged = pd.concat([old, df_new], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
    merged.to_csv(p, index=False)
    # NEW: upload this CSV daily to GitHub
    _upload_csv_daily(g, merged)
    return merged

def load_hist(g):
    p = HIST(g)
    if p.exists():
        df = pd.read_csv(p); df["date"]=pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"])
    return pd.DataFrame(columns=["date","numbers","game"])

# ---------- seeding from /data/N5_source.zip ----------
def _norm_from_grid(df:pd.DataFrame, game:str, keep_only_five:bool=True)->pd.DataFrame:
    # Expect a grid like: Date, N1..N5 (PB: we keep only five mains)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    dcol = next((c for c in cols if "date" in c), cols[0])
    rest = [c for c in cols if c != dcol]
    if keep_only_five and len(rest) >= 5:
        rest = rest[:5]
    numbers = df[rest].astype(str).agg(",".join, axis=1)
    out = pd.DataFrame({"date": pd.to_datetime(df[dcol], errors="coerce"),
                        "numbers": numbers, "game": game})
    out = out.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
    return out[["date","numbers","game"]]

def seed_zip():
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
            df = pd.read_excel(path) if path.suffix==".xlsx" else pd.read_csv(path)
            norm = _norm_from_grid(df, game, keep_only_five=True)
            norm.to_csv(HIST(game), index=False)
            st.success(f"{game}: seeded {len(norm)} rows from {path.name}")

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

# ---------- analytics ----------
def explode(df):
    rows=[]
    for s in df["numbers"].astype(str):
        v = [int(x) for x in re.findall(r"\d+", s)][:5]
        if len(v)==5: rows.append(v)
    return pd.DataFrame(rows, columns=[f"n{i}" for i in range(1,6)]) if rows else pd.DataFrame()

def trickle(n5):
    nums=[]
    for s in n5.tail(20)["numbers"]:
        nums += [int(x) for x in re.findall(r"\d+", s)]
    ser = pd.Series(nums).value_counts(); w={}
    for n,f in ser.items():
        n=int(n); w[n]=float(f)
        for d in (-1,1): w[n+d]=w.get(n+d,0)+0.3
    return w

def choice(rng, w, k=5):
    picks=set(); local=w.copy()
    while len(picks)<k and not local.empty:
        p = rng.choice(local.index, p=(local/local.sum()).values)
        picks.add(int(p)); local = local.drop(p, errors="ignore")
    return sorted(picks)

def monte(hist, tr=None):
    if hist.empty: return [], 0.0
    df = explode(hist.tail(60))
    if df.empty: return [], 0.0
    freq = pd.Series(df.values.reshape(-1,)).value_counts().astype(float)
    if tr:
        for n,w in tr.items(): freq.loc[n] = freq.get(n,0) + w*0.15
    rng = np.random.default_rng(int(datetime.now().timestamp()))
    trials=9000; bucket={}
    for _ in range(trials):
        pick = choice(rng, freq)
        if pick:
            t=tuple(pick); bucket[t]=bucket.get(t,0)+1
    top,cnt = max(bucket.items(), key=lambda kv: kv[1])
    conf = round(min(99, (cnt/trials)*130), 2)
    return list(top), conf

def log_conf(g,c):
    row = pd.DataFrame([[now(),g,c]], columns=["timestamp","game","confidence"])
    base = pd.read_csv(CONF_PATH) if CONF_PATH.exists() else pd.DataFrame(columns=row.columns)
    pd.concat([base,row], ignore_index=True).to_csv(CONF_PATH, index=False)

def score(g, pred, act):
    aset=set(act); ex=len(set(pred)&aset)
    pm1=sum(1 for p in pred if any(abs(p-a)==1 for a in aset))
    row=pd.DataFrame([[now(),g,ex,pm1]], columns=["timestamp","game","exact","plusminus1"])
    base=pd.read_csv(PERF_PATH) if PERF_PATH.exists() else pd.DataFrame(columns=row.columns)
    pd.concat([base,row], ignore_index=True).to_csv(PERF_PATH, index=False)

# ---------- runs ----------
def run_post(g):
    df = pull(g); hist = save_hist(g, df)
    if hist.empty: return
    pred,_ = monte(hist)
    last = [int(x) for x in re.findall(r"\d+", hist.iloc[-1]["numbers"])]
    score(g, pred, last)

def run_pre(g):
    h = load_hist(g)
    tr = trickle(load_hist("N5"))
    p,c = monte(h, tr if g in ("G5","PB") else None)
    pd.DataFrame([[",".join(map(str,p)), c]], columns=["numbers","confidence"]).to_csv(LAST(g), index=False)
    log_conf(g, c)

def archive():
    n=now()
    if n.weekday()==6 and 16<=n.hour<17:
        tag=n.strftime("%Y%m%d_%H%M")
        zp=ARCH/f"northstar_{tag}.zip"
        if not zp.exists():
            with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as z:
                for f in DATA.glob("*.csv"): z.write(f, f.name)
        # Upload weekly archive to GitHub
        if GH_TOKEN and GH_OWNER and GH_REPO:
            ok = gh_upload_bytes(f"archives/{zp.name}", zp.read_bytes(), f"Archive {zp.name}")
            if ok: st.success(f"📤 Weekly archive uploaded to GitHub: {zp.name}")
        # Optional email
        if EM:
            sent = email_send(subject=f"Northstar weekly archive {tag}",
                              body="Attached: weekly archive generated by Northstar.",
                              attachment=zp)
            if sent: st.success("✉️ Archive emailed.")

def daily_pull():
    for g in GAMES:
        save_hist(g, pull(g))

def scheduled_post_draws():
    for g in GAMES:
        if now().weekday() in DRAW_DAYS[g]: run_post(g)

def scheduled_pre_draws():
    for g in GAMES:
        if now().weekday() in DRAW_DAYS[g]: run_pre(g)

# ---------- scheduler tick ----------
def sched():
    # Seed once (from /data/N5_source.zip)
    seed_zip()
    # 5 AM window OR first open fallback: ensure one daily pull per day
    if not_today("daily") and in_window(HOURS["daily"]):
        daily_pull(); mark("daily")
    if not_today("daily"):   # fallback on first open
        daily_pull(); mark("daily")

    # 7 AM post
    if not_today("post") and in_window(HOURS["post"]):
        scheduled_post_draws(); mark("post")

    # 11 AM and 1 PM pre
    if not_today("mid1") and in_window(HOURS["mid1"]):
        scheduled_pre_draws(); mark("mid1")
    if not_today("mid2") and in_window(HOURS["mid2"]):
        scheduled_pre_draws(); mark("mid2")

    # 4 PM final (respect draw calendars)
    if not_today("final") and in_window(HOURS["final"]):
        for g in GAMES:
            if now().weekday() in DRAW_DAYS[g]: run_pre(g)
        mark("final")

    # Weekly archive
    archive()

# ---------- UI ----------
st.set_page_config(page_title=f"Northstar {APP_VER}", page_icon="🌟", layout="wide")
st.title(f"🌟 Northstar Ecosystem — v{APP_VER}")
st.caption("Auto ZIP seed • Live MN pulls • Monte Carlo v2.7-M • 5/7/11/1/4 CST • Daily GitHub CSV + Weekly archive GitHub/email")

sched()

tab1,tab2,tab3,tab4,tab5 = st.tabs(["Status","Predictions","Confidence","History","Tools"])

with tab1:
    m=manifest()
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Daily (5 AM)", m.get("daily") or "—")
    c2.metric("Post (7 AM)", m.get("post")  or "—")
    c3.metric("Pre 11 AM",   m.get("mid1")  or "—")
    c4.metric("Pre 1 PM",    m.get("mid2")  or "—")
    c5.metric("Final 4 PM",  m.get("final") or "—")

    st.subheader("Manual controls")
    b1,b2,b3,b4 = st.columns(4)
    if b1.button("🔄 Pull now"):
        daily_pull(); st.success("Pulled & pushed CSVs (if GH configured).")
    if b2.button("🧪 Run post-draw now"):
        scheduled_post_draws(); st.success("Post-draw done.")
    if b3.button("🔮 Run pre-draw now"):
        scheduled_pre_draws(); st.success("Pre-draw done.")
    if b4.button("📦 Archive now"):
        archive(); st.success("Archive step complete (GitHub/email if configured).")

    st.divider()
    if st.button("🚀 Run Entire System Now"):
        with st.spinner("Pull → Post → Pre(11/1) → Final(4) → Archive …"):
            daily_pull()
            scheduled_post_draws()
            scheduled_pre_draws()  # stand-in for mid-windows
            scheduled_pre_draws()  # again (simulating 1 PM)
            for g in GAMES:
                if now().weekday() in DRAW_DAYS[g]: run_pre(g)  # final
            archive()
            st.success(f"✅ Completed — {now().strftime('%I:%M %p %Z')}")

with tab2:
    st.subheader("Next Predictions")
    for g in GAMES:
        p = LAST(g)
        if p.exists(): st.write(f"**{g}**"); st.dataframe(pd.read_csv(p).tail(1), use_container_width=True)
        else: st.write(f"**{g}** — no prediction yet.")

with tab3:
    st.subheader("Confidence")
    if CONF_PATH.exists():
        df=pd.read_csv(CONF_PATH)
        if not df.empty:
            df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
            df=df.dropna(subset=["timestamp"]).tail(300)
            st.line_chart(df.pivot(index="timestamp", columns="game", values="confidence"))
    st.subheader("Performance (tail)")
    if PERF_PATH.exists():
        st.dataframe(pd.read_csv(PERF_PATH).tail(50), use_container_width=True)

with tab4:
    st.subheader("Latest histories")
    cols=st.columns(3)
    for i,g in enumerate(GAMES):
        with cols[i]:
            df=load_hist(g)
            st.write(f"**{g} ({len(df)} rows)**")
            st.dataframe(df.tail(15), use_container_width=True, height=300)

with tab5:
    st.write("**Data folder**"); st.code(str(DATA))
    c1,c2,c3 = st.columns(3)
    if c1.button("📥 Seed ZIP Now"):
        seed_zip(); st.success("Seeding attempted. Check History tab.")
    if c2.button("🧪 Test GitHub upload"):
        ok = gh_upload_text("archives/dummy.txt", f"Northstar test {now().isoformat()}", "Test upload")
        st.success("GitHub OK") if ok else st.warning("GitHub not configured or failed.")
    if c3.button("✉️ Send test email"):
        ok = email_send("Northstar test email", "This is a test from Northstar.", None)
        st.success("Email OK") if ok else st.warning("Email not configured/failed.")
    st.caption("Daily CSVs auto-push to GitHub after each save if [github] secrets are set.")
