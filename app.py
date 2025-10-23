# --- Google Drive Auto-Sync ---
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os

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
