import streamlit as st

st.title
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Northstar Dashboard", layout="wide")

st.title("🌟 Northstar — All-in-One System")
st.write("System is connected and ready.")

uploaded = st.file_uploader("Upload your N5, G5, or Powerball CSV files")
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())
