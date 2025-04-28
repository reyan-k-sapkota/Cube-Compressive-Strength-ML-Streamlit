import streamlit as st

predict = st.Page(
    page = "stream_pages/PredictYourself.py",
    title = '👉Predict Yourself',
    default = True,
)

st.set_page_config(layout="centered", initial_sidebar_state="auto")

pg = st.navigation(
    {
        "🎯Predict Cube Compressive Strength of Concrete": [predict],
    }
)

pg.run()
