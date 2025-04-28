import streamlit as st
import sklearn

predict = st.Page(
    page = "stream_pages/PredictYourself.py",
    title = '👉Predict Yourself',
    default = True,
)

st.set_page_config(layout="centered", initial_sidebar_state="auto")

abc = st.navigation(
    {
        "🎯Predict Cube Compressive Strength of Concrete": [predict],
    }
)

abc.run()
