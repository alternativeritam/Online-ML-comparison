import main
import noise
import dataexpo
import visual
import streamlit as st
st.set_page_config(page_title='The Machine Learning Algorithm Comparison App',
                   layout='wide')

PAGES = {
    "Algortihm selection": main,
    "Noise detection": noise,
    "Data Exploration": dataexpo,
    "Data visualization": visual
}

with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader(
        "Upload your input CSV file", type=["csv"])

value = st.selectbox("select option", options=list(PAGES.keys()))
page = PAGES[value]
if uploaded_file is not None:
    if value == "Algortihm selection":
        page.app1(uploaded_file)
    elif value == "Data Exploration":
        page.app3(uploaded_file)
    elif value == "Data visualization":
        page.app4(uploaded_file)

    else:
        page.app2(uploaded_file)
else:
    st.warning("Upload your dataset")
