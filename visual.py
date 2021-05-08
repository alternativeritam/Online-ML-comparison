import pandas as pd
from sklearn.impute import SimpleImputer
import streamlit as st
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import random
import main
import numpy as np
from sklearn.preprocessing import LabelEncoder
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

import warnings


def app4(uploaded_file):

    st.set_option('deprecation.showPyplotGlobalUse', False)
    warnings.filterwarnings("ignore")
    df = pd.read_csv(uploaded_file)
    st.write("**Overview of Dataset**")
    cor = st.checkbox("Correlation Matrix")
    if cor:
        st.subheader('2. Correlation matrix')
        corr_matrix = df.corr()
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(corr_matrix, annot=True, linewidths=0.1,
                    fmt=".2f", cmap="YlGnBu")
        st.pyplot()
    inter = st.checkbox("Columns Intercation")
    if inter:
        l = list(df.columns)
        col1 = st.selectbox('Enter the column1', l)
        col2 = st.selectbox('Enter the column2', l)
        typ = st.selectbox("Select the graph type", ("HeatMap", "Scatterplot"))
        if typ == "Scatterplot":
            st.markdown("**Scatterplot**")
            if col1 != "" and col2 != "":
                if col1 not in l or col2 not in l:
                    st.error("Enter proper column name")
                else:
                    sns.set_theme(style="ticks")
                    fig, ax = plt.subplots(figsize=(5, 5))
                    sns.lmplot(x=col1, y=col2, data=df)
                    st.pyplot()
        else:
            st.markdown("**HeatMap**")
            if col1 != "" and col2 != "":
                if col1 not in l or col2 not in l:
                    st.error("Enter proper column name")
                else:
                    sns.set_theme(style="ticks")
                    fig, ax = plt.subplots(figsize=(5, 5))
                    sns.jointplot(x=df[col1], y=df[col2],
                                  kind="hex", color="#4CB391")
                    st.pyplot()
    his_graph = st.checkbox("Histogram")
    if his_graph:
        df.hist(alpha=0.8, figsize=(50, 50))
        st.pyplot()
