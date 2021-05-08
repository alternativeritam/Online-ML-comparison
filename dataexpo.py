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


def app3(uploaded_file):
    warnings.filterwarnings("ignore")
    df = pd.read_csv(uploaded_file)
    st.write("Total data sample numbers : "+str(len(df)))
    num = st.number_input("Enter The sample size", 0)
    if num != 0 and num <= len(df):
        pr = ProfileReport(df.sample(n=num), minimal=True)
        st.header('**Exploratory data Analysis**')
        st_profile_report(pr)
    elif num > len(df) and num != 0:
        st.warning("Enter proper sample size")
    else:
        st.warning("Enter the sample size")
