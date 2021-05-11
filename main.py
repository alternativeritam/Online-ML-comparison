import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn .svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes, load_boston
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import numpy as np
import io
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor, DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.linear_model import Lars
from sklearn.svm import SVR, NuSVR, LinearSVC, NuSVC
import lightgbm as lgbm
import math
import sklearn
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, DotProduct, RationalQuadratic


def add_parameter_ui(classifier_name):
    param = dict()
    if classifier_name == "KNN":
        k = st.sidebar.slider("K", 1, 15)
        param["K"] = k

    elif classifier_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        param["C"] = C
    elif classifier_name == "Random Forest":
        max_depth = st.sidebar.slider("Max depth", 2, 16)
        estimator = st.sidebar.slider("estimator", 1, 100)
        param["max_depth"] = max_depth
        param["estimator"] = estimator
    elif classifier_name == "BaggingClassifier":
        param["base_estimator"] = SVC()
        n_estimator = st.sidebar.slider("n_estimator", 1, 20)
        max_sample = st.sidebar.slider("Max Sample", 0.1, 1.0)
        param["n_estimator"] = n_estimator
        param["max_sample"] = max_sample
    elif classifier_name == "DecisionTreeClassifier":
        max_depth = st.sidebar.slider("Max depth", 2, 16)
        leaf = st.sidebar.slider("Leaf Nodes", 1, 100)
        param["max_depth"] = max_depth
        param["Leaf"] = leaf
    elif classifier_name == "ExtraTreeClassifier":
        max_depth = st.sidebar.slider("Max depth", 2, 16)
        leaf = st.sidebar.slider("Leaf Nodes", 1, 100)
        min_sample_split = st.sidebar.slider("min_sample_split", 0.1, 5.0)
        param["max_depth"] = max_depth
        param["Leaf"] = leaf
        if min_sample_split <= 1:
            param["min_sample_split"] = float(min_sample_split)
        elif min_sample_split > 1 and min_sample_split < 2:
            param["min_sample_split"] = int(min_sample_split)+1
        else:
            param["min_sample_split"] = int(min_sample_split)
    elif classifier_name == "GaussianProcessClassifier":
        n_restarts_optimizer = st.sidebar.slider("n_restarts_optimizer", 0, 10)
        max_iter_predict = st.sidebar.slider("max_iter_predict", 50, 100)
        l = [1*RBF(), 1*DotProduct(), 1*ConstantKernel(), 1 *
             RationalQuadratic()]
        kernal = st.sidebar.selectbox("Kernal", l)
        param["optimizer"] = n_restarts_optimizer
        param["predict"] = max_iter_predict
        param["kernal"] = kernal
    elif classifier_name == "LinearSVC":
        loss = st.sidebar.selectbox("loss", ('hinge', 'squared_hinge'))
        C = st.sidebar.slider("C", 0.1, 5.0)
        param["loss"] = loss
        param["c"] = C
    elif classifier_name == "NuSVC":
        kernel = st.sidebar.selectbox(
            "kernel", ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'))
        degree = st.sidebar.slider("degree", 0, 5)
        gamma = st.sidebar.selectbox("Gamma", ('scale', 'auto'))
        param["kernel"] = kernel
        param["degree"] = degree
        param["gamma"] = gamma
    elif classifier_name == "AdaBoostClassifier":
        n_estimator = st.sidebar.slider("n_estimators", 10, 100)
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.00)
        algorithm = st.sidebar.selectbox("Algorithm", ('SAMME', 'SAMME.R'))
        param["estimators"] = n_estimator
        param["learning"] = learning_rate
        param["algo"] = algorithm
    return param


def add_linear_para(linear_model):
    pass


def get_classifier(classifier_name, params):
    if classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])

    elif classifier_name == "SVM":
        clf = SVC(C=params["C"])
    elif classifier_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=params["estimator"], max_depth=params["max_depth"], random_state=42)
    elif classifier_name == "BaggingClassifier":
        clf = BaggingClassifier(
            base_estimator=params["base_estimator"], n_estimators=params["n_estimator"], max_samples=params["max_sample"])
    elif classifier_name == "DecisionTreeClassifier":
        clf = DecisionTreeClassifier(
            max_leaf_nodes=params["Leaf"], max_depth=params["max_depth"], random_state=42)
    elif classifier_name == "ExtraTreeClassifier":
        clf = ExtraTreeClassifier(
            max_depth=params["max_depth"], min_samples_leaf=params["Leaf"], min_samples_split=params["min_sample_split"])
    elif classifier_name == "GaussianProcessClassifier":
        clf = GaussianProcessClassifier(
            kernel=params["kernal"], max_iter_predict=params["predict"], n_restarts_optimizer=params["optimizer"], random_state=42)
    elif classifier_name == "LinearSVC":
        clf = LinearSVC(loss=params["loss"], C=params["c"])
    elif classifier_name == "NuSVC":
        clf = NuSVC(kernel=params["kernel"],
                    degree=params["degree"], gamma=params["gamma"])
    elif classifier_name == "AdaBoostClassifier":
        clf = AdaBoostClassifier(
            n_estimators=params["estimators"], learning_rate=params["learning"], algorithm=params["algo"])

    return clf


def get_linear_model(model_name):
    if model_name == "LinearRegression":
        reg = LinearRegression()
    elif model_name == "RandomForestRegressor":
        reg = RandomForestRegressor()
    elif model_name == "DecisionTreeRegressor":
        reg = DecisionTreeRegressor()
    elif model_name == "GaussianProcessRegressor":
        reg = GaussianProcessRegressor()
    elif model_name == "ExtraTreeRegressor":
        reg = ExtraTreeRegressor()
    elif model_name == "LGBMRegressor":
        reg = lgbm.sklearn.LGBMRegressor()
    elif model_name == "BaggingRegressor":
        reg = BaggingRegressor()
    elif model_name == "KNeighborsRegressor":
        reg = KNeighborsRegressor()
    elif model_name == "Lars":
        reg = Lars()
    elif model_name == "SVR":
        reg = SVR()
    elif model_name == "NuSVR":
        reg = NuSVR()
    return reg


def filedownload(df, filename):
    csv = df.to_csv(index=False)
    # strings <-> bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href


def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    # strings <-> bytes conversions
    b64 = base64.b64encode(s.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href


def build_model(df, model, split_size, seed_number, label):
    # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
    progress = 0
    if label != "":
        dataTypeObj = df.dtypes[label]
        Y = df[[label]]
        # Using all column except for the last column as X
        X = df.drop([label], axis=1)
        progress = 1

    if progress == 1:
        st.markdown('**Dataset dimension**')
        st.write('X')
        st.info(X.shape)
        st.write('Y')
        st.info(Y.shape)

        st.markdown('**Variable details**:')
        st.write('X variable')
        st.info(list(X.columns))
        st.write('Y variable')
        st.info(label)
        # Build lazy model
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=split_size, random_state=seed_number)
        agree = st.selectbox("Which type of choose", ("One", "All"))
        if agree == "One":
            if model == 'Classification':
                l = ["KNN", "SVM", "Random Forest", "BaggingClassifier",
                     "DecisionTreeClassifier", "ExtraTreeClassifier", "GaussianProcessClassifier", "LinearSVC", "NuSVC", "AdaBoostClassifier"]
                classifier_name = st.selectbox(
                    "Select Classifier", l)
                params = add_parameter_ui(classifier_name)

                clf = get_classifier(classifier_name, params)
                clf.fit(X_train, Y_train)

                Y_pred = clf.predict(X_test)

                acc = accuracy_score(Y_test, Y_pred)
                mse = mean_squared_error(Y_pred, Y_test)

                rmse = math. sqrt(mse)

                st.write(f"Classifier = {classifier_name}")
                st.write(f"Accuracy = {acc}")
                st.write(f"rmse = {rmse}")
            else:
                l = ["LinearRegression", "RandomForestRegressor", "DecisionTreeRegressor", "GaussianProcessRegressor",
                     "ExtraTreeRegressor", "LGBMRegressor", "BaggingRegressor", "KNeighborsRegressor", "Lars", "SVR", "NuSVR"]
                model_name = st.selectbox(
                    "Select Regression Model", l)
                reg = get_linear_model(model_name)
                reg.fit(X_train, Y_train)
                Y_pred = reg.predict(X_test)
                acc = reg.score(X_test, Y_test)
                mse = mean_squared_error(Y_pred, Y_test)
                rmse = math.sqrt(mse)
                st.write(f"Regression Model = {model_name}")
                st.write(f"Accuracy = {acc}")
                st.write(f"rmse = {rmse}")

        elif agree == "All":

            if model == 'Regression':
                reg = LazyRegressor(
                    verbose=0, ignore_warnings=False, custom_metric=None)
            elif model == 'Classification':
                reg = LazyClassifier(
                    verbose=0, ignore_warnings=True, custom_metric=None)
            models_train, predictions_train = reg.fit(
                X_train, X_train, Y_train, Y_train)
            models_test, predictions_test = reg.fit(
                X_train, X_test, Y_train, Y_test)

            st.subheader('Table of Model Performance')
            st.write('Training set')
            st.write(predictions_train)
            st.markdown(filedownload(predictions_train, 'training.csv'),
                        unsafe_allow_html=True)

            st.write('Test set')
            st.write(Y_test)
            st.write(predictions_test)
            st.write(models_test)
            st.markdown(filedownload(predictions_test, 'test.csv'),
                        unsafe_allow_html=True)

            st.subheader('4. Plot of Model Performance (Test set)')

            with st.markdown('**R-squared**'):
                # Tall
                predictions_test["R-Squared"] = [0 if i <
                                                 0 else i for i in predictions_test["R-Squared"]]
                plt.figure(figsize=(3, 9))
                sns.set_theme(style="whitegrid")
                ax1 = sns.barplot(y=predictions_test.index,
                                  x="R-Squared", data=predictions_test)
                ax1.set(xlim=(0, 1))
            st.markdown(imagedownload(plt, 'plot-r2-tall.pdf'),
                        unsafe_allow_html=True)
            # Wide
            plt.figure(figsize=(9, 3))
            sns.set_theme(style="whitegrid")
            ax1 = sns.barplot(x=predictions_test.index,
                              y="R-Squared", data=predictions_test)
            ax1.set(ylim=(0, 1))
            plt.xticks(rotation=90)
            st.pyplot(plt)
            st.markdown(imagedownload(plt, 'plot-r2-wide.pdf'),
                        unsafe_allow_html=True)

            with st.markdown('**RMSE (capped at 50)**'):
                # Tall
                plt.figure(figsize=(3, 9))
                sns.set_theme(style="whitegrid")
                ax2 = sns.barplot(y=predictions_test.index,
                                  x="RMSE", data=predictions_test)
            st.markdown(imagedownload(plt, 'plot-rmse-tall.pdf'),
                        unsafe_allow_html=True)
            # Wide
            plt.figure(figsize=(9, 3))
            sns.set_theme(style="whitegrid")
            ax2 = sns.barplot(x=predictions_test.index,
                              y="RMSE", data=predictions_test)
            plt.xticks(rotation=90)
            st.pyplot(plt)
            st.markdown(imagedownload(plt, 'plot-rmse-wide.pdf'),
                        unsafe_allow_html=True)

            with st.markdown('**Calculation time**'):
                # Tall
                predictions_test["Time Taken"] = [
                    0 if i < 0 else i for i in predictions_test["Time Taken"]]  # plt.figure(figsize=(3, 9))
                sns.set_theme(style="whitegrid")
                ax3 = sns.barplot(y=predictions_test.index,
                                  x="Time Taken", data=predictions_test)
            st.markdown(imagedownload(plt, 'plot-calculation-time-tall.pdf'),
                        unsafe_allow_html=True)
            # Wide
            plt.figure(figsize=(9, 3))
            sns.set_theme(style="whitegrid")
            ax3 = sns.barplot(x=predictions_test.index,
                              y="Time Taken", data=predictions_test)
            plt.xticks(rotation=90)
            st.pyplot(plt)
            st.markdown(imagedownload(plt, 'plot-calculation-time-wide.pdf'),
                        unsafe_allow_html=True)


def app1(uploaded_file):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #---------------------------------#
    # Page layout
    # Page expands to full width
    st.header("Algorithm Selection")
    #---------------------------------#
    # Model building

    # Download CSV data
    # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
    #---------------------------------#
    # Sidebar - Collects user input features into dataframe

    #---------------------------------#
    # Main panel

    # Displays the dataset
    st.subheader('1. Dataset')
    model = st.sidebar.selectbox(
        '2.Choose your Model', ('Classification', 'Regression'))
    with st.sidebar.header('3. Set Parameters'):
        split_size = st.sidebar.slider(
            'Data split ratio (% for Training Set)', 10, 90, 80, 5)
        seed_number = st.sidebar.slider(
            'Set the random seed number', 1, 100, 42, 1)
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    col_list = list(df.columns)
    label = st.selectbox("Select the label", col_list)
    build_model(df, model, split_size, seed_number, label)
