import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans

import plotly.express as px

st.set_page_config(page_title="Travel Analytics", layout="wide")
st.title("Travel Experience Analytics Dashboard")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    df_enc = df.copy()
    for col in df_enc.columns:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

    X = df_enc.drop("Likelihood", axis=1)
    y = df_enc["Likelihood"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.subheader("Classification Metrics")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
    st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
    st.write("F1:", f1_score(y_test, y_pred, average='weighted'))

    st.subheader("Feature Importance")
    imp = pd.Series(clf.feature_importances_, index=X.columns)
    st.bar_chart(imp)

    st.subheader("Clustering")
    kmeans = KMeans(n_clusters=4)
    df['Cluster'] = kmeans.fit_predict(X)
    st.write(df['Cluster'].value_counts())

    st.subheader("Regression (Spend)")
    if "Spend" in df_enc.columns:
        reg = RandomForestRegressor()
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        st.write(preds[:10])

    st.subheader("New Customer Prediction")
    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(col, value=0)

    if st.button("Predict"):
        user_df = pd.DataFrame([user_input])
        pred = clf.predict(user_df)
        st.write("Prediction:", pred)
