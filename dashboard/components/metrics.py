import streamlit as st

def show_metrics(metrics: dict):
    for k,v in metrics.items():
        st.metric(k, v)
