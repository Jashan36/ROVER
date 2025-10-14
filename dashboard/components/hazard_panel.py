import streamlit as st

def show_hazards(hazards):
    st.write('Hazards')
    for h in hazards:
        st.write(h)
