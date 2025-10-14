import streamlit as st


def show_map(img, mask=None):
    st.image(img, caption='Map')
    if mask is not None:
        st.image(mask, caption='Mask')
