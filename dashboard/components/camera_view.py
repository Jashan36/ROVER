import streamlit as st
from PIL import Image

def show_camera(img: Image.Image):
    st.image(img, caption='Camera')
