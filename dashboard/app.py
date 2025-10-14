import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(layout='wide', page_title='Rover Dashboard')

st.title('Rover-in-a-Box Dashboard')

col1, col2 = st.columns([2,1])
with col1:
    st.header('Camera Feed')
    uploaded = st.file_uploader('Upload a rover image', type=['png','jpg','jpeg'])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption='Rover image', use_column_width=True)

with col2:
    st.header('Metrics')
    st.write('No data yet. Start the rover and AI pipeline to see metrics.')

st.sidebar.header('Controls')
if st.sidebar.button('Run demo (local)'):
    st.info('Demo not yet implemented — follow Day 1 instructions')
