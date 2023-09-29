"""This modules contains data about about page"""

# Import necessary modules
import streamlit as st
from PIL import Image


def app():
    """This function creates the about page"""
    st.balloons()
    st.title('Contact Us')
    st.markdown('''### Name: Mainak Chaudhuri''')
    st.markdown('''Passionate Engineer and Rational Thinker. Data Scientist, Web Developer, Blockchain Maestro, Quantum Computing Tutor''')
    st.image('./images/icon.jpg')
    st.markdown('''### Linkedin: [Mainak](https://www.linkedin.com/in/mainak-chaudhuri-127898176/)''')
    st.markdown('''### GitHub: [Mainak](https://github.com/MainakRepositor/)''')
    