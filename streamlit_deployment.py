import streamlit as st
from transformers import pipeline
import numpy
import pandas


classifier=pipeline('text-classification',model="checkpoints")

st.title('classification of depression')

text=st.text_area('enter your text')

if st.button('predict'):
    result=classifier([text])
    st.write(result)
elif st.button('calculate'):
    st.write('no output')
