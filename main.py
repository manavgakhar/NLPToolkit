import streamlit as st
from sentiment import nlpToolkit

st.title("Sentiment Analysis using DistilBERT")

sent = st.text_input("Enter sentence")

if(sent != ""):
    toolk = nlpToolkit()

    result = toolk.sentiment(sent)

    st.text("The sentiment is: " + str(result[0]))
