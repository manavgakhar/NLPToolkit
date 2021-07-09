import streamlit as st
from sentiment import nlpToolkit

st.image("data/logo.png",width=None)
st.markdown(" ## This toolkit is based on pre-trained transformers so processing might take some time lol.")


option = st.selectbox('Which feature would you like to use?',('Sentiment Analysis','Part of Speech Tagging','Named Entity Recognition'))

sent = st.text_input("Enter sentence")

toolk = nlpToolkit()

if(sent != "" and option == "Sentiment Analysis"):
    try:
        result = toolk.sentiment(sent)
        st.write("The sentiment is: " + str(result[0]))
    except:
        st.text("Invalid input. Please try again.")

if(sent != "" and option == "Named Entity Recognition"):
    try:
        result = toolk.ner(sent)
        st.text("The recognised entities are: " + str(result[1]))
    except:
        st.text("Invalid input. Please try again.")

if(sent != "" and option == "Part of Speech Tagging"):
    try:
        result = toolk.pos(sent)
        st.write("The POS tags are: " + str(result[1]))
    except:
        st.text("Invalid input. Please try again.")
