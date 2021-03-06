import streamlit as st
from sentiment import nlpToolkit

st.image("data/logo.png")
st.markdown(" ### Inference is CPU based so processing might take some time.")


option = st.selectbox('Which feature would you like to use?',('Sentiment Analysis','Part of Speech Tagging','Named Entity Recognition'))

sent = st.text_input("Enter sentence")

toolk = nlpToolkit()

if(sent != "" and option == "Sentiment Analysis"):
    try:
        slot = st.empty()
        slot.text("please wait....")
        st.write("")
        result = toolk.sentiment(sent)
        slot.text("The sentiment is: " + str(result[0]))
    except:
        st.text("Invalid input. Please try again.")

if(sent != "" and option == "Named Entity Recognition"):
    try:
        slot = st.empty()
        slot.text("please wait....")
        st.write("")
        result = toolk.ner(sent)
        slot.write("The recognised entities are: " + str(result[1]))
    except:
        st.text("Invalid input. Please try again.")

if(sent != "" and option == "Part of Speech Tagging"):
    try:
        slot = st.empty()
        slot.text("please wait....")
        st.write("")
        result = toolk.pos(sent)
        slot.write("The POS tags are: " + str(result[1]))
    except:
        st.text("Invalid input. Please try again.")
