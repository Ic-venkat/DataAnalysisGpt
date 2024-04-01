from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import pandas as pd
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(), override=True)

llm = OpenAI(api_token=os.environ['OPENAI_API_KEY'])

st.title("AI Data Analytics Application")

uploaded_file = st.sidebar.file_uploader("Upload File For Data Analysis", type="csv", accept_multiple_files= False, key="file-uploader-side-bar")
if(uploaded_file):
    df = pd.read_csv(uploaded_file, encoding= 'ANSI')
    st.dataframe(df)
    df1 = SmartDataframe(df, config={"llm": llm})


def chat_actions():
    st.session_state["chat_history"].append(
        {"role": "user", "content": st.session_state["Chat_input-chatbot"]},
    )
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


prompt = st.chat_input("Say something",key='Chat_input-chatbot',on_submit = chat_actions)

try:
    output_result = df1.chat(prompt)
    st.session_state["chat_history"].append(
        {"role": "assistant", "content": output_result},
    )
except NameError:
    with st.chat_message(name="assistant"):
        st.write("Input A file to use Data Analysis AI")



for i in st.session_state["chat_history"]:
    with st.chat_message(name=i["role"]):
        st.write(i["content"])





