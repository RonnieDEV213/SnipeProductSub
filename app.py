import streamlit as st
import os
from openai import OpenAI

st.title("API Key Test")

# make sure your key is set in environment
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

if st.button("Test GPT"):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello from GPT!"}]
    )
    st.write(resp.choices[0].message.content)
