import streamlit as st
import google.generativeai as genai
from dapoer_module import handle_user_query

st.set_page_config(page_title="Dapoer-AI", page_icon="🍲")
st.title("🍛 Dapoer-AI - Asisten Resep Masakan Indonesia")

# Konfigurasi API Key
GOOGLE_API_KEY = st.text_input("Masukkan API Key Gemini kamu:", type="password")
if not GOOGLE_API_KEY:
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

# Chat session
if "messages" not in st.session_state:
    st.session_state.messages = []
# Opening message
    opening = "👋 Hai! Mau masak apa hari ini?"
    st.session_state.messages.append({"role": "assistant", "content": opening})

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Tanyakan resep, bahan, atau nama masakan..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = handle_user_query(prompt, model)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
