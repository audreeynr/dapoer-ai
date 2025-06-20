import streamlit as st
import google.generativeai as genai
from dapoer_module import handle_user_query
from io import BytesIO

st.set_page_config(page_title="Dapoer-AI", page_icon="🍲")
st.title("🍛 Dapoer-AI - Asisten Resep Masakan Indonesia")

GOOGLE_API_KEY = st.text_input("Masukkan API Key Gemini kamu:", type="password")
if not GOOGLE_API_KEY:
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

# Session state untuk chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "👋 Hai! Mau masak apa hari ini?"})

# Tampilkan riwayat chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input pengguna
if prompt := st.chat_input("Tanyakan resep, bahan, atau nama masakan..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = handle_user_query(prompt, model)
        
        if isinstance(response, BytesIO):
            st.success("Resep siap diunduh:")
            st.download_button("📄 Unduh PDF Resep", data=response, file_name="resep_masakan.pdf", mime="application/pdf")
            st.session_state.messages.append({"role": "assistant", "content": "Resep berhasil disiapkan dalam bentuk PDF 📄"})
        else:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
