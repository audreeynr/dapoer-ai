# dapoer_ai.py
import streamlit as st
from dapoer_module import init_agent

st.set_page_config(page_title="Dapoer-AI", page_icon="🍲")
st.title("🍛 Dapoer-AI - Asisten Resep Masakan Indonesia")

# Input API Key Gemini
GOOGLE_API_KEY = st.text_input("Masukkan API Key Gemini kamu:", type="password")
if not GOOGLE_API_KEY:
    st.warning("Silakan masukkan API key untuk mulai.")
    st.stop()

# Inisialisasi Langchain Agent
agent = init_agent(GOOGLE_API_KEY)

# Inisialisasi memory chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hai! Mau masak apa hari ini? Tanya aja judul, bahan, atau cara masaknya ya~"
    })

# Tampilkan riwayat chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input pengguna
if prompt := st.chat_input("Tanyakan resep, bahan, atau metode memasak..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = agent.invoke({"input": prompt})
            jawaban = response["output"]
        except Exception as e:
            jawaban = "Maaf, terjadi kesalahan saat menjawab. Coba lagi nanti ya."

        st.markdown(jawaban)
        st.session_state.messages.append({"role": "assistant", "content": jawaban})
