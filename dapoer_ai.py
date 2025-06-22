# dapoer_ai.py
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from dapoer_module import tools  # pastikan 'tools' sudah ada di modul

st.set_page_config(page_title="Dapoer-AI", page_icon="🍲")
st.title("🍛 Dapoer-AI - Asisten Resep Masakan Indonesia")

# Masukkan API Key Gemini
GOOGLE_API_KEY = st.text_input("Masukkan API Key Gemini kamu:", type="password")
if not GOOGLE_API_KEY:
    st.warning("Silakan masukkan API key untuk mulai.")
    st.stop()

# Inisialisasi model Gemini via Langchain
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# Inisialisasi Memory & Agent
if "agent" not in st.session_state:
    memory = ConversationBufferMemory(memory_key="chat_history")
    st.session_state.agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type="zero-shot-react-description",
        memory=memory,
        verbose=False
    )
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "👋 Hai! Mau masak apa hari ini?"})

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
        response = st.session_state.agent.run(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
