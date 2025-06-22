# dapoer_ai.py
import streamlit as st
import google.generativeai as genai
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from dapoer_module import tools  # Langchain tools dari dapoer_module

# Pengaturan halaman
st.set_page_config(page_title="Dapoer-AI", page_icon="🍲")
st.title("🍛 Dapoer-AI - Asisten Resep Masakan Indonesia")

# Input API Key
GOOGLE_API_KEY = st.text_input("Masukkan API Key Gemini kamu:", type="password")
if not GOOGLE_API_KEY:
    st.warning("Silakan masukkan API key untuk mulai.")
    st.stop()

# Konfigurasi model Gemini
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatOpenAI(model="gemini-pro", temperature=0)  # ChatOpenAI bisa diganti wrapper Gemini kamu

# Inisialisasi Langchain Agent dan Memory
if "agent" not in st.session_state:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type="zero-shot-react-description",
        memory=memory,
        verbose=False,
    )

# Inisialisasi chat session
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "👋 Hai! Mau masak apa hari ini?"})

# Tampilkan chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input dari user
if prompt := st.chat_input("Tanyakan resep, bahan, atau metode memasak..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Jalankan Langchain Agent
    with st.chat_message("assistant"):
        response = st.session_state.agent.run(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
