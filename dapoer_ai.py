import streamlit as st
import pandas as pd
import re
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.agents import load_tools
from langchain.agents.agent_toolkits import create_retriever_tool

# --- Load & Clean Recipe Data ---
CSV_FILE = 'https://raw.githubusercontent.com/valengrcla/celerates/refs/heads/main/Indonesian_Food_Recipes.csv'
df = pd.read_csv(CSV_FILE).dropna(subset=['Title', 'Ingredients', 'Steps']).drop_duplicates()

def preprocess(row):
    return f"Judul: {row['Title']}\n\nBahan-bahan:\n{row['Ingredients']}\n\nLangkah-langkah:\n{row['Steps']}"

documents = [Document(page_content=preprocess(row), metadata={"judul": row['Title']}) for _, row in df.iterrows()]

# --- Streamlit UI Setup ---
st.set_page_config(page_title="🍳 Dapoer-AI", page_icon="🍲")
st.title("🍛 Dapoer-AI - Chatbot Resep Masakan Indonesia dengan LLM + RAG")

GOOGLE_API_KEY = st.text_input("Masukkan API Key Gemini kamu:", type="password")
if not GOOGLE_API_KEY:
    st.warning("Masukkan API key terlebih dahulu.")
    st.stop()

# --- Konfigurasi LLM & Embedding ---
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(documents, embedding)
retriever = vectorstore.as_retriever(search_type="similarity", k=3)

# --- Buat Tool Langchain ---
@tool
def get_easy_recipe(_: str) -> str:
    """Memberikan resep yang mudah dengan sedikit bahan dan langkah."""
    df["score"] = df["Ingredients"].str.count("--") + df["Steps"].str.count("\n")
    easiest = df.sort_values(by="score").iloc[0]
    return f"Resep termudah:\n{preprocess(easiest)}"

@tool
def list_all_recipes(_: str) -> str:
    """Menampilkan daftar semua judul resep yang tersedia."""
    return "\n".join(df['Title'].tolist())

retriever_tool = create_retriever_tool(
    retriever,
    name="search_resep_masakan",
    description="Mencari resep makanan Indonesia dari data yang tersedia berdasarkan permintaan pengguna."
)

tools = [retriever_tool, get_easy_recipe, list_all_recipes]

# --- Setup LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# --- Agent Initialization ---
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# --- Chat Memory (3 terakhir) ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Display Chat History ---
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# --- User Chat Input ---
if prompt := st.chat_input("Tanyakan resep, bahan, atau jenis masakan..."):
    st.session_state.chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = agent_executor.run(prompt)
        st.markdown(response)
        st.session_state.chat_history.append(("assistant", response))

    # Batasi ke 6 pesan (3 interaksi)
    if len(st.session_state.chat_history) > 6:
        st.session_state.chat_history = st.session_state.chat_history[-6:]
