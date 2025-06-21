import pandas as pd
import re
from langchain_core.tools import tool
from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document

# Load Data
CSV_FILE_PATH = 'https://raw.githubusercontent.com/valengrcla/celerates/refs/heads/main/Indonesian_Food_Recipes.csv'
df = pd.read_csv(CSV_FILE_PATH)
df_cleaned = df.dropna(subset=['Title', 'Ingredients', 'Steps']).drop_duplicates()

# Normalisasi
def normalize_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return text

df_cleaned['Title_Normalized'] = df_cleaned['Title'].apply(normalize_text)
df_cleaned['Ingredients_Normalized'] = df_cleaned['Ingredients'].apply(normalize_text)
df_cleaned['Steps_Normalized'] = df_cleaned['Steps'].apply(normalize_text)

def format_recipe(row):
    bahan_raw = re.split(r'\n|--|,', row['Ingredients'])
    bahan_list = [b.strip().capitalize() for b in bahan_raw if b.strip()]
    bahan_md = "\n".join([f"- {b}" for b in bahan_list])
    langkah_md = row['Steps'].strip()
    return f"""🍽 *{row['Title']}*

*Bahan-bahan:*  
{bahan_md}

*Langkah Memasak:*  
{langkah_md}"""

# Tool 1: Cari berdasarkan judul
@tool
def cari_judul(query: str) -> str:
    """Mencari resep berdasarkan judul masakan"""
    q = normalize_text(query)
    match = df_cleaned[df_cleaned['Title_Normalized'].str.contains(q)]
    if not match.empty:
        return format_recipe(match.iloc[0])
    return "Resep tidak ditemukan berdasarkan judul."

# Tool 2: Cari berdasarkan bahan
@tool
def cari_bahan(query: str) -> str:
    """Mencari daftar resep yang menggunakan bahan tertentu"""
    q = normalize_text(query)
    match = df_cleaned[df_cleaned['Ingredients_Normalized'].str.contains(q)]
    if not match.empty:
        return "Resep dengan bahan tersebut:\n- " + "\n- ".join(match.head(5)['Title'].tolist())
    return "Tidak ada resep dengan bahan tersebut."

# Tool 3: Cari berdasarkan metode masak
@tool
def cari_metode(query: str) -> str:
    """Mencari resep berdasarkan metode memasak: goreng, kukus, rebus, panggang"""
    q = normalize_text(query)
    for metode in ['goreng', 'kukus', 'rebus', 'panggang']:
        if metode in q:
            match = df_cleaned[df_cleaned['Steps_Normalized'].str.contains(metode)]
            if not match.empty:
                return f"Resep yang dimasak dengan cara {metode}:\n- " + "\n- ".join(match.head(5)['Title'].tolist())
    return "Tidak ditemukan metode memasak yang cocok."

# Tool 4: Resep mudah (heuristik)
@tool
def resep_mudah(_: str) -> str:
    """Menampilkan resep-resep yang mudah dan cocok untuk pemula"""
    match = df_cleaned[df_cleaned['Steps'].str.len() < 300]
    return "Rekomendasi resep mudah:\n- " + "\n- ".join(match.head(5)['Title'].tolist())

# Tool 5: RAG - vektor database
def init_vector_db(api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    documents = [
        Document(
            page_content=f"{row['Title']}\nBahan:\n{row['Ingredients']}\nLangkah:\n{row['Steps']}",
            metadata={"title": row['Title']}
        )
        for _, row in df_cleaned.iterrows()
    ]
    return FAISS.from_documents(documents, embedding=embeddings)

@tool
def rag_resep(query: str) -> str:
    """Menjawab pertanyaan menggunakan RAG dari kumpulan resep"""
    retriever = init_vector_db(GOOGLE_API_KEY).as_retriever(search_type="similarity", k=5)
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""Berikut beberapa referensi resep masakan:\n\n{context}\n\nGunakan ini untuk menjawab:\n{query}"""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
    response = llm.invoke(prompt)
    return response.content

# Init Agent
def init_agent(api_key):
    global GOOGLE_API_KEY
    GOOGLE_API_KEY = api_key
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    tools = [cari_judul, cari_bahan, cari_metode, resep_mudah, rag_resep]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True
    )
    return agent
