# dapoer_module.py
import pandas as pd
import re
import random
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load dan bersihkan data
CSV_FILE_PATH = 'https://raw.githubusercontent.com/valengrcla/celerates/refs/heads/main/Indonesian_Food_Recipes.csv'
df = pd.read_csv(CSV_FILE_PATH)
df_cleaned = df.dropna(subset=['Title', 'Ingredients', 'Steps']).drop_duplicates()

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

# Format hasil resep
def format_recipe(row):
    bahan_raw = re.split(r'\n|--|,', row['Ingredients'])
    bahan_list = [b.strip().capitalize() for b in bahan_raw if b.strip()]
    bahan_md = "\n".join([f"- {b}" for b in bahan_list])
    langkah_md = row['Steps'].strip()
    return f"""🍽 **{row['Title']}**

**Bahan-bahan:**  
{bahan_md}

**Langkah Memasak:**  
{langkah_md}"""

# Tool 1: Cari resep berdasarkan nama
def search_by_title(query):
    match = df_cleaned[df_cleaned['Title_Normalized'].str.contains(normalize_text(query))]
    if not match.empty:
        return format_recipe(match.iloc[0])
    return "Tidak ada resep yang cocok dengan judul tersebut."

# Tool 2: Cari berdasarkan bahan
def search_by_ingredient(query):
    match = df_cleaned[df_cleaned['Ingredients_Normalized'].str.contains(normalize_text(query))]
    if not match.empty:
        hasil = match.head(5)['Title'].tolist()
        return "Masakan dengan bahan tersebut:\n- " + "\n- ".join(hasil)
    return "Tidak ditemukan masakan dengan bahan tersebut."

# Tool 3: Berdasarkan metode masak
def search_by_method(query):
    metode_list = ['goreng', 'panggang', 'rebus', 'kukus']
    for metode in metode_list:
        if metode in normalize_text(query):
            match = df_cleaned[df_cleaned['Steps_Normalized'].str.contains(metode)]
            if not match.empty:
                hasil = match.head(5)['Title'].tolist()
                return f"Masakan dengan metode {metode}:\n- " + "\n- ".join(hasil)
    return "Tidak ada metode memasak yang cocok ditemukan."

# Tool 4: Rekomendasi masakan mudah
def easy_recipes(_):
    match = df_cleaned[df_cleaned['Steps'].str.len() < 300].head(5)['Title'].tolist()
    return "Rekomendasi masakan mudah:\n- " + "\n- ".join(match)

# Tool 5: RAG-like tool
def rag_context_response(query):
    docs = "\n\n".join([
        f"{row['Title']}:\nBahan: {row['Ingredients']}\nLangkah: {row['Steps']}"
        for _, row in df_cleaned.sample(5, random_state=42).iterrows()
    ])
    prompt = f"""
Berikut beberapa resep masakan Indonesia:

{docs}

Gunakan referensi di atas untuk menjawab pertanyaan berikut:
{query}
"""
    return prompt

# Fungsi untuk inisialisasi agent Langchain
def init_agent(api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    tools = [
        Tool(name="Cari berdasarkan judul", func=search_by_title, description="Mencari resep masakan berdasarkan nama/judul masakan."),
        Tool(name="Cari berdasarkan bahan", func=search_by_ingredient, description="Mencari resep masakan berdasarkan bahan yang digunakan."),
        Tool(name="Cari berdasarkan metode memasak", func=search_by_method, description="Mencari resep berdasarkan cara masak seperti goreng, rebus, panggang."),
        Tool(name="Rekomendasi masakan mudah", func=easy_recipes, description="Menampilkan masakan mudah yang cocok untuk pemula."),
        Tool(name="RAG Resep Indonesia", func=rag_context_response, description="Mengambil beberapa resep sebagai referensi untuk menjawab pertanyaan.")
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True
    )
    return agent
