# dapoer_module.py

import pandas as pd
import re
import random
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType

# Load dan bersihkan data
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

# Format hasil masakan
def format_recipe(row):
    return f"""🍽️ **{row['Title']}**

**Bahan-bahan:**  
{row['Ingredients']}

**Langkah Memasak:**  
{row['Steps']}"""

# Tool 1: Cari berdasarkan judul
def search_by_title(query: str) -> str:
    query_norm = normalize_text(query)
    result = df_cleaned[df_cleaned['Title_Normalized'].str.contains(query_norm)]
    if not result.empty:
        return format_recipe(result.iloc[0])
    return "Tidak ditemukan masakan dengan judul tersebut."

# Tool 2: Cari berdasarkan bahan
def search_by_ingredient(query: str) -> str:
    query_norm = normalize_text(query)
    result = df_cleaned[df_cleaned['Ingredients_Normalized'].str.contains(query_norm)]
    if not result.empty:
        titles = result.head(5)['Title'].tolist()
        return "Masakan yang menggunakan bahan tersebut:\n- " + "\n- ".join(titles)
    return "Tidak ditemukan masakan dengan bahan tersebut."

# Tool 3: Cari berdasarkan metode masak
def search_by_method(query: str) -> str:
    query_norm = normalize_text(query)
    for method in ['goreng', 'rebus', 'panggang', 'kukus']:
        if method in query_norm:
            result = df_cleaned[df_cleaned['Steps_Normalized'].str.contains(method)]
            if not result.empty:
                titles = result.head(5)['Title'].tolist()
                return f"Masakan yang dimasak dengan cara {method}:\n- " + "\n- ".join(titles)
    return "Tidak ditemukan metode memasak yang sesuai."

# Tool 4: Filter masakan mudah
def search_easy_recipes(_: str) -> str:
    result = df_cleaned[df_cleaned['Steps'].str.len() < 300].head(5)['Title'].tolist()
    return "Rekomendasi masakan mudah:\n- " + "\n- ".join(result)

# Tool 5: RAG-like - 5 resep acak sebagai referensi
def rag_response(query: str) -> str:
    samples = df_cleaned.sample(5, random_state=random.randint(1, 999))
    context = "\n\n".join([
        f"{row['Title']}:\nBahan: {row['Ingredients']}\nLangkah: {row['Steps']}"
        for _, row in samples.iterrows()
    ])
    return f"""
Berikut referensi resep masakan:

{context}

Pertanyaan: {query}
Silakan jawab berdasarkan referensi di atas.
""".strip()

# Fungsi utama agent handler
def handle_user_query(prompt, model):
    llm = ChatGoogleGenerativeAI(model=model, temperature=0.5)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    tools = [
        Tool(name="CariJudul", func=search_by_title, description="Gunakan untuk mencari resep berdasarkan nama masakan"),
        Tool(name="CariBahan", func=search_by_ingredient, description="Gunakan untuk mencari masakan berdasarkan bahan"),
        Tool(name="CariMetode", func=search_by_method, description="Gunakan untuk mencari masakan berdasarkan metode memasak"),
        Tool(name="ResepMudah", func=search_easy_recipes, description="Gunakan jika pengguna ingin resep mudah"),
        Tool(name="ResepRAG", func=rag_response, description="Gunakan untuk menjawab pertanyaan umum tentang masakan menggunakan referensi acak")
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=False
    )

    return agent.run(prompt)
