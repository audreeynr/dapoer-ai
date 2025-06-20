import pandas as pd
import re
import google.generativeai as genai
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models import BaseLanguageModel
from langchain_community.chat_models import ChatGoogleGenerativeAI

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

# Format output
def format_recipe(row):
    return f"""🍽️ **{row['Title']}**

**Bahan-bahan:**  
{row['Ingredients']}

**Langkah Memasak:**  
{row['Steps']}"""

# Tool 1: Pencarian berdasarkan judul
def tool_judul(prompt):
    prompt_lower = normalize_text(prompt)
    match = df_cleaned[df_cleaned['Title_Normalized'].str.contains(prompt_lower)]
    if not match.empty:
        return format_recipe(match.iloc[0])
    return "Tidak ditemukan resep dengan nama tersebut."

# Tool 2: Berdasarkan bahan
def tool_bahan(prompt):
    prompt_lower = normalize_text(prompt)
    match = df_cleaned[df_cleaned['Ingredients_Normalized'].str.contains(prompt_lower)]
    if not match.empty:
        hasil = match.head(5)['Title'].tolist()
        return "Masakan dengan bahan tersebut:\n- " + "\n- ".join(hasil)
    return "Tidak ditemukan resep dengan bahan tersebut."

# Tool 3: Berdasarkan metode masak
def tool_metode(prompt):
    prompt_lower = normalize_text(prompt)
    for metode in ['goreng', 'panggang', 'rebus', 'kukus']:
        if metode in prompt_lower:
            cocok = df_cleaned[df_cleaned['Steps_Normalized'].str.contains(metode)]
            if not cocok.empty:
                hasil = cocok.head(5)['Title'].tolist()
                return f"Masakan dengan cara {metode}:\n- " + "\n- ".join(hasil)
    return "Tidak ditemukan metode masak yang cocok."

# Tool 4: Berdasarkan kesulitan
def tool_mudah(prompt):
    if "mudah" in normalize_text(prompt) or "pemula" in normalize_text(prompt):
        hasil = df_cleaned[df_cleaned['Steps'].str.len() < 300].head(5)['Title'].tolist()
        return "Rekomendasi masakan mudah:\n- " + "\n- ".join(hasil)
    return "Tidak ada filter kesulitan dikenali."

# Tool 5: RAG-like retrieval
def tool_rag(prompt):
    docs = "\n\n".join([
        f"{row['Title']}:\nBahan: {row['Ingredients']}\nLangkah: {row['Steps']}"
        for _, row in df_cleaned.sample(5, random_state=42).iterrows()
    ])
    return f"""
Berikut beberapa resep masakan Indonesia:\n\n{docs}\n\nGunakan referensi di atas untuk menjawab pertanyaan berikut:\n{prompt}
""".strip()

# Fungsi untuk membungkus tools dan agent
def initialize_dapoer_agent(api_key: str):
    genai.configure(api_key=api_key)
    llm: BaseLanguageModel = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)

    tools = [
        Tool(name="CariJudul", func=tool_judul, description="Cari resep berdasarkan nama masakan"),
        Tool(name="CariBahan", func=tool_bahan, description="Cari resep berdasarkan bahan"),
        Tool(name="CariMetode", func=tool_metode, description="Cari resep berdasarkan metode masak"),
        Tool(name="ResepMudah", func=tool_mudah, description="Rekomendasi resep yang mudah"),
        Tool(name="RAGResep", func=tool_rag, description="Cari jawaban berdasarkan kumpulan resep (RAG)")
    ]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=False
    )
    return agent
