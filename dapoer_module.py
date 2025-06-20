# dapoer_module.py
import pandas as pd
import re
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load dan bersihkan data
CSV_FILE_PATH = 'https://raw.githubusercontent.com/valengrcla/celerates/refs/heads/main/Indonesian_Food_Recipes.csv'
df = pd.read_csv(CSV_FILE_PATH)
df_cleaned = df.dropna(subset=['Title', 'Ingredients', 'Steps']).drop_duplicates()

# Normalisasi teks
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
    return f"""🍽 **{row['Title']}**\n\n**Bahan-bahan:**\n{bahan_md}\n\n**Langkah Memasak:**\n{langkah_md}"""

# Tool 1 - Cari resep berdasarkan judul
def cari_berdasarkan_judul(query):
    query_norm = normalize_text(query)
    hasil = df_cleaned[df_cleaned['Title_Normalized'].str.contains(query_norm)]
    if not hasil.empty:
        return format_recipe(hasil.iloc[0])
    return "Resep tidak ditemukan berdasarkan judul."

# Tool 2 - Cari berdasarkan bahan
def cari_berdasarkan_bahan(query):
    query_norm = normalize_text(query)
    hasil = df_cleaned[df_cleaned['Ingredients_Normalized'].str.contains(query_norm)]
    if not hasil.empty:
        judul_list = hasil.head(5)['Title'].tolist()
        return "Masakan yang menggunakan bahan tersebut:\n- " + "\n- ".join(judul_list)
    return "Tidak ditemukan resep dengan bahan tersebut."

# Tool 3 - Cari berdasarkan metode masak
def cari_berdasarkan_metode(query):
    query_norm = normalize_text(query)
    metode_list = ['goreng', 'panggang', 'rebus', 'kukus']
    cocok = [m for m in metode_list if m in query_norm]
    if cocok:
        hasil = df_cleaned[df_cleaned['Steps_Normalized'].str.contains(cocok[0])]
        if not hasil.empty:
            judul_list = hasil.head(5)['Title'].tolist()
            return f"Masakan yang dimasak dengan cara {cocok[0]}:\n- " + "\n- ".join(judul_list)
    return "Tidak ditemukan metode memasak yang cocok."

# Tool 4 - Filter masakan mudah
def rekomendasi_masakan_mudah(_):
    hasil = df_cleaned[df_cleaned['Steps'].str.len() < 300].head(5)['Title'].tolist()
    return "Rekomendasi masakan mudah:\n- " + "\n- ".join(hasil)

# Tool 5 - RAG-like: Ambil 5 resep sebagai context
def rag_lookup(query):
    docs = "\n\n".join([
        f"{row['Title']}:\nBahan: {row['Ingredients']}\nLangkah: {row['Steps']}"
        for _, row in df_cleaned.sample(5, random_state=42).iterrows()
    ])
    template = PromptTemplate.from_template(
        """Berikut adalah kumpulan resep masakan:

{docs}

Jawablah pertanyaan pengguna dengan informasi dari atas:
{query}
""")
    prompt = template.format(docs=docs, query=query)
    llm_local = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    return llm_local.invoke(prompt).content

# Inisialisasi Agent Langchain
def init_agent(api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    # Chat model Langchain
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)

    # Memory Langchain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Tools
    tools = [
        Tool(name="CariJudulResep", func=cari_berdasarkan_judul, description="Cari resep berdasarkan nama masakan."),
        Tool(name="CariBahan", func=cari_berdasarkan_bahan, description="Cari resep menggunakan bahan tertentu."),
        Tool(name="CariMetode", func=cari_berdasarkan_metode, description="Cari resep berdasarkan metode memasak seperti goreng, kukus, dll."),
        Tool(name="ResepMudah", func=rekomendasi_masakan_mudah, description="Tampilkan resep yang mudah dan cocok untuk pemula."),
        Tool(name="ResepRAG", func=rag_lookup, description="Jawab pertanyaan resep berdasarkan contoh dari beberapa resep.")
    ]

    # Agent Langchain
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent="chat-conversational-react-description",
        verbose=False,
        memory=memory,
        handle_parsing_errors=True
    )

    return agent_executor
