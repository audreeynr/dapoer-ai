# dapoer_module.py
import pandas as pd
import re

from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory

### --- Load dan Normalisasi Data --- ###
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

### --- Format Resep --- ###
def format_recipe(row):
    bahan = [b.strip().capitalize() for b in re.split(r'\n|--|,', row['Ingredients']) if b.strip()]
    bahan_md = "\n".join([f"- {b}" for b in bahan])
    langkah_md = row['Steps'].strip()
    return f"""🍽 {row['Title']}\n\n**Bahan-bahan:**\n{bahan_md}\n\n**Langkah Memasak:**\n{langkah_md}"""

### --- Tool 1: Cari Berdasarkan Judul --- ###
def search_by_title(query):
    query_n = normalize_text(query)
    result = df_cleaned[df_cleaned['Title_Normalized'].str.contains(query_n)]
    if not result.empty:
        return format_recipe(result.iloc[0])
    return "❌ Resep tidak ditemukan berdasarkan judul."

### --- Tool 2: Cari Berdasarkan Bahan --- ###
def search_by_ingredients(query):
    stopwords = {"masakan", "apa", "saja", "yang", "bisa", "dibuat", "dari", "menggunakan", "bahan", "resep"}
    keywords = [w for w in normalize_text(query).split() if w not in stopwords and len(w) > 2]
    if not keywords:
        return "❌ Tidak ditemukan keyword bahan valid."

    matches = df_cleaned[df_cleaned['Ingredients_Normalized'].apply(lambda x: all(k in x for k in keywords))]
    if not matches.empty:
        titles = matches.head(5)['Title'].tolist()
        return "✅ Masakan dengan bahan tersebut:\n- " + "\n- ".join(titles)
    return "❌ Tidak ada resep dengan bahan tersebut."

### --- Tool 3: Cari Berdasarkan Metode Memasak --- ###
def search_by_method(query):
    methods = ['goreng', 'panggang', 'rebus', 'kukus']
    query_n = normalize_text(query)
    for m in methods:
        if m in query_n:
            match = df_cleaned[df_cleaned['Steps_Normalized'].str.contains(m)]
            if not match.empty:
                return f"✅ Masakan yang dimasak dengan cara {m}:\n- " + "\n- ".join(match.head(5)['Title'].tolist())
    return "❌ Tidak ditemukan metode memasak yang sesuai."

### --- Tool 4: Rekomendasi Masakan Mudah --- ###
def recommend_easy_recipes(query):
    if "mudah" in normalize_text(query) or "pemula" in normalize_text(query):
        easy = df_cleaned[df_cleaned['Steps'].str.len() < 300].head(5)
        return "🍳 Rekomendasi masakan mudah:\n- " + "\n- ".join(easy['Title'].tolist())
    return "❌ Tidak ada masakan mudah yang cocok."

### --- Tool 5: RAG dengan FAISS --- ###
def build_vectorstore(api_key):
    docs = [
        Document(page_content=f"Judul: {row['Title']}\nBahan: {row['Ingredients']}\nLangkah: {row['Steps']}")
        for _, row in df_cleaned.iterrows()
    ]
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key)
    return FAISS.from_documents(chunks, embeddings)

def rag_search(api_key, query):
    vs = build_vectorstore(api_key)
    retriever = vs.as_retriever()
    docs = retriever.get_relevant_documents(query)

    if not docs:
        fallback = df_cleaned.sample(3)
        return "🤷 Tidak ditemukan informasi relevan. Coba ini:\n\n" + "\n\n".join(
            [format_recipe(row) for _, row in fallback.iterrows()]
        )

    return "📚 Hasil pencarian:\n\n" + "\n\n".join([doc.page_content for doc in docs[:3]])

### --- Agent LangChain --- ###
def create_agent(api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.7
    )

    tools = [
        Tool(name="SearchByTitle", func=search_by_title, description="Cari resep berdasarkan judul."),
        Tool(name="SearchByIngredients", func=search_by_ingredients, description="Cari resep berdasarkan bahan."),
        Tool(name="SearchByMethod", func=search_by_method, description="Cari resep berdasarkan metode memasak."),
        Tool(name="RecommendEasyRecipes", func=recommend_easy_recipes, description="Rekomendasi resep mudah."),
        Tool(name="RAGSearch", func=lambda q: rag_search(api_key, q), description="Cari informasi resep menggunakan FAISS RAG.")
    ]

    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        memory=memory,
        verbose=False
    )

    return agent
