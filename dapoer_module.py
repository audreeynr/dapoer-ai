# dapoer_module.py
import pandas as pd
import re
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/valengrcla/celerates/refs/heads/main/Indonesian_Food_Recipes.csv")

# Membersihkan kolom jika ada null
df.fillna("", inplace=True)

# Persiapan RAG
def create_retriever():
    docs_df = df[["Nama", "Deskripsi"]].copy()
    docs_df["text"] = docs_df["Nama"] + "\n" + docs_df["Deskripsi"]

    loader = DataFrameLoader(docs_df, page_content_column="text")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(split_docs, embedding)

    return vectordb.as_retriever()

retriever = create_retriever()

# Fungsi utama untuk handle query user
def handle_user_query(prompt, model):
    prompt_lower = prompt.lower()

    # Tool 1: Cari berdasarkan nama
    match_name = df[df["Nama"].str.lower().str.contains(prompt_lower)]
    if not match_name.empty:
        result = match_name.iloc[0]
        return f"**{result['Nama']}**\n\nDeskripsi: {result['Deskripsi']}\n\nBahan: {result['Bahan']}\n\nLangkah: {result['Langkah']}\n\nTingkat Kesulitan: {result['Tingkat_Kesulitan']}\nCara Masak: {result['Cara_Masak']}"

    # Tool 2: Filter berdasarkan bahan
    bahan_match = df[df["Bahan"].str.lower().str.contains(prompt_lower)]
    if not bahan_match.empty:
        nama_masakan = ", ".join(bahan_match["Nama"].tolist()[:5])
        return f"Beberapa masakan yang menggunakan **{prompt}** adalah: {nama_masakan}"

    # Tool 3: Sort berdasarkan tingkat kesulitan
    if "termudah" in prompt_lower or "mudah" in prompt_lower:
        mudah = df[df["Tingkat_Kesulitan"].str.lower() == "mudah"].head(5)
        return "Masakan termudah:\n- " + "\n- ".join(mudah["Nama"].tolist())

    # Tool 4: Sort berdasarkan cara masak
    for method in ["rebus", "panggang", "goreng", "kukus"]:
        if method in prompt_lower:
            filtered = df[df["Cara_Masak"].str.lower().str.contains(method)].head(5)
            return f"Berikut beberapa masakan yang dimasak dengan cara {method}:\n- " + "\n- ".join(filtered["Nama"].tolist())

    # Tool 5: RAG jika tidak cocok ke semua tools di atas
    chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever)
    result = chain.run(prompt)
    return result
