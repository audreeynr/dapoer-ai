import pandas as pd
import re
import time
import google.generativeai as genai

# === Load dan bersihkan data ===
CSV_FILE_PATH = 'https://raw.githubusercontent.com/valengrcla/celerates/refs/heads/main/Indonesian_Food_Recipes.csv'
df = pd.read_csv(CSV_FILE_PATH)
df_cleaned = df.dropna(subset=['Title', 'Ingredients', 'Steps']).drop_duplicates()

# === Normalisasi teks ===
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

# === Format hasil masakan ===
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

# === Tools (fungsi-fungsi pencarian) ===

# Tool 1: Cari berdasarkan nama/judul masakan
def search_by_title(prompt):
    prompt_lower = normalize_text(prompt)
    match = df_cleaned[df_cleaned['Title_Normalized'].str.contains(prompt_lower)]
    if not match.empty:
        return format_recipe(match.iloc[0])
    return "Tidak ditemukan resep dengan nama tersebut."

# Tool 2: Cari berdasarkan bahan
def search_by_ingredients(prompt):
    prompt_lower = normalize_text(prompt)
    match = df_cleaned[df_cleaned['Ingredients_Normalized'].str.contains(prompt_lower)]
    if not match.empty:
        hasil = match.head(5)['Title'].tolist()
        return "Masakan yang menggunakan bahan tersebut:\n- " + "\n- ".join(hasil)
    return "Tidak ditemukan masakan dengan bahan tersebut."

# Tool 3: Cari berdasarkan metode masak
def search_by_method(prompt):
    prompt_lower = normalize_text(prompt)
    for metode in ['goreng', 'panggang', 'rebus', 'kukus']:
        if metode in prompt_lower:
            match = df_cleaned[df_cleaned['Steps_Normalized'].str.contains(metode)]
            if not match.empty:
                hasil = match.head(5)['Title'].tolist()
                return f"Masakan yang dimasak dengan cara {metode}:\n- " + "\n- ".join(hasil)
    return "Tidak ditemukan metode memasak yang sesuai."

# Tool 4: Cari berdasarkan tingkat kesulitan
def search_easy_recipes(prompt):
    prompt_lower = normalize_text(prompt)
    if "mudah" in prompt_lower or "pemula" in prompt_lower:
        hasil = df_cleaned[df_cleaned['Steps'].str.len() < 300].head(5)['Title'].tolist()
        return "Rekomendasi masakan mudah:\n- " + "\n- ".join(hasil)
    return "Tidak ada indikasi tingkat kesulitan ditemukan dalam prompt."

# Tool 5: RAG-style contextual answer
def answer_with_context(prompt):
    prompt_lower = normalize_text(prompt)
    docs = "\n\n".join([
        f"{row['Title']}:\nBahan: {row['Ingredients']}\nLangkah: {row['Steps']}"
        for _, row in df_cleaned.sample(5, random_state=42).iterrows()
    ])
    full_prompt = f"""
Berikut beberapa resep masakan Indonesia:

{docs}

Gunakan referensi di atas untuk menjawab pertanyaan berikut:
{prompt}
"""
    response = genai.GenerativeModel('gemini-pro').generate_content(full_prompt)
    return response.text

# === Komposisi utama: satu fungsi gabungan ===
def handle_user_query(prompt, model):
    for tool_func in [
        search_by_title,
        search_by_method,
        search_easy_recipes,
        search_by_ingredients,
        answer_with_context,
    ]:
        try:
            result = tool_func(prompt)
            if result and "tidak ditemukan" not in result.lower():
                return result
        except:
            continue
    return "Maaf, saya tidak menemukan resep yang cocok."

# === Daftarkan Tools Langchain (minimal 5) ===
from langchain.agents import Tool

tools = [
    Tool(
        name="CariBerdasarkanJudul",
        func=search_by_title,
        description="Cari resep berdasarkan nama atau judul masakan"
    ),
    Tool(
        name="CariBerdasarkanBahan",
        func=search_by_ingredients,
        description="Temukan resep berdasarkan bahan yang disebutkan"
    ),
    Tool(
        name="CariBerdasarkanMetodeMasak",
        func=search_by_method,
        description="Temukan masakan berdasarkan metode memasak seperti goreng, kukus, rebus"
    ),
    Tool(
        name="CariResepMudah",
        func=search_easy_recipes,
        description="Berikan rekomendasi masakan mudah untuk pemula"
    ),
    Tool(
        name="JawabDenganContext",
        func=answer_with_context,
        description="Gunakan beberapa resep acak sebagai referensi dan jawab pakai RAG-like style"
    ),
]
