# dapoer_module.py
import pandas as pd
import re
import google.generativeai as genai

# Load dan bersihkan data
CSV_FILE_PATH = 'https://raw.githubusercontent.com/audreeynr/dapoer-ai/refs/heads/main/data/Indonesian_Food_Recipes.csv'
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
    # Normalisasi bahan: pisah berdasarkan newline, '--', atau koma
    bahan_raw = re.split(r'\n|--|,', row['Ingredients'])
    bahan_list = [b.strip().capitalize() for b in bahan_raw if b.strip()]
    bahan_md = "\n".join([f"- {b}" for b in bahan_list])

    # Langkah memasak langsung tampilkan tanpa tambahan bullet
    langkah_md = row['Steps'].strip()

    return f"""🍽 *{row['Title']}*

*Bahan-bahan:*  
{bahan_md}

*Langkah Memasak:*  
{langkah_md}"""

# Fungsi utama untuk handle pertanyaan
def handle_user_query(prompt, model):
    prompt_lower = normalize_text(prompt)

    # Tool 1: Cari berdasarkan nama masakan
    match_title = df_cleaned[df_cleaned['Title_Normalized'].str.contains(prompt_lower)]
    if not match_title.empty:
        return format_recipe(match_title.iloc[0])

    # Tool 2: Cari berdasarkan bahan
    match_bahan = df_cleaned[df_cleaned['Ingredients_Normalized'].str.contains(prompt_lower)]
    if not match_bahan.empty:
        hasil = match_bahan.head(5)['Title'].tolist()
        return "Masakan yang menggunakan bahan tersebut:\n- " + "\n- ".join(hasil)

    # Tool 3: Cari berdasarkan metode masak
    for metode in ['goreng', 'panggang', 'rebus', 'kukus']:
        if metode in prompt_lower:
            cocok = df_cleaned[df_cleaned['Steps_Normalized'].str.contains(metode)]
            if not cocok.empty:
                hasil = cocok.head(5)['Title'].tolist()
                return f"Masakan yang dimasak dengan cara {metode}:\n- " + "\n- ".join(hasil)

    # Tool 4: Filter kesulitan (pakai heuristik kata di steps)
    if "mudah" in prompt_lower or "pemula" in prompt_lower:
        hasil = df_cleaned[df_cleaned['Steps'].str.len() < 300].head(5)['Title'].tolist()
        return "Rekomendasi masakan mudah:\n- " + "\n- ".join(hasil)

    # Tool 5: RAG-like: Ambil 5 resep acak sebagai context
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
    response = model.generate_content(full_prompt)
    return response.text
