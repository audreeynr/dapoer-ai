import pandas as pd
import re
import time
import google.generativeai as genai

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

# Format hasil masakan
def format_recipe(row):
    # Pisah bahan berdasarkan newline, --, atau koma
    bahan_raw = re.split(r'\n|--|,', row['Ingredients'])
    bahan_list = [b.strip().capitalize() for b in bahan_raw if b.strip()]
    bahan_md = "\n".join([f"- {b}" for b in bahan_list])

    langkah_md = row['Steps'].strip()
    return f"""🍽 {row['Title']}

**Bahan-bahan:**  
{bahan_md}

**Langkah Memasak:**  
{langkah_md}"""

# Fungsi utama untuk menangani pertanyaan user
def handle_user_query(prompt, model):
    prompt_lower = normalize_text(prompt)

    # Ekstraksi keyword dari user prompt
    def extract_bahan_keywords(prompt_lower):
        stopwords = {"masakan", "apa", "saja", "yang", "bisa", "dibuat", "dari", "menggunakan", "bahan", "resep",
                     "dengan", "dan", "dong", "pengen", "ingin", "saya", "mau", "masak"}
        metode_masak = {"goreng", "panggang", "rebus", "kukus", "bakar", "tumis"}
        return [w for w in prompt_lower.split() if w not in stopwords and w not in metode_masak and len(w) > 2]

    # === Tool 1: Cari berdasarkan nama/judul masakan ===
    judul_keywords = extract_bahan_keywords(prompt_lower)
    match_title = df_cleaned[df_cleaned['Title_Normalized'].apply(
        lambda x: any(k in x for k in judul_keywords)
    )]
    if not match_title.empty:
        return format_recipe(match_title.iloc[0])

    # === Tool 2: Cari berdasarkan metode memasak ===
    for metode in ['goreng', 'panggang', 'rebus', 'kukus', 'bakar', 'tumis']:
        if metode in prompt_lower:
            cocok = df_cleaned[df_cleaned['Steps_Normalized'].str.contains(metode)]
            if not cocok.empty:
                hasil = cocok.head(5)['Title'].tolist()
                return f"Masakan yang dimasak dengan cara {metode}:\n- " + "\n- ".join(hasil)

    # === Tool 3: Rekomendasi masakan mudah (untuk pemula) ===
    if "mudah" in prompt_lower or "pemula" in prompt_lower or "gampang" in prompt_lower:
        hasil = df_cleaned[df_cleaned['Steps'].str.len() < 300].head(5)['Title'].tolist()
        return "Rekomendasi masakan mudah untuk pemula:\n- " + "\n- ".join(hasil)

    # === Tool 4: Cari berdasarkan bahan (di akhir supaya tidak mendominasi) ===
    bahan_keywords = extract_bahan_keywords(prompt_lower)
    if bahan_keywords:
        mask = df_cleaned['Ingredients_Normalized'].apply(
            lambda x: all(k in x for k in bahan_keywords)
        )
        match_bahan = df_cleaned[mask]
        if not match_bahan.empty:
            hasil = match_bahan.head(5)['Title'].tolist()
            return "Masakan yang menggunakan bahan tersebut:\n- " + "\n- ".join(hasil)

    # === Tool 5: RAG-like fallback ===
    try:
        keywords = [w for w in prompt_lower.split() if len(w) > 3]
        filter_mask = (
            df_cleaned['Title_Normalized'].apply(lambda x: any(k in x for k in keywords)) |
            df_cleaned['Ingredients_Normalized'].apply(lambda x: any(k in x for k in keywords)) |
            df_cleaned['Steps_Normalized'].apply(lambda x: any(k in x for k in keywords))
        )
        filtered_df = df_cleaned[filter_mask]
        rag_df = filtered_df.sample(5) if not filtered_df.empty else df_cleaned.sample(5)

        cache_buster = str(time.time())
        docs = "\n\n".join([
            f"{row['Title']}:\nBahan: {row['Ingredients']}\nLangkah: {row['Steps']}"
            for _, row in rag_df.iterrows()
        ])

        full_prompt = f"""
[session={cache_buster}]
Berikut beberapa resep masakan Indonesia:

{docs}

Gunakan referensi di atas untuk menjawab pertanyaan berikut:
{prompt}
"""
        response = model.generate_content(full_prompt)
        return response.text

    except Exception as e:
        return "Maaf, terjadi kesalahan saat memproses permintaanmu."
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
    return f"""🍽 {row['Title']}

**Bahan-bahan:**  
{bahan_md}

**Langkah Memasak:**  
{langkah_md}"""

# === Fungsi utama ===
def handle_user_query(prompt, model):
    prompt_lower = normalize_text(prompt)

    # Ekstrak keyword dari prompt (digunakan untuk beberapa tool)
    def extract_bahan_keywords(prompt_lower):
        stopwords = {
            "masakan", "apa", "saja", "yang", "bisa", "dibuat", "dari", "menggunakan",
            "bahan", "resep", "dengan", "dan", "dong", "pengen", "ingin", "saya", "mau", "masak", "pengin"
        }
        metode_masak = {"goreng", "panggang", "rebus", "kukus", "bakar", "tumis"}
        return [w for w in prompt_lower.split() if w not in stopwords and w not in metode_masak and len(w) > 2]

    judul_keywords = extract_bahan_keywords(prompt_lower)

    # === Tool 1: Cari berdasarkan judul ===
    match_title = df_cleaned[df_cleaned['Title_Normalized'].apply(
        lambda x: any(k in x for k in judul_keywords)
    )]
    if not match_title.empty:
        return format_recipe(match_title.iloc[0])

    # === Tool 2: Deteksi metode memasak ===
    for metode in ['goreng', 'panggang', 'rebus', 'kukus', 'bakar', 'tumis']:
        if metode in prompt_lower:
            cocok = df_cleaned[df_cleaned['Steps_Normalized'].str.contains(metode)]
            if not cocok.empty:
                hasil = cocok.head(5)['Title'].tolist()
                return f"Masakan yang dimasak dengan cara {metode}:\n- " + "\n- ".join(hasil)

    # === Tool 3: Rekomendasi masakan mudah (heuristik langkah pendek) ===
    if "mudah" in prompt_lower or "pemula" in prompt_lower or "gampang" in prompt_lower:
        hasil = df_cleaned[df_cleaned['Steps'].str.len() < 300].head(5)['Title'].tolist()
        return "Rekomendasi masakan mudah untuk pemula:\n- " + "\n- ".join(hasil)

    # === Tool 4: Cari berdasarkan bahan ===
    bahan_keywords = extract_bahan_keywords(prompt_lower)
    if bahan_keywords:
        mask = df_cleaned['Ingredients_Normalized'].apply(
            lambda x: all(k in x for k in bahan_keywords)
        )
        match_bahan = df_cleaned[mask]
        if not match_bahan.empty:
            hasil = match_bahan.head(5)['Title'].tolist()
            return "Masakan yang menggunakan bahan tersebut:\n- " + "\n- ".join(hasil)

    # === Tool 5: RAG-like fallback ===
    try:
        keywords = [w for w in prompt_lower.split() if len(w) > 3]
        filter_mask = (
            df_cleaned['Title_Normalized'].apply(lambda x: any(k in x for k in keywords)) |
            df_cleaned['Ingredients_Normalized'].apply(lambda x: any(k in x for k in keywords)) |
            df_cleaned['Steps_Normalized'].apply(lambda x: any(k in x for k in keywords))
        )
        filtered_df = df_cleaned[filter_mask]
        rag_df = filtered_df.sample(5) if not filtered_df.empty else df_cleaned.sample(5)

        cache_buster = str(time.time())
        docs = "\n\n".join([
            f"{row['Title']}:\nBahan: {row['Ingredients']}\nLangkah: {row['Steps']}"
            for _, row in rag_df.iterrows()
        ])

        full_prompt = f"""
[session={cache_buster}]
Berikut beberapa resep masakan Indonesia:

{docs}

Gunakan referensi di atas untuk menjawab pertanyaan berikut:
{prompt}
"""
        response = model.generate_content(full_prompt)
        return response.text

    except Exception as e:
        return "Maaf, terjadi kesalahan saat memproses permintaanmu."
