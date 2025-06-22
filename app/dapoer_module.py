# dapoer_module.py
import pandas as pd
import re
import google.generativeai as genai

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
    # Normalisasi bahan: pisah berdasarkan newline, '--', atau koma
    bahan_raw = re.split(r'\n|--|,', row['Ingredients'])
    bahan_list = [b.strip().capitalize() for b in bahan_raw if b.strip()]
    bahan_md = "\n".join([f"- {b}" for b in bahan_list])

    # Langkah memasak langsung tampilkan tanpa tambahan bullet
    langkah_md = row['Steps'].strip()

    return f"""🍽 {row['Title']}

Bahan-bahan:  
{bahan_md}

Langkah Memasak:  
{langkah_md}"""

# Fungsi utama untuk handle pertanyaan
def handle_user_query(prompt, model):
    prompt_lower = normalize_text(prompt)

    # Tool 1: Cari berdasarkan nama masakan
    match_title = df_cleaned[df_cleaned['Title_Normalized'].str.contains(prompt_lower)]
    if not match_title.empty:
        return format_recipe(match_title.iloc[0])

    # Ekstraksi keyword bahan dari prompt
    def extract_bahan_keywords(prompt_lower):
        stopwords = {"masakan", "apa", "saja", "yang", "bisa", "dibuat", "dari", "menggunakan", "bahan", "resep"}
        kata_kunci = [w for w in prompt_lower.split() if w not in stopwords and len(w) > 2]
        return kata_kunci
    
    # Tool 2: Cari berdasarkan bahan (lebih fleksibel)
    bahan_keywords = extract_bahan_keywords(prompt_lower)
    if bahan_keywords:
        mask = df_cleaned['Ingredients_Normalized'].apply(lambda x: all(k in x for k in bahan_keywords))
        match_bahan = df_cleaned[mask]
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

    # Tool 5: RAG-like - Ambil context relevan (bukan hanya random)
    try:
        # Ambil keyword penting dari prompt
        keywords = [w for w in prompt_lower.split() if len(w) > 3]
        filter_mask = df_cleaned['Title_Normalized'].apply(lambda x: any(k in x for k in keywords)) | \
                      df_cleaned['Ingredients_Normalized'].apply(lambda x: any(k in x for k in keywords)) | \
                      df_cleaned['Steps_Normalized'].apply(lambda x: any(k in x for k in keywords))

        filtered_df = df_cleaned[filter_mask]

        # Jika relevan cukup banyak, ambil 5 dari yang relevan, jika tidak, fallback ke random
        rag_df = filtered_df.sample(5) if len(filtered_df) >= 5 else df_cleaned.sample(5)

        # Tambahkan penanda waktu/acak untuk hindari cache LLM
        import time
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
        return "Maaf, terjadi kesalahan saat mengambil data resep."
