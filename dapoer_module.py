import pandas as pd
import re
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
    bahan_raw = re.split(r'\n|--|,', row['Ingredients'])
    bahan_list = [b.strip().capitalize() for b in bahan_raw if b.strip()]
    bahan_md = "\n".join([f"- {b}" for b in bahan_list])
    langkah_md = row['Steps'].strip()

    return f"""🍽 *{row['Title']}*

*Bahan-bahan:*  
{bahan_md}

*Langkah Memasak:*  
{langkah_md}"""

# Ekstrak bahan dari prompt user (dengan asumsi ada kata kunci "dari", "dengan", "menggunakan")
def extract_ingredients_from_prompt(prompt):
    keywords = ['dari', 'dengan', 'menggunakan']
    for key in keywords:
        if key in prompt:
            potongan = prompt.split(key, 1)[-1]
            potongan = normalize_text(potongan)
            # Ambil kata-kata yang kemungkinan bahan (pakai koma/atau)
            candidates = re.split(r',|atau|dan', potongan)
            return [c.strip() for c in candidates if c.strip()]
    return []

# Fungsi utama
def handle_user_query(prompt, model):
    prompt_lower = normalize_text(prompt)

    # Tool 1: Berdasarkan nama masakan
    match_title = df_cleaned[df_cleaned['Title_Normalized'].str.contains(prompt_lower)]
    if not match_title.empty:
        return format_recipe(match_title.iloc[0])

    # Tool 2: Berdasarkan bahan eksplisit
    extracted_ingredients = extract_ingredients_from_prompt(prompt_lower)
    if extracted_ingredients:
        hasil = []
        for ing in extracted_ingredients:
            cocok = df_cleaned[df_cleaned['Ingredients_Normalized'].str.contains(ing)]
            hasil += cocok['Title'].tolist()
        if hasil:
            hasil_unik = list(dict.fromkeys(hasil))[:5]
            return f"Masakan yang bisa dibuat dari {' / '.join(extracted_ingredients)}:\n- " + "\n- ".join(hasil_unik)

    # Tool 2b: Fallback bahan jika langsung menyebut bahan saja
    match_bahan = df_cleaned[df_cleaned['Ingredients_Normalized'].str.contains(prompt_lower)]
    if not match_bahan.empty:
        hasil = match_bahan.head(5)['Title'].tolist()
        return "Masakan yang menggunakan bahan tersebut:\n- " + "\n- ".join(hasil)

    # Tool 3: Berdasarkan metode memasak
    for metode in ['goreng', 'panggang', 'rebus', 'kukus']:
        if metode in prompt_lower:
            cocok = df_cleaned[df_cleaned['Steps_Normalized'].str.contains(metode)]
            if not cocok.empty:
                hasil = cocok.head(5)['Title'].tolist()
                return f"Masakan yang dimasak dengan cara {metode}:\n- " + "\n- ".join(hasil)

    # Tool 4: Filter kesulitan
    if "mudah" in prompt_lower or "pemula" in prompt_lower:
        hasil = df_cleaned[df_cleaned['Steps'].str.len() < 300].head(5)['Title'].tolist()
        return "Rekomendasi masakan mudah:\n- " + "\n- ".join(hasil)

    # Tool 5: RAG-like
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
