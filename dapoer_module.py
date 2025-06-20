# dapoer_module.py
import pandas as pd
import re

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/valengrcla/celerates/refs/heads/main/Indonesian_Food_Recipes.csv")
df.fillna("", inplace=True)

# Format satu resep sebagai string
def format_recipe(row):
    return f"""
**{row['Nama']}**
Deskripsi: {row['Deskripsi']}
Bahan: {row['Bahan']}
Langkah: {row['Langkah']}
Tingkat Kesulitan: {row['Tingkat_Kesulitan']}
Cara Masak: {row['Cara_Masak']}
"""

# Fungsi utama
def handle_user_query(prompt, model):
    prompt_lower = prompt.lower()

    # Tool 1: Cari berdasarkan nama
    match_name = df[df["Nama"].str.lower().str.contains(prompt_lower)]
    if not match_name.empty:
        result = match_name.iloc[0]
        return format_recipe(result)

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

    # Tool 5: Prompt ke Gemini (pseudo-RAG)
    top_docs = "\n\n".join([
        f"{row['Nama']}:\n{row['Deskripsi']}" for _, row in df.head(10).iterrows()
    ])

    full_prompt = f"""
Berikut beberapa deskripsi masakan Indonesia:
{top_docs}

Berdasarkan informasi di atas, jawab pertanyaan ini:
{prompt}
"""

    response = model.generate_content(full_prompt)
    return response.text
