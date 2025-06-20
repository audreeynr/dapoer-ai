import pandas as pd
import re
from io import BytesIO
from fpdf import FPDF

# Load dan bersihkan data
df = pd.read_csv("https://raw.githubusercontent.com/valengrcla/celerates/refs/heads/main/Indonesian_Food_Recipes.csv")
df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
df.fillna("", inplace=True)

def format_recipe(row):
    return f"""
**{row['Nama']}**
Deskripsi: {row['Deskripsi']}
Bahan: {row['Bahan']}
Langkah: {row['Langkah']}
Tingkat Kesulitan: {row['Tingkat_Kesulitan']}
Cara Masak: {row['Cara_Masak']}
"""

# Fitur tambahan: Export ke PDF
def export_to_pdf(recipe_row):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 10, f"Nama: {recipe_row['Nama']}")
    pdf.multi_cell(0, 10, f"\nDeskripsi:\n{recipe_row['Deskripsi']}")
    pdf.multi_cell(0, 10, f"\nBahan:\n{recipe_row['Bahan']}")
    pdf.multi_cell(0, 10, f"\nLangkah:\n{recipe_row['Langkah']}")
    pdf.multi_cell(0, 10, f"\nTingkat Kesulitan: {recipe_row['Tingkat_Kesulitan']}")
    pdf.multi_cell(0, 10, f"Cara Masak: {recipe_row['Cara_Masak']}")

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# Fitur tambahan: Buat daftar belanja
def generate_shopping_list(nama_masakan):
    match = df[df["Nama"].str.lower() == nama_masakan.lower()]
    if not match.empty:
        bahan = match.iloc[0]["Bahan"]
        daftar = re.split(r",|\n|•|-", bahan)
        daftar = [item.strip(" •-") for item in daftar if item.strip()]
        return "🛒 Daftar Belanja:\n- " + "\n- ".join(daftar)
    return "Resep tidak ditemukan."

# Fungsi utama chatbot
def handle_user_query(prompt, model):
    prompt_lower = prompt.lower()

    # Tool 1: Cek apakah export ke PDF
    if "export" in prompt_lower and "pdf" in prompt_lower:
        for nama in df["Nama"]:
            if nama.lower() in prompt_lower:
                match = df[df["Nama"].str.lower() == nama.lower()]
                if not match.empty:
                    return export_to_pdf(match.iloc[0])
        return "Resep tidak ditemukan."

    # Tool 2: Cek apakah minta list belanja
    if "list belanja" in prompt_lower or "belanja" in prompt_lower:
        for nama in df["Nama"]:
            if nama.lower() in prompt_lower:
                return generate_shopping_list(nama)

    # Tool 3: Pencarian nama resep
    match_name = df[df["Nama"].str.lower().str.contains(prompt_lower)]
    if not match_name.empty:
        return format_recipe(match_name.iloc[0])

    # Tool 4: Bahan
    bahan_match = df[df["Bahan"].str.lower().str.contains(prompt_lower)]
    if not bahan_match.empty:
        return "Masakan dengan bahan tersebut:\n- " + "\n- ".join(bahan_match["Nama"].tolist()[:5])

    # Tool 5: Tingkat kesulitan
    if "mudah" in prompt_lower:
        mudah = df[df["Tingkat_Kesulitan"].str.lower() == "mudah"].head(5)
        return "Masakan termudah:\n- " + "\n- ".join(mudah["Nama"].tolist())

    # Tool 6: Cara masak
    for metode in ["rebus", "panggang", "goreng", "kukus"]:
        if metode in prompt_lower:
            filtered = df[df["Cara_Masak"].str.lower().str.contains(metode)].head(5)
            return f"Masakan dengan cara {metode}:\n- " + "\n- ".join(filtered["Nama"].tolist())

    # Tool 7: Gemini (fallback)
    context = "\n\n".join([
        f"{row['Nama']}:\n{row['Deskripsi']}" for _, row in df.head(10).iterrows()
    ])
    full_prompt = f"""
Berikut adalah ringkasan masakan Indonesia:
{context}

Pertanyaan: {prompt}
"""

    response = model.generate_content(full_prompt)
    return response.text
