def handle_user_query(prompt, model):
    prompt_lower = normalize_text(prompt)

    # Bersihkan kata-kata umum seperti 'resep', 'masakan'
    cleaned_prompt = re.sub(r'\b(resep|masakan|cara membuat)\b', '', prompt_lower).strip()

    # Tool 1: Cari berdasarkan nama masakan (judul)
    try:
        match_title = df_cleaned[df_cleaned['Title_Normalized'].str.contains(cleaned_prompt, na=False)]
        if not match_title.empty:
            return format_recipe(match_title.iloc[0])
    except Exception as e:
        pass  # skip tool ini jika error

    # Tool 2: Cari berdasarkan bahan (keyword token match)
    try:
        bahan_keywords = cleaned_prompt.split()
        cocok_bahan = df_cleaned[df_cleaned['Ingredients_Normalized'].apply(
            lambda x: any(k in x for k in bahan_keywords)
        )]
        if not cocok_bahan.empty:
            hasil = cocok_bahan.head(5)['Title'].tolist()
            return "Masakan yang menggunakan bahan tersebut:\n- " + "\n- ".join(hasil)
    except Exception as e:
        pass

    # Tool 3: Cari berdasarkan metode masak
    try:
        for metode in ['goreng', 'panggang', 'rebus', 'kukus']:
            if metode in prompt_lower:
                cocok = df_cleaned[df_cleaned['Steps_Normalized'].str.contains(metode, na=False)]
                if not cocok.empty:
                    hasil = cocok.head(5)['Title'].tolist()
                    return f"Masakan yang dimasak dengan cara {metode}:\n- " + "\n- ".join(hasil)
    except Exception as e:
        pass

    # Tool 4: Filter masakan mudah
    try:
        if "mudah" in prompt_lower or "pemula" in prompt_lower:
            hasil = df_cleaned[df_cleaned['Steps'].str.len() < 300].head(5)['Title'].tolist()
            return "Rekomendasi masakan mudah:\n- " + "\n- ".join(hasil)
    except Exception as e:
        pass

    # Tool 5: RAG-like (ambil 5 resep acak untuk referensi LLM)
    try:
        docs = "\n\n".join([
            f"{row['Title']}:\nBahan: {row['Ingredients']}\nLangkah: {row['Steps']}"
            for _, row in df_cleaned.sample(5).iterrows()
        ])
        full_prompt = f"""
Berikut beberapa resep masakan Indonesia:

{docs}

Gunakan referensi di atas untuk menjawab pertanyaan berikut:
{prompt}
"""
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return "Maaf, terjadi kesalahan saat mengambil data resep."
