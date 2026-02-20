import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse, urlunparse
from io import BytesIO
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIGURAZIONE
# =========================
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

# =========================
# UI SETUP
# =========================
st.set_page_config(page_title="AI Redirect Mapper", layout="wide")
st.title("🎯 AI Redirect Mapper")

with st.sidebar:
    st.header("⚙️ Configurazione Strategica")
    openai_api_key = st.text_input("OpenAI key", type="password")
    
    st.markdown("---")
    st.subheader("📚 Logica SEO")
    st.info("""
    - **Solo HTML**: Filtra ed elabora solo pagine text/html.
    - **Silo Linguistico**: Se l'URL è spagnolo, cercherà match SOLO in spagnolo.
    - **Zero 404**: Se il match è incerto, punta alla Home della lingua corretta.
    """)
    
    threshold_primary = st.slider("Soglia Similarità Minima", 0.0, 1.0, 0.45, 
                                 help="Sotto questa soglia, per sicurezza, l'URL andrà alla Home di lingua.")

    st.markdown("---")
    st.subheader("🌐 Mappatura Domini/Lingue")
    domain_map_input = st.text_area(
        "Formato: dominio:lingua (uno per riga)", 
        value="bossong-befestigungssysteme.de:en\nbossong.es:es\nbossong.it:it",
        help="Esempio: bossong.de:en mappa i vecchi URL tedeschi sul pool inglese."
    )

# =========================
# FUNZIONI DI SUPPORTO
# =========================

def get_domain_map(input_text):
    d_map = {}
    if not input_text.strip(): return d_map
    for line in input_text.split('\n'):
        line = line.strip()
        if ":" in line:
            parts = line.split(':')
            if len(parts) >= 2:
                domain = parts[0].strip().lower().replace("www.", "")
                lang = parts[1].strip().lower()
                d_map[domain] = lang
    return d_map

def load_files_combined(files) -> pd.DataFrame:
    all_dfs = []
    for file in files:
        try:
            if file.name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(file, engine="openpyxl")
            else:
                for enc in ["utf-8", "ISO-8859-1", "cp1252"]:
                    try:
                        file.seek(0)
                        df = pd.read_csv(file, encoding=enc, low_memory=False)
                        break
                    except: continue
            if df is not None: all_dfs.append(df)
        except: continue
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None

def is_html_and_200(row, check_status=False):
    c_type = str(row.get("Content Type", "")).lower()
    is_html = "text/html" in c_type or "html" in c_type
    if check_status:
        status = str(row.get("Status Code", ""))
        return is_html and status == "200"
    return is_html

def detect_language(url: str, domain_mapping: dict) -> str:
    p = urlparse(url)
    domain = p.netloc.lower().replace("www.", "")
    path = p.path.lower()
    if domain in domain_mapping: return domain_mapping[domain]
    m = re.search(r"/([a-z]{2})(?:/|$)", path)
    if m: return m.group(1)
    if domain.endswith(".it"): return "it"
    if domain.endswith(".es"): return "es"
    return "it" # Default

def clean_seo_text(row: pd.Series) -> str:
    url = str(row.get("Address", ""))
    slug = urlparse(url).path.replace("/", " ").replace("-", " ").replace(".html", "")
    t = str(row.get("Title 1", ""))
    h1 = str(row.get("H1-1", ""))
    return f"SLUG: {slug} | TITLE: {t} | H1: {h1}".strip()[:7000]

# =========================
# CORE ENGINE
# =========================

col1, col2 = st.columns(2)
with col1: old_files = st.file_uploader("📂 Vecchio Sito", accept_multiple_files=True)
with col2: new_files = st.file_uploader("📂 Nuovo Sito", accept_multiple_files=True)

if old_files and new_files:
    d_mapping = get_domain_map(domain_map_input)
    df_old_raw = load_files_combined(old_files)
    df_new_raw = load_files_combined(new_files)

    if df_old_raw is not None and df_new_raw is not None:
        # FILTRO E RESET INDICI (Fondamentale per evitare IndexError)
        df_old = df_old_raw[df_old_raw.apply(lambda r: is_html_and_200(r, False), axis=1)].copy().reset_index(drop=True)
        df_new = df_new_raw[df_new_raw.apply(lambda r: is_html_and_200(r, True), axis=1)].copy().reset_index(drop=True)

        df_old["lang"] = df_old["Address"].apply(lambda x: detect_language(x, d_mapping))
        df_new["lang"] = df_new["Address"].apply(lambda x: detect_language(x, {}))

        st.success(f"Dati: {len(df_old)} sorgenti vs {len(df_new)} destinazioni HTML filtrate.")

        if st.button("🚀 AVVIA GENERAZIONE REDIRECT"):
            if not openai_api_key: st.error("Manca API Key"); st.stop()
            client = OpenAI(api_key=openai_api_key)
            status = st.status("Elaborazione...")

            # 1. Embedding
            status.write("🧠 Creazione vettori semantici...")
            old_texts = df_old.apply(clean_seo_text, axis=1).tolist()
            new_texts = df_new.apply(clean_seo_text, axis=1).tolist()
            
            def get_embeddings(txt_list):
                embs = []
                for i in range(0, len(txt_list), BATCH_SIZE):
                    res = client.embeddings.create(input=txt_list[i:i+BATCH_SIZE], model=EMBED_MODEL)
                    embs.extend([d.embedding for d in res.data])
                return embs

            emb_old = get_embeddings(old_texts)
            emb_new = get_embeddings(new_texts)
            sims = cosine_similarity(emb_old, emb_new)

            # 2. Matching con Progress Bar
            status.write("🔍 Applicazione regole SEO...")
            progress = st.progress(0, text="Inizio...")
            
            # Identificazione Home di Lingua
            home_pages = {}
            for l in df_new["lang"].unique():
                subset = df_new[df_new["lang"] == l]
                home_pages[l] = subset.loc[subset['Address'].str.len().idxmin()]['Address']

            results = []
            for i in range(len(df_old)):
                progress.progress((i + 1) / len(df_old), text=f"Mappatura {i+1}/{len(df_old)}")
                
                old_url = df_old.iloc[i]["Address"]
                old_lang = df_old.iloc[i]["lang"]
                
                # REGOLE DI SILOS LINGUISTICO
                pool_idxs = df_new.index[df_new["lang"] == old_lang].tolist()
                
                # Se la lingua non esiste (es. DE rimosso), fallback su EN
                target_lang_for_fallback = old_lang
                if not pool_idxs:
                    pool_idxs = df_new.index[df_new["lang"] == "en"].tolist()
                    target_lang_for_fallback = "en"
                
                # Inizializza con Home di lingua (Zero 404 policy)
                best_url = home_pages.get(target_lang_for_fallback, df_new.iloc[0]["Address"])
                best_score = 0
                method = "Fallback: Home di Lingua"

                if pool_idxs:
                    # Estraiamo solo le colonne del pool linguistico dalla matrice sims
                    pool_sims = sims[i, pool_idxs]
                    local_idx = np.argmax(pool_sims)
                    score = pool_sims[local_idx]
                    
                    if score >= threshold_primary:
                        best_url = df_new.iloc[pool_idxs[local_idx]]["Address"]
                        best_score = score
                        method = f"Match Semantico ({target_lang_for_fallback.upper()})"
                
                results.append({
                    "Old URL": old_url,
                    "New URL": best_url,
                    "Confidence": f"{round(best_score * 100, 1)}%",
                    "Method": method
                })

            progress.empty()
            status.update(label="Completato!", state="complete", expanded=False)
            
            final_df = pd.DataFrame(results)
            st.dataframe(final_df, use_container_width=True)
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                final_df.to_excel(writer, index=False)
            st.download_button("📥 Scarica Redirect Map", output.getvalue(), "redirect_map_final.xlsx")
