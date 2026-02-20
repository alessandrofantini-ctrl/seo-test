import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse, urlunparse
from io import BytesIO
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

# =========================
# UI SETUP
# =========================
st.set_page_config(page_title="AI Redirect Mapper - Domain & Lang Precision", layout="wide")
st.title("🎯 AI Redirect Mapper")

with st.sidebar:
    st.header("⚙️ Configurazione")
    openai_api_key = st.text_input("OpenAI key", type="password")
    
    st.markdown("---")
    st.subheader("📚 Logica di Reindirizzamento")
    st.info("Solo file HTML e Status 200 vengono elaborati.")
    
    threshold_primary = st.slider("Soglia Similarità (Minima)", 0.0, 1.0, 0.40, 
                                 help="Sotto questa soglia, l'URL andrà alla Home di lingua invece che a una pagina interna non pertinente.")

    st.markdown("---")
    st.subheader("🌐 Mappatura Domini Dismessi")
    st.write("Indica dove devono finire i vecchi domini se la lingua originale non esiste più.")
    
    # Esempio pratico per l'utente
    domain_map_input = st.text_area("Formato: dominio:lingua_target (uno per riga)", 
                                    value="bossong-befestigungssysteme.de:en\nbossong.es:es\nbossong.it:it\nbossong.co.uk:en")

# =========================
# SEO HELPERS
# =========================
def get_domain_map(input_text):
    d_map = {}
    for line in input_text.split('\n'):
        if ':' in line:
            domain, lang = line.split(':')
            d_map[domain.strip().lower()] = lang.strip().lower()
    return d_map

def load_files_combined(files) -> pd.DataFrame:
    all_dfs = []
    for file in files:
        try:
            name = (file.name or "").lower()
            if name.endswith((".xlsx", ".xls")):
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

def is_html(row):
    c_type = str(row.get("Content Type", "")).lower()
    return "text/html" in c_type or "html" in c_type

def detect_language(url: str, domain_mapping: dict) -> str:
    p = urlparse(url)
    domain = p.netloc.lower().replace("www.", "")
    path = p.path.lower()
    
    # 1. Controlla se il dominio è mappato esplicitamente
    if domain in domain_mapping:
        return domain_mapping[domain]
    
    # 2. Controlla sottocartelle
    m = re.search(r"/([a-z]{2})(?:/|$)", path)
    if m: return m.group(1)
    
    # 3. Fallback TLD
    if domain.endswith(".it"): return "it"
    if domain.endswith(".es"): return "es"
    if domain.endswith(".de"): return "en" # Tedesco rimosso -> EN
    
    return "en"

def clean_seo_text(row: pd.Series) -> str:
    url = str(row.get("Address", ""))
    slug = urlparse(url).path.replace("/", " ").replace("-", " ").replace(".html", "")
    t = str(row.get("Title 1", ""))
    h1 = str(row.get("H1-1", ""))
    return f"KEYWORDS: {slug} | TITLE: {t} | H1: {h1}".strip()[:7000]

# =========================
# ENGINE
# =========================
col1, col2 = st.columns(2)
with col1: old_files = st.file_uploader("📂 Vecchio Sito", accept_multiple_files=True)
with col2: new_files = st.file_uploader("📂 Nuovo Sito (Destinazioni)", accept_multiple_files=True)

if old_files and new_files:
    d_mapping = get_domain_map(domain_map_input)
    df_old_raw = load_files_combined(old_files)
    df_new_raw = load_files_combined(new_files)

    if df_old_raw is not None and df_new_raw is not None:
        # Filtro HTML (solo pagine vere)
        df_old = df_old_raw[df_old_raw.apply(is_html, axis=1)].copy()
        df_new = df_new_raw[df_new_raw.apply(is_html, axis=1) & (df_new_raw["Status Code"].astype(str) == "200")].copy()

        df_old["lang"] = df_old["Address"].apply(lambda x: detect_language(x, d_mapping))
        df_new["lang"] = df_new["Address"].apply(lambda x: detect_language(x, {})) # Nuovo sito ha già cartelle

        st.success(f"Dati caricati. {len(df_old)} URL sorgente filtrati.")

        if st.button("🚀 GENERA MAPPATURA"):
            if not openai_api_key: st.error("Inserisci API Key"); st.stop()
            client = OpenAI(api_key=openai_api_key)
            
            with st.spinner("Analisi semantica..."):
                old_texts = df_old.apply(clean_seo_text, axis=1).tolist()
                new_texts = df_new.apply(clean_seo_text, axis=1).tolist()
                
                def get_emb_batch(txt_list):
                    embs = []
                    for i in range(0, len(txt_list), BATCH_SIZE):
                        res = client.embeddings.create(input=txt_list[i:i+BATCH_SIZE], model=EMBED_MODEL)
                        embs.extend([d.embedding for d in res.data])
                    return embs

                emb_old = get_emb_batch(old_texts)
                emb_new = get_emb_batch(new_texts)
                sims = cosine_similarity(emb_old, emb_new)

            # Matching logic
            results = []
            # Trova le Home Page per ogni lingua nel nuovo sito (per fallback)
            home_pages = {lang: df_new[df_new["lang"] == lang].iloc[0]["Address"] 
                          for lang in df_new["lang"].unique() if len(df_new[df_new["lang"] == lang]) > 0}

            for i in range(len(df_old)):
                old_url = df_old.iloc[i]["Address"]
                old_lang = df_old.iloc[i]["lang"]
                
                # Pool di destinazione: solo stessa lingua
                pool_idxs = df_new.index[df_new["lang"] == old_lang].tolist()
                
                best_url = home_pages.get(old_lang, list(home_pages.values())[0]) # Default a Home di lingua
                best_score = 0
                method = "Fallback: Home di Lingua"

                if pool_idxs:
                    pool_sims = sims[i, pool_idxs]
                    best_local_idx = np.argmax(pool_sims)
                    score = pool_sims[best_local_idx]
                    
                    if score >= threshold_primary:
                        best_url = df_new.iloc[pool_idxs[best_local_idx]]["Address"]
                        best_score = score
                        method = f"Match Semantico ({old_lang})"
                
                results.append({
                    "Old URL": old_url,
                    "New URL": best_url,
                    "Score %": round(best_score * 100, 1),
                    "Method": method
                })

            final_df = pd.DataFrame(results)
            st.dataframe(final_df, use_container_width=True)
            
            # Download
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                final_df.to_excel(writer, index=False)
            st.download_button("📥 Scarica Report", output.getvalue(), "redirect_map_senior.xlsx")
