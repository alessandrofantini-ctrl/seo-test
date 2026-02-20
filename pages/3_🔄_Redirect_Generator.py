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

st.set_page_config(page_title="AI Redirect Mapper - E-commerce SEO", layout="wide")
st.title("🎯 AI Redirect Mapper - Category & Product Logic")

with st.sidebar:
    st.header("⚙️ Configurazione")
    openai_api_key = st.text_input("OpenAI key", type="password")
    
    st.markdown("---")
    st.subheader("📚 Logica E-commerce")
    st.info("""
    1. **Exact Match**: Cerca il prodotto identico.
    2. **Category Fallback**: Se il prodotto non esiste, analizza la cartella superiore (categoria).
    3. **Domain Mapping**: Forza i domini specifici (es. .co.uk -> /en/).
    """)
    
    threshold_primary = st.slider("Soglia Similarità Prodotto", 0.0, 1.0, 0.50)
    
    domain_map_input = st.text_area(
        "Mappatura Domini (dominio:lingua)", 
        value="bossong.co.uk:en\nbossong.es:es\nbossong.it:it\nbossong-befestigungssysteme.de:en"
    )

# =========================
# FUNZIONI SEO AVANZATE
# =========================

def get_domain_map(input_text):
    d_map = {}
    for line in input_text.split('\n'):
        if ":" in line:
            parts = line.split(':')
            d_map[parts[0].strip().lower().replace("www.", "")] = parts[1].strip().lower()
    return d_map

def get_parent_url(url):
    """Ritorna l'URL della categoria superiore (es. rimuove il prodotto)."""
    p = urlparse(url)
    path = p.path.strip('/')
    if not path: return url
    parts = path.split('/')
    if len(parts) > 1:
        new_path = '/' + '/'.join(parts[:-1]) + '/'
        return urlunparse((p.scheme, p.netloc, new_path, '', '', ''))
    return urlunparse((p.scheme, p.netloc, '/', '', '', ''))

def detect_language(url: str, domain_mapping: dict) -> str:
    p = urlparse(url)
    domain = p.netloc.lower().replace("www.", "")
    if domain in domain_mapping: return domain_mapping[domain]
    m = re.search(r"/([a-z]{2})(?:/|$)", p.path.lower())
    if m: return m.group(1)
    if domain.endswith(".it"): return "it"
    if domain.endswith(".es"): return "es"
    return "en"

def clean_seo_text(row: pd.Series, is_category=False):
    """Estrae testo SEO. Se is_category=True, dà più peso al path."""
    url = str(row.get("Address", ""))
    path_parts = urlparse(url).path.replace("/", " ").replace("-", " ").replace(".html", "")
    t = str(row.get("Title 1", ""))
    h1 = str(row.get("H1-1", ""))
    prefix = "CATEGORY_LOOKUP:" if is_category else "PRODUCT_LOOKUP:"
    return f"{prefix} {path_parts} | TITLE: {t} | H1: {h1}"[:7000]

# =========================
# CORE ENGINE
# =========================

col1, col2 = st.columns(2)
with col1: old_files = st.file_uploader("📂 Export Vecchio Sito", accept_multiple_files=True)
with col2: new_files = st.file_uploader("📂 Export Nuovo Sito", accept_multiple_files=True)

if old_files and new_files:
    d_mapping = get_domain_map(domain_map_input)
    df_old_raw = pd.concat([pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f) for f in old_files], ignore_index=True)
    df_new_raw = pd.concat([pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f) for f in new_files], ignore_index=True)

    # Filtro solo HTML e Status 200 per il nuovo
    df_old = df_old_raw[df_old_raw["Content Type"].str.contains("html", na=False)].copy().reset_index(drop=True)
    df_new = df_new_raw[(df_new_raw["Content Type"].str.contains("html", na=False)) & (df_new_raw["Status Code"].astype(str) == "200")].copy().reset_index(drop=True)

    df_old["lang"] = df_old["Address"].apply(lambda x: detect_language(x, d_mapping))
    df_new["lang"] = df_new["Address"].apply(lambda x: detect_language(x, {}))

    if st.button("🚀 GENERA REDIRECT MAP"):
        client = OpenAI(api_key=openai_api_key)
        
        # 1. Calcolo Embeddings (Prodotto)
        with st.spinner("Analisi prodotti e categorie..."):
            emb_old = [client.embeddings.create(input=clean_seo_text(r), model=EMBED_MODEL).data[0].embedding for _, r in df_old.iterrows()]
            emb_new = [client.embeddings.create(input=clean_seo_text(r), model=EMBED_MODEL).data[0].embedding for _, r in df_new.iterrows()]
            sims = cosine_similarity(emb_old, emb_new)

            # 2. Calcolo Embeddings (Solo Categorie per fallback)
            old_cat_texts = [clean_seo_text(r, is_category=True) for _, r in df_old.iterrows()]
            emb_old_cat = [client.embeddings.create(input=t, model=EMBED_MODEL).data[0].embedding for t in old_cat_texts]
            sims_cat = cosine_similarity(emb_old_cat, emb_new)

        # 3. Matching con Logica a Cascata
        home_pages = {l: df_new[df_new["lang"] == l].sort_values(by="Address", key=lambda x: x.str.len()).iloc[0]["Address"] for l in df_new["lang"].unique()}
        
        results = []
        progress = st.progress(0)
        
        for i in range(len(df_old)):
            progress.progress((i+1)/len(df_old))
            old_url = df_old.iloc[i]["Address"]
            old_lang = df_old.iloc[i]["lang"]
            pool_idxs = df_new.index[df_new["lang"] == old_lang].tolist()
            
            if not pool_idxs: pool_idxs = df_new.index[df_new["lang"] == "en"].tolist()
            
            best_url = home_pages.get(old_lang, list(home_pages.values())[0])
            method = "Fallback: Home di Lingua"
            max_score = 0

            if pool_idxs:
                # STEP A: Prova match prodotto (Slug + Title)
                p_sims = sims[i, pool_idxs]
                best_p_idx = np.argmax(p_sims)
                
                if p_sims[best_p_idx] >= threshold_primary:
                    best_url = df_new.iloc[pool_idxs[best_p_idx]]["Address"]
                    max_score = p_sims[best_p_idx]
                    method = "Match Prodotto"
                else:
                    # STEP B: Fallback Categoria (Folder Stripping)
                    c_sims = sims_cat[i, pool_idxs]
                    best_c_idx = np.argmax(c_sims)
                    best_url = df_new.iloc[pool_idxs[best_c_idx]]["Address"]
                    max_score = c_sims[best_c_idx]
                    method = "Fallback: Categoria Prodotto"

            results.append({"Old URL": old_url, "New URL": best_url, "Score": f"{round(max_score*100,1)}%", "Method": method})

        st.dataframe(pd.DataFrame(results))
