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
BATCH_SIZE = 100 # Velocizza di 100x rispetto alla versione precedente

st.set_page_config(page_title="AI Redirect Mapper - High Speed", layout="wide")
st.title("🚀 AI Redirect Mapper - Performance Mode")

with st.sidebar:
    st.header("⚙️ Configurazione")
    openai_api_key = st.text_input("OpenAI key", type="password")
    
    st.markdown("---")
    st.subheader("🌐 Mappatura Domini")
    domain_map_input = st.text_area(
        "dominio:lingua (uno per riga)", 
        value="bossong.co.uk:en\nbossong.es:es\nbossong.it:it\nbossong-befestigungssysteme.de:en",
        height=150
    )
    
    threshold_primary = st.slider("Soglia Similarità", 0.0, 1.0, 0.45)

# =========================
# FUNZIONI OTTIMIZZATE
# =========================

def get_domain_map(input_text):
    d_map = {}
    for line in input_text.split('\n'):
        if ":" in line:
            parts = line.split(':')
            d_map[parts[0].strip().lower().replace("www.", "")] = parts[1].strip().lower()
    return d_map

def detect_language(url: str, domain_mapping: dict) -> str:
    p = urlparse(url)
    domain = p.netloc.lower().replace("www.", "")
    if domain in domain_mapping: return domain_mapping[domain]
    path = p.path.lower()
    m = re.search(r"/([a-z]{2})(?:/|$)", path)
    if m: return m.group(1)
    if domain.endswith(".it"): return "it"
    if domain.endswith(".es"): return "es"
    return "en"

def get_seo_content(row: pd.Series) -> str:
    """Combina i segnali SEO in una stringa pulita."""
    url = str(row.get("Address", ""))
    path = urlparse(url).path.replace("/", " ").replace("-", " ").replace(".html", "").strip()
    title = str(row.get("Title 1", ""))
    h1 = str(row.get("H1-1", ""))
    # Diamo molta importanza al path per gli e-commerce
    return f"PATH: {path} | TITLE: {title} | H1: {h1}"[:7000]

def get_embeddings_batched(text_list, client):
    """Ottiene embeddings in batch (molto più veloce)."""
    all_embeddings = []
    for i in range(0, len(text_list), BATCH_SIZE):
        batch = [t.replace("\n", " ") for t in text_list[i:i+BATCH_SIZE]]
        response = client.embeddings.create(input=batch, model=EMBED_MODEL)
        all_embeddings.extend([d.embedding for d in response.data])
    return all_embeddings

# =========================
# CORE ENGINE
# =========================

col1, col2 = st.columns(2)
with col1: old_files = st.file_uploader("📂 Vecchio Sito", accept_multiple_files=True)
with col2: new_files = st.file_uploader("📂 Nuovo Sito", accept_multiple_files=True)

if old_files and new_files:
    d_mapping = get_domain_map(domain_map_input)
    
    # Caricamento rapido
    df_old_raw = pd.concat([pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f) for f in old_files], ignore_index=True)
    df_new_raw = pd.concat([pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f) for f in new_files], ignore_index=True)

    # Filtri HTML e indici puliti
    df_old = df_old_raw[df_old_raw["Content Type"].str.contains("html", na=False)].copy().reset_index(drop=True)
    df_new = df_new_raw[(df_new_raw["Content Type"].str.contains("html", na=False)) & (df_new_raw["Status Code"].astype(str) == "200")].copy().reset_index(drop=True)

    df_old["lang"] = df_old["Address"].apply(lambda x: detect_language(x, d_mapping))
    df_new["lang"] = df_new["Address"].apply(lambda x: detect_language(x, {}))

    if st.button("🚀 GENERA REDIRECT MAP (MODALITÀ VELOCE)"):
        if not openai_api_key: st.error("Manca API Key"); st.stop()
        client = OpenAI(api_key=openai_api_key)
        
        status = st.status("Elaborazione accelerata...")
        
        # 1. Calcolo Embeddings in BATCH
        status.write("🧠 Creazione impronte digitali dei contenuti (Batch Mode)...")
        old_texts = df_old.apply(get_seo_content, axis=1).tolist()
        new_texts = df_new.apply(get_seo_content, axis=1).tolist()
        
        emb_old = get_embeddings_batched(old_texts, client)
        emb_new = get_embeddings_batched(new_texts, client)
        
        sims = cosine_similarity(emb_old, emb_new)

        # 2. Identificazione Home di Lingua
        home_pages = {}
        for l in df_new["lang"].unique():
            subset = df_new[df_new["lang"] == l]
            home_pages[l] = subset.loc[subset['Address'].str.len().idxmin()]['Address']

        # 3. Matching
        status.write("🔍 Applicazione regole gerarchiche...")
        results = []
        
        for i in range(len(df_old)):
            old_url = df_old.iloc[i]["Address"]
            old_lang = df_old.iloc[i]["lang"]
            
            # Pool di destinazione
            pool_idxs = df_new.index[df_new["lang"] == old_lang].tolist()
            if not pool_idxs: pool_idxs = df_new.index[df_new["lang"] == "en"].tolist()
            
            # Default
            target_lang = old_lang if old_lang in home_pages else "en"
            best_url = home_pages.get(target_lang, df_new.iloc[0]["Address"])
            score = 0
            method = "Fallback: Home di Lingua"

            # Se è la home del vecchio dominio, forza la home del nuovo
            if urlparse(old_url).path.strip('/') == "":
                best_url = home_pages.get(target_lang, best_url)
                method = "Forced: Home Mapping"
            elif pool_idxs:
                pool_sims = sims[i, pool_idxs]
                best_local_idx = np.argmax(pool_sims)
                if pool_sims[best_local_idx] >= threshold_primary:
                    best_url = df_new.iloc[pool_idxs[best_local_idx]]["Address"]
                    score = pool_sims[best_local_idx]
                    method = "Match Semantico"
                else:
                    # Fallback Categoria (Semplice: il secondo miglior match nel pool è spesso la categoria o listino)
                    # In alternativa, l'AI ha già dato il miglior risultato possibile nel pool.
                    best_url = df_new.iloc[pool_idxs[best_local_idx]]["Address"]
                    score = pool_sims[best_local_idx]
                    method = "Fallback: Miglior Corrispondenza (Categoria/Prodotto)"

            results.append({
                "Old URL": old_url, 
                "New URL": best_url, 
                "Score": f"{round(score*100,1)}%", 
                "Method": method
            })

        status.update(label="Mappatura completata!", state="complete", expanded=False)
        
        final_df = pd.DataFrame(results)
        st.dataframe(final_df, use_container_width=True)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            final_df.to_excel(writer, index=False)
        st.download_button("📥 Scarica Excel", output.getvalue(), "redirect_map_bossong.xlsx")
