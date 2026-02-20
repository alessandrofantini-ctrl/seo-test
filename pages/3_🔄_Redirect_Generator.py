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
TOPK = 5 

# =========================
# UI SETUP
# =========================
st.set_page_config(page_title="AI Redirect Mapper Pro", layout="wide")
st.title("🎯 AI Redirect Mapper - Senior SEO Edition")

with st.sidebar:
    st.header("⚙️ Configurazione Avanzata")
    openai_api_key = st.text_input("OpenAI key", type="password")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]

    st.markdown("---")
    st.subheader("📚 Logica delle Soglie")
    st.info("""
    **1. Soglia Stessa Lingua (Consigliato: 0.75 - 0.85)**
    Se lo script trova una pagina nella stessa lingua con questo punteggio, la mappa immediatamente.
    
    **2. Soglia Fallback Cross-Language (Consigliato: 0.50 - 0.65)**
    Se la lingua originale è stata rimossa (es. Tedesco), lo script cerca nella sezione Inglese. Essendo lingue diverse, la similarità semantica è più bassa, quindi usiamo una soglia più permissiva.
    """)
    
    threshold_primary = st.slider("Soglia stessa lingua", 0.0, 1.0, 0.75)
    threshold_fallback = st.slider("Soglia fallback (Cross-lang)", 0.0, 1.0, 0.55)
    
    st.markdown("---")
    st.subheader("Strategia Anti-404")
    force_match = st.checkbox("Forza sempre un match (No 404)", value=True, 
                             help="Se attivo, anche se il punteggio è basso, lo script assegnerà la pagina più simile trovata o la Home.")

# =========================
# FUNZIONI DI SUPPORTO
# =========================
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
        except Exception as e: st.error(f"Errore caricamento {file.name}: {e}")
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None

def detect_language(url: str) -> str:
    path = urlparse(url).path.lower()
    domain = urlparse(url).netloc.lower()
    # Pattern comuni /it/, /es/, /de/, /fr/
    m = re.search(r"/([a-z]{2})(?:/|$)", path)
    if m: return m.group(1)
    # TLD check
    if domain.endswith(".it"): return "it"
    if domain.endswith(".es"): return "es"
    if domain.endswith(".de"): return "de"
    if domain.endswith(".fr"): return "fr"
    return "en" # Fallback globale su inglese

def clean_slug(url: str) -> str:
    path = urlparse(url).path.lower()
    path = re.sub(r"\.(html|php|asp|aspx)$", "", path)
    return path.replace("/", " ").replace("-", " ").replace("_", " ").strip()

def make_seo_text(row: pd.Series) -> str:
    # Un SEO Senior sa che il Title e lo Slug dell'URL sono i segnali più forti
    t = str(row.get("Title 1", ""))
    h1 = str(row.get("H1-1", ""))
    desc = str(row.get("Meta Description 1", ""))
    slug = clean_slug(str(row.get("Address", "")))
    # Diamo peso allo slug mettendolo all'inizio
    return f"URL_KEYWORDS: {slug} | TITLE: {t} | H1: {h1} | DESC: {desc}"[:7000]

def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Redirect_Map")
    return output.getvalue()

# =========================
# LOGICA DI MATCHING
# =========================
col1, col2 = st.columns(2)
with col1: old_files = st.file_uploader("📂 Vecchio Sito (SF Export)", accept_multiple_files=True, key="old")
with col2: new_files = st.file_uploader("📂 Nuovo Sito (SF Export)", accept_multiple_files=True, key="new")

if old_files and new_files:
    df_old_raw = load_files_combined(old_files)
    df_new_raw = load_files_combined(new_files)

    if df_old_raw is not None and df_new_raw is not None:
        # Pre-processing: Filtriamo solo 200 OK per il nuovo sito
        df_new = df_new_raw[df_new_raw["Status Code"].astype(str).str.contains("200")].copy()
        df_old = df_old_raw.copy()

        df_old["lang"] = df_old["Address"].apply(detect_language)
        df_new["lang"] = df_new["Address"].apply(detect_language)

        st.success(f"Dati caricati. {len(df_old)} URL da mappare.")

        if st.button("🚀 GENERA REDIRECT MAP PROFESSIONALE"):
            if not openai_api_key: st.error("Inserisci la API Key!"); st.stop()
            
            client = OpenAI(api_key=openai_api_key)
            status = st.status("Analisi semantica in corso...")

            # 1. Embedding
            status.write("🧠 Creazione impronte digitali dei contenuti...")
            old_texts = df_old.apply(make_seo_text, axis=1).tolist()
            new_texts = df_new.apply(make_seo_text, axis=1).tolist()
            
            emb_old = []
            for i in range(0, len(old_texts), BATCH_SIZE):
                res = client.embeddings.create(input=old_texts[i:i+BATCH_SIZE], model=EMBED_MODEL)
                emb_old.extend([d.embedding for d in res.data])
            
            emb_new = []
            for i in range(0, len(new_texts), BATCH_SIZE):
                res = client.embeddings.create(input=new_texts[i:i+BATCH_SIZE], model=EMBED_MODEL)
                emb_new.extend([d.embedding for d in res.data])

            sims = cosine_similarity(emb_old, emb_new)
            
            # 2. Matching a cascata (SEO Cascade Strategy)
            status.write("🔍 Applicazione regole SEO...")
            progress_bar = st.progress(0, text="Mappatura in corso...")
            results = []

            # Creiamo indici per velocizzare i fallback
            new_by_lang = {l: df_new.index[df_new["lang"] == l].tolist() for l in df_new["lang"].unique()}
            en_idxs = new_by_lang.get("en", [])

            for i in range(len(df_old)):
                progress_bar.progress((i + 1) / len(df_old), text=f"Mappatura {i+1} di {len(df_old)}")
                
                old_row = df_old.iloc[i]
                old_url = old_row["Address"]
                old_lang = old_row["lang"]
                
                best_url = ""
                best_score = 0
                method = "404"

                # STRATEGIA 1: Match nella stessa lingua
                same_lang_idxs = new_by_lang.get(old_lang, [])
                if same_lang_idxs:
                    row_sims = sims[i, same_lang_idxs]
                    idx_in_subset = np.argmax(row_sims)
                    if row_sims[idx_in_subset] >= threshold_primary:
                        best_url = df_new.iloc[same_lang_idxs[idx_in_subset]]["Address"]
                        best_score = row_sims[idx_in_subset]
                        method = f"Match Semantico ({old_lang.upper()})"

                # STRATEGIA 2: Fallback su Inglese (se lingua originale non ha match o è rimossa)
                if not best_url and old_lang != "en" and en_idxs:
                    en_sims = sims[i, en_idxs]
                    idx_in_en = np.argmax(en_sims)
                    if en_sims[idx_in_en] >= threshold_fallback:
                        best_url = df_new.iloc[en_idxs[idx_in_en]]["Address"]
                        best_score = en_sims[idx_in_en]
                        method = "Fallback Strategico (EN)"

                # STRATEGIA 3: Forza il miglior match assoluto (SEO Safety Net)
                if not best_url and force_match:
                    best_idx_overall = np.argmax(sims[i])
                    best_url = df_new.iloc[best_idx_overall]["Address"]
                    best_score = sims[i, best_idx_overall]
                    method = "Best Effort (Migliore somiglianza)"

                results.append({
                    "Old URL": old_url,
                    "New URL": best_url,
                    "Score": round(best_score * 100, 1),
                    "Method": method,
                    "Old Lang": old_lang
                })

            progress_bar.empty()
            status.update(label="Analisi completata!", state="complete", expanded=False)

            # Risultati
            final_df = pd.DataFrame(results)
            st.subheader("Anteprima Risultati")
            st.dataframe(final_df, use_container_width=True)

            # Export
            st.download_button(
                "📥 Scarica Excel (Redirect Map)",
                data=to_excel_bytes(final_df),
                file_name="bossong_redirect_map.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
