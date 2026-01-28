import streamlit as st
import pandas as pd
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
from io import BytesIO
import re

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Multilingual AI Redirect Mapper", layout="wide")

st.title("🌐 Multilingual AI Redirect Mapper")
st.markdown("""
Strumento ottimizzato per **migrazioni internazionali**.
1. **Language Aware 🇮🇹🇪🇸**: Rileva TLD (.it, .es) e sottocartelle lingua (/it/, /en/) per forzare la coerenza linguistica.
2. **Deep Semantic 🧠**: Usa l'AI per mappare i contenuti, ma rispetta la lingua di origine.
3. **Clean Export 📤**: Scarica un Excel pulito pronto per gli sviluppatori.
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configurazione")
    openai_api_key = st.text_input("OpenAI Key", type="password")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    
    threshold = st.slider("Soglia Confidenza AI", 0.0, 1.0, 0.82, help="Consigliato > 0.80 per siti multilingua per evitare falsi positivi.")

# --- FUNZIONI ---

def load_sf_data(uploaded_file):
    """Carica dati, filtra SOLO HTML e gestisce Custom Extraction."""
    df = None
    filename = uploaded_file.name
    
    try:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            encodings = ['utf-8', 'ISO-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc, low_memory=False)
                    break
                except: continue
    except Exception as e:
        st.error(f"Errore file: {e}"); return None

    if df is None: return None

    try:
        df.columns = df.columns.str.strip()
        cols_lower = df.columns.str.lower()
        orig_cols = df.columns
        
        # Filtro HTML
        col_type_idx = next((i for i, c in enumerate(cols_lower) if "content" in c and "type" in c), None)
        total_rows = len(df)
        
        if col_type_idx is not None:
            col_name = orig_cols[col_type_idx]
            df = df[df[col_name].astype(str).str.contains("html", case=False, na=False)]
        
        html_rows = len(df)
        if total_rows > html_rows:
            st.toast(f"🧹 Rimasti: {html_rows} URL HTML (filtrati {total_rows - html_rows}).", icon="🗑️")

        # Mappatura Colonne
        col_url = next((c for c in cols_lower if "address" in c or "indirizzo" in c), None)
        col_title = next((c for c in cols_lower if ("title 1" in c or "titolo 1" in c) and "len" not in c), None)
        col_h1 = next((c for c in cols_lower if "h1-1" in c and "len" not in c), None)
        col_content = next((c for c in cols_lower if ("content" in c or "body" in c or "text" in c or "testo" in c) and "type" not in c), None)
        
        if not col_url:
            st.error(f"Colonna 'Address' mancante in {filename}"); return None
            
        clean_df = pd.DataFrame()
        clean_df['url'] = df[orig_cols[cols_lower.get_loc(col_url)]].astype(str)
        
        def safe_get(c_name):
            if c_name: return df[orig_cols[cols_lower.get_loc(c_name)]].fillna("").astype(str)
            return pd.Series([""] * len(df))

        clean_df['title'] = safe_get(col_title)
        clean_df['h1'] = safe_get(col_h1)
        clean_df['body_text'] = safe_get(col_content) 
        
        return clean_df

    except Exception as e:
        st.error(f"Errore struttura file: {e}"); return None

def get_embedding_batch(text_list, client):
    try:
        clean_texts = [str(t)[:8000].replace("\n", " ") for t in text_list] 
        response = client.embeddings.create(input=clean_texts, model="text-embedding-3-small")
        return [data.embedding for data in response.data]
    except Exception as e:
        st.error(f"OpenAI Error: {e}"); return []

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Redirects')
    return output.getvalue()

def extract_language_signal(url):
    """
    Estrae segnali forti di lingua dall'URL per aiutare l'AI.
    Ritorna stringhe tipo: 'LANG_IT DOMAIN_IT'
    """
    signal = []
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    
    # 1. Analisi TLD (Top Level Domain)
    if domain.endswith(".it"): signal.append("DOMAIN_IT LANGUAGE_ITALIAN")
    elif domain.endswith(".es"): signal.append("DOMAIN_ES LANGUAGE_SPANISH")
    elif domain.endswith(".fr"): signal.append("DOMAIN_FR LANGUAGE_FRENCH")
    elif domain.endswith(".de"): signal.append("DOMAIN_DE LANGUAGE_GERMAN")
    elif domain.endswith(".co.uk"): signal.append("DOMAIN_UK LANGUAGE_ENGLISH")
    
    # 2. Analisi Path (Sottocartelle)
    # Cerca pattern come /it/, /en-us/, /es/
    path_segments = path.split('/')
    if len(path_segments) > 1:
        first_seg = path_segments[1].lower()
        if len(first_seg) == 2 or (len(first_seg) == 5 and '-' in first_seg):
            if first_seg in ['it', 'it-it']: signal.append("PATH_IT LANGUAGE_ITALIAN")
            elif first_seg in ['en', 'en-us', 'en-gb']: signal.append("PATH_EN LANGUAGE_ENGLISH")
            elif first_seg in ['es', 'es-es']: signal.append("PATH_ES LANGUAGE_SPANISH")
            elif first_seg in ['fr', 'fr-fr']: signal.append("PATH_FR LANGUAGE_FRENCH")
            elif first_seg in ['de', 'de-de']: signal.append("PATH_DE LANGUAGE_GERMAN")
            
    return " ".join(signal)

# --- INTERFACCIA ---

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Vecchio Sito")
    old_file = st.file_uploader("Vecchio (CSV/Excel)", type=["csv", "xlsx"], key="old")
with col2:
    st.subheader("2. Nuovo Sito")
    new_file = st.file_uploader("Nuovo (CSV/Excel)", type=["csv", "xlsx"], key="new")

if old_file and new_file:
    df_old = load_sf_data(old_file)
    df_new = load_sf_data(new_file)
    
    if df_old is not None and df_new is not None:
        
        st.info(f"📁 Analisi: {len(df_old)} URL Old vs {len(df_new)} URL New.")

        if st.button("🚀 Avvia Language-Aware Matching"):
            
            if not openai_api_key:
                st.error("Inserisci API Key.")
            else:
                status = st.status("Analisi Lingua e Contenuti...", expanded=True)
                results = []
                
                client = OpenAI(api_key=openai_api_key)

                # --- CONTEXT CREATION CON LANGUAGE BOOSTING ---
                def make_enhanced_context(df):
                    contexts = []
                    for _, row in df.iterrows():
                        # Estraiamo il segnale lingua
                        lang_sig = extract_language_signal(row['url'])
                        
                        # Creiamo la stringa per l'embedding mettendo la LINGUA all'inizio
                        # Esempio: "[DOMAIN_IT LANGUAGE_ITALIAN] URL: ... TITLE: ..."
                        ctx = f"[{lang_sig}] URL: {row['url']} | TITLE: {row['title']} | H1: {row['h1']}"
                        
                        # Aggiungiamo body text se c'è, ma troncato
                        if row['body_text']:
                            ctx += f" | CONTENT: {row['body_text'][:800]}"
                        
                        contexts.append(ctx)
                    return contexts

                status.write("🧠 Estrazione segnali linguistici (TLD, Path)...")
                df_old['ctx'] = make_enhanced_context(df_old)
                df_new['ctx'] = make_enhanced_context(df_new)
                
                # --- EMBEDDINGS ---
                status.write("📐 Vettorializzazione...")
                emb_old = []
                batch_s = 100
                prog = status.progress(0)
                
                # Old
                for i in range(0, len(df_old), batch_s):
                    b = df_old['ctx'].iloc[i:i+batch_s].tolist()
                    emb_old.extend(get_embedding_batch(b, client))
                    prog.progress(0.4)
                
                # New
                emb_new = []
                for i in range(0, len(df_new), batch_s):
                    b = df_new['ctx'].iloc[i:i+batch_s].tolist()
                    emb_new.extend(get_embedding_batch(b, client))
                    prog.progress(0.8)

                # --- MATCHING ---
                status.write("🔍 Ricerca corrispondenze semantiche...")
                
                if emb_old and emb_new:
                    mat_old = np.array(emb_old)
                    mat_new = np.array(emb_new)
                    sims = cosine_similarity(mat_old, mat_new)
                    
                    for i, vector_idx in enumerate(df_old.index):
                        row_old = df_old.loc[vector_idx]
                        scores = sims[i]
                        best_idx = scores.argmax()
                        score = scores[best_idx]
                        
                        sug_url = ""
                        # Logica Threshold
                        if score >= threshold:
                            new_row = df_new.iloc[best_idx]
                            sug_url = new_row['url']
                        
                        results.append({
                            "Old URL": row_old['url'],
                            "New URL": sug_url,
                            "Confidence": round(score * 100, 1),
                            "Debug Context": row_old['ctx'][:50] # Utile per vedere se ha preso la lingua
                        })

                # --- EXPORT ---
                final_df = pd.DataFrame(results)
                status.update(label="Fatto!", state="complete", expanded=False)
                
                st.subheader("🎯 Risultati (Language Aware)")
                
                # Preview
                st.dataframe(final_df.head(10))
                
                # Export Clean
                export_df = final_df[final_df['New URL'] != ""][['Old URL', 'New URL']]
                excel_data = to_excel(export_df)
                
                st.download_button(
                    label="📥 Scarica Excel Finale (Solo Old & New)",
                    data=excel_data,
                    file_name="redirect_map_multilang.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.info(f"Redirect generati: {len(export_df)}. Il sistema ha penalizzato automaticamente i match tra lingue diverse.")
