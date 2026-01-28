import streamlit as st
import pandas as pd
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import time
from io import BytesIO # Necessario per l'export Excel

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Pure AI Redirect Mapper", layout="wide")

st.title("🧠 Pure AI Redirect Mapper")
st.markdown("""
Questo strumento utilizza **esclusivamente l'Intelligenza Artificiale** per mappare i redirect.
1. **Filtro HTML 🧹**: Scarta automaticamente immagini, CSS, JS e PDF.
2. **AI Deep Semantic 🧠**: Analizza URL, Title, H1 e Body Text per trovare la corrispondenza semantica migliore, ignorando la somiglianza dei percorsi.
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configurazione")
    openai_api_key = st.text_input("OpenAI Key", type="password")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    
    threshold = st.slider("Soglia Confidenza AI", 0.0, 1.0, 0.75)

# --- FUNZIONI ---

def load_sf_data(uploaded_file):
    """Carica dati, filtra SOLO HTML e gestisce Custom Extraction."""
    df = None
    filename = uploaded_file.name
    
    # 1. CARICAMENTO
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

    # 2. PULIZIA E FILTRO HTML
    try:
        df.columns = df.columns.str.strip()
        cols_lower = df.columns.str.lower()
        orig_cols = df.columns
        
        # --- FILTRO HTML ---
        col_type_idx = next((i for i, c in enumerate(cols_lower) if "content" in c and "type" in c), None)
        total_rows = len(df)
        
        if col_type_idx is not None:
            col_name = orig_cols[col_type_idx]
            df = df[df[col_name].astype(str).str.contains("html", case=False, na=False)]
        
        html_rows = len(df)
        if total_rows > html_rows:
            st.toast(f"🧹 Filtrati {total_rows - html_rows} file non-HTML. Rimasti: {html_rows}.", icon="🗑️")

        # --- MAPPATURA COLONNE ---
        col_url = next((c for c in cols_lower if "address" in c or "indirizzo" in c), None)
        col_title = next((c for c in cols_lower if ("title 1" in c or "titolo 1" in c) and "len" not in c), None)
        col_h1 = next((c for c in cols_lower if "h1-1" in c and "len" not in c), None)
        col_desc = next((c for c in cols_lower if "meta description 1" in c and "len" not in c), None)
        col_h2 = next((c for c in cols_lower if "h2-1" in c and "len" not in c), None)
        col_content = next((c for c in cols_lower if ("content" in c or "body" in c or "text" in c or "testo" in c) and "type" not in c), None)
        
        if not col_url:
            st.error(f"Colonna 'Address' mancante in {filename}"); return None
            
        clean_df = pd.DataFrame()
        
        # Estrazione Dati
        clean_df['url'] = df[orig_cols[cols_lower.get_loc(col_url)]].astype(str)
        
        def safe_get(c_name):
            if c_name: return df[orig_cols[cols_lower.get_loc(c_name)]].fillna("").astype(str)
            return pd.Series([""] * len(df))

        clean_df['title'] = safe_get(col_title)
        clean_df['h1'] = safe_get(col_h1)
        clean_df['desc'] = safe_get(col_desc)
        clean_df['h2'] = safe_get(col_h2)
        clean_df['body_text'] = safe_get(col_content) 
        
        return clean_df

    except Exception as e:
        st.error(f"Errore struttura file: {e}"); return None

def get_embedding_batch(text_list, client):
    try:
        clean_texts = [str(t)[:6000].replace("\n", " ") for t in text_list] 
        response = client.embeddings.create(input=clean_texts, model="text-embedding-3-small")
        return [data.embedding for data in response.data]
    except Exception as e:
        st.error(f"OpenAI Error: {e}"); return []

def to_excel(df):
    """Converte DF in Excel Bytes per il download"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Redirects')
    processed_data = output.getvalue()
    return processed_data

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
        
        st.info(f"📁 **Analisi**: {len(df_old)} URL validi (HTML) nel vecchio sito vs {len(df_new)} nel nuovo.")

        has_text = "body_text" in df_old.columns and df_old['body_text'].str.len().sum() > 100
        if has_text:
            st.success("✅ 'Body Text' rilevato! Analisi semantica potenziata.")
        
        if st.button("🚀 Avvia Full AI Matching"):
            
            if not openai_api_key:
                st.error("Per usare l'AI devi inserire la API Key nella sidebar.")
            else:
                status = st.status("Preparazione Analisi Semantica...", expanded=True)
                results = []
                
                # --- PREPARAZIONE AI CONTEXT ---
                # Non facciamo filtri logici, mandiamo tutto all'AI
                
                def make_context(df):
                    ctx = "URL: " + df['url'] + " | T: " + df['title'] + " | H1: " + df['h1']
                    ctx += " | BODY: " + df['body_text'].str.slice(0, 1000)
                    return ctx

                df_old['ctx'] = make_context(df_old)
                df_new['ctx'] = make_context(df_new)
                
                client = OpenAI(api_key=openai_api_key)
                
                # --- CALCOLO EMBEDDINGS ---
                status.write(f"🧠 Vettorializzazione di {len(df_old)} URL vecchi e {len(df_new)} URL nuovi...")
                
                emb_old = []
                batch_s = 100
                prog = status.progress(0)
                
                # Embedding Old
                for i in range(0, len(df_old), batch_s):
                    b = df_old['ctx'].iloc[i:i+batch_s].tolist()
                    emb_old.extend(get_embedding_batch(b, client))
                    prog.progress(0.4)
                
                # Embedding New
                emb_new = []
                for i in range(0, len(df_new), batch_s):
                    b = df_new['ctx'].iloc[i:i+batch_s].tolist()
                    emb_new.extend(get_embedding_batch(b, client))
                    prog.progress(0.8)

                # --- MATCHING ---
                status.write("📐 Calcolo similarità vettoriale...")
                
                if emb_old and emb_new:
                    mat_old = np.array(emb_old)
                    mat_new = np.array(emb_new)
                    sims = cosine_similarity(mat_old, mat_new)
                    
                    for i, vector_idx in enumerate(df_old.index):
                        row_old = df_old.loc[vector_idx]
                        scores = sims[i]
                        best_idx = scores.argmax()
                        score = scores[best_idx]
                        
                        if score >= threshold:
                            new_row = df_new.iloc[best_idx]
                            sug_url = new_row['url']
                        else:
                            sug_url = "" # Nessun match sopra soglia
                        
                        results.append({
                            "Old URL": row_old['url'],
                            "New URL": sug_url, # Colonna richiesta pulita
                            "Confidence": round(score * 100, 1),
                            "Old Title": row_old['title'] # Utile per debug ma non per export finale
                        })

                # --- OUTPUT ---
                final_df = pd.DataFrame(results)
                status.update(label="Mapping Completato!", state="complete", expanded=False)
                
                st.subheader("🎯 Risultati Finali")
                
                # Anteprima con colonne extra per controllo
                st.dataframe(final_df)
                
                # --- EXPORT PULITO (SOLO 2 COLONNE) ---
                # Filtriamo solo le righe che hanno un match e solo le colonne richieste
                export_df = final_df[final_df['New URL'] != ""][['Old URL', 'New URL']]
                
                excel_data = to_excel(export_df)
                
                st.download_button(
                    label="📥 Scarica Excel Finale (Solo Old & New URL)",
                    data=excel_data,
                    file_name="redirect_map_clean.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.info(f"Il file Excel contiene {len(export_df)} redirect pronti all'uso.")
