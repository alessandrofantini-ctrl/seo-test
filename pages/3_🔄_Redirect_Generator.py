import streamlit as st
import pandas as pd
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import time

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Hybrid Redirect Mapper", layout="wide")

st.title("⚡ Hybrid SEO Redirect Mapper")
st.markdown("""
Questo strumento usa un approccio ibrido per la massima precisione:
1. **Filtro HTML 🧹**: Scarta automaticamente immagini, CSS, JS e PDF.
2. **Smart Slug Match ⚡**: Accoppia istantaneamente URL identiche o con lo stesso percorso.
3. **AI Deep Semantic 🧠**: Usa l'AI e il contenuto della pagina per mappare le URL orfane.
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configurazione")
    openai_api_key = st.text_input("OpenAI Key", type="password")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    
    threshold = st.slider("Soglia Confidenza AI", 0.0, 1.0, 0.80)

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
        # Cerchiamo la colonna "Content Type"
        col_type_idx = next((i for i, c in enumerate(cols_lower) if "content" in c and "type" in c), None)
        
        total_rows = len(df)
        
        if col_type_idx is not None:
            # Filtriamo: teniamo solo le righe dove Content Type contiene "html"
            col_name = orig_cols[col_type_idx]
            df = df[df[col_name].astype(str).str.contains("html", case=False, na=False)]
        
        html_rows = len(df)
        
        # Se abbiamo filtrato qualcosa, lo notifichiamo (ma non blocchiamo)
        if total_rows > html_rows:
            st.toast(f"🧹 Filtrati {total_rows - html_rows} file non-HTML (immagini, js, pdf). Rimasti: {html_rows}.", icon="🗑️")

        # --- MAPPATURA COLONNE ---
        # Ricalcoliamo cols_lower perché il dataframe è stato filtrato ma le colonne sono le stesse
        # (Non serve ricalcolare cols_lower, ma serve ritrovare i dati)

        col_url = next((c for c in cols_lower if "address" in c or "indirizzo" in c), None)
        
        # Colonne Standard
        col_title = next((c for c in cols_lower if ("title 1" in c or "titolo 1" in c) and "len" not in c), None)
        col_h1 = next((c for c in cols_lower if "h1-1" in c and "len" not in c), None)
        
        # Colonne Avanzate
        col_desc = next((c for c in cols_lower if "meta description 1" in c and "len" not in c), None)
        col_h2 = next((c for c in cols_lower if "h2-1" in c and "len" not in c), None)
        
        # Custom Extraction (Body Text)
        # Escludiamo "Content Type" dalla ricerca del body text per evitare conflitti
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
        
        # Path per matching
        clean_df['path'] = clean_df['url'].apply(lambda x: urlparse(x).path)
        
        # Filtro Status 200 (Ridondante se abbiamo già filtrato HTML, ma utile per sicurezza)
        col_status = next((c for c in cols_lower if "status code" in c), None)
        if col_status:
            idx = cols_lower.get_loc(col_status)
            clean_df = clean_df[pd.to_numeric(df[orig_cols[idx]], errors='coerce') == 200]
            
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

        # Rilevamento Testo
        has_text = "body_text" in df_old.columns and df_old['body_text'].str.len().sum() > 100
        if has_text:
            st.success("✅ 'Body Text' rilevato! Analisi semantica potenziata.")
        
        if st.button("🚀 Avvia Hybrid Matching"):
            
            status = st.status("Fase 1: Smart Slug Matching (Logica)...", expanded=True)
            results = []
            
            # --- FASE 1: LOGICAL MATCHING ---
            new_map = dict(zip(df_new['path'], df_new['url']))
            ai_old_indices = []
            count_logical = 0
            
            for idx, row in df_old.iterrows():
                old_path = row['path']
                if old_path in new_map:
                    results.append({
                        "Old URL": row['url'],
                        "Old Title": row['title'],
                        "Suggested URL": new_map[old_path],
                        "New Title": "Match Esatto Path",
                        "Confidence": 100.0,
                        "Method": "Slug Match ⚡"
                    })
                    count_logical += 1
                else:
                    ai_old_indices.append(idx)
            
            status.write(f"✅ {count_logical} Match Logici. Procedo con AI per i restanti {len(ai_old_indices)} URL...")
            
            # --- FASE 2: AI MATCHING ---
            if ai_old_indices and openai_api_key:
                status.write("🧠 Fase 2: Analisi Semantica Profonda...")
                
                df_old_ai = df_old.loc[ai_old_indices].copy()
                
                def make_context(df):
                    ctx = "URL: " + df['url'] + " | T: " + df['title'] + " | H1: " + df['h1']
                    ctx += " | BODY: " + df['body_text'].str.slice(0, 1000)
                    return ctx

                df_old_ai['ctx'] = make_context(df_old_ai)
                df_new['ctx'] = make_context(df_new)
                
                client = OpenAI(api_key=openai_api_key)
                
                # Embedding
                emb_old = []
                batch_s = 100
                prog = status.progress(0)
                
                # Embedding Old (Solo orfani)
                for i in range(0, len(df_old_ai), batch_s):
                    b = df_old_ai['ctx'].iloc[i:i+batch_s].tolist()
                    emb_old.extend(get_embedding_batch(b, client))
                    prog.progress(0.3)
                
                # Embedding New (Tutti)
                emb_new = []
                for i in range(0, len(df_new), batch_s):
                    b = df_new['ctx'].iloc[i:i+batch_s].tolist()
                    emb_new.extend(get_embedding_batch(b, client))
                    prog.progress(0.6)

                if emb_old and emb_new:
                    mat_old = np.array(emb_old)
                    mat_new = np.array(emb_new)
                    sims = cosine_similarity(mat_old, mat_new)
                    
                    for i, vector_idx in enumerate(df_old_ai.index):
                        row_old = df_old.loc[vector_idx]
                        scores = sims[i]
                        best_idx = scores.argmax()
                        score = scores[best_idx]
                        
                        if score >= threshold:
                            new_row = df_new.iloc[best_idx]
                            sug_url = new_row['url']
                            sug_tit = new_row['title']
                        else:
                            sug_url = ""
                            sug_tit = ""
                        
                        results.append({
                            "Old URL": row_old['url'],
                            "Old Title": row_old['title'],
                            "Suggested URL": sug_url,
                            "New Title": sug_tit,
                            "Confidence": round(score * 100, 1),
                            "Method": "AI Semantic 🧠"
                        })

            elif ai_old_indices and not openai_api_key:
                st.warning("Inserisci API Key per il matching AI.")
                
            # --- OUTPUT ---
            final_df = pd.DataFrame(results)
            status.update(label="Mapping Completato!", state="complete", expanded=False)
            
            st.subheader("🎯 Risultati Finali")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("URL HTML Analizzati", len(df_old))
            col_m2.metric("Match Logici", count_logical)
            col_m3.metric("Match AI", len(final_df) - count_logical)
            
            st.dataframe(final_df)
            
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Scarica CSV Mapping", csv, "redirect_mapping_html_only.csv", "text/csv")
            
            with st.expander("Genera Codice .htaccess"):
                code = "RewriteEngine On\n"
                for _, r in final_df.iterrows():
                    if r['Suggested URL']:
                        try:
                            o_p = urlparse(r['Old URL']).path
                            n_p = urlparse(r['Suggested URL']).path
                            if o_p != n_p:
                                code += f"Redirect 301 {o_p} {n_p}\n"
                        except: pass
                st.code(code)
