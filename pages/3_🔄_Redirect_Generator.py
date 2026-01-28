import streamlit as st
import pandas as pd
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
from io import BytesIO
import re

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Auto-Home Redirect Mapper", layout="wide")

st.title("🌐 Auto-Home AI Redirect Mapper")
st.markdown("""
Strumento avanzato per migrazioni internazionali.
1. **AI Semantic 🧠**: Cerca la pagina specifica equivalente.
2. **Fallback Inglese 🇬🇧**: Se non esiste in lingua locale, cerca la versione inglese.
3. **Smart Home Catch-All 🏠**: Se nessun match, reindirizza automaticamente alla **Homepage della lingua corretta** (es. /it/ -> /it/home).
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configurazione")
    openai_api_key = st.text_input("OpenAI Key", type="password")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    
    st.markdown("---")
    st.subheader("Soglie di Confidenza")
    threshold_primary = st.slider("Match Esatto (Contenuto)", 0.0, 1.0, 0.80)
    threshold_fallback = st.slider("Fallback Inglese", 0.0, 1.0, 0.75)
    
    st.markdown("---")
    st.subheader("🚨 Rete di Sicurezza")
    use_auto_home = st.checkbox("Usa Home Lingua come fallback", value=True, help="Se l'AI non trova nulla, manda l'utente alla Homepage della sua lingua.")

# --- FUNZIONI ---

def load_single_file(uploaded_file):
    try:
        filename = uploaded_file.name
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            return pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            encodings = ['utf-8', 'ISO-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    uploaded_file.seek(0)
                    return pd.read_csv(uploaded_file, encoding=enc, low_memory=False)
                except: continue
            return None
    except Exception as e:
        st.error(f"Errore lettura {uploaded_file.name}: {e}")
        return None

def process_dataframe(df, filename):
    if df is None: return None
    try:
        df.columns = df.columns.str.strip()
        cols_lower = df.columns.str.lower()
        orig_cols = df.columns
        
        # Filtro HTML
        col_type_idx = next((i for i, c in enumerate(cols_lower) if "content" in c and "type" in c), None)
        if col_type_idx is not None:
            col_name = orig_cols[col_type_idx]
            df = df[df[col_name].astype(str).str.contains("html", case=False, na=False)]
        
        # Mappatura
        col_url = next((c for c in cols_lower if "address" in c or "indirizzo" in c), None)
        col_title = next((c for c in cols_lower if ("title 1" in c or "titolo 1" in c) and "len" not in c), None)
        col_h1 = next((c for c in cols_lower if "h1-1" in c and "len" not in c), None)
        col_content = next((c for c in cols_lower if ("content" in c or "body" in c or "text" in c or "testo" in c) and "type" not in c), None)
        
        if not col_url:
            st.warning(f"File '{filename}' saltato: Address mancante.")
            return None
            
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
        st.error(f"Errore processamento {filename}: {e}")
        return None

def load_multiple_files(uploaded_files):
    all_dfs = []
    for file in uploaded_files:
        raw_df = load_single_file(file)
        processed_df = process_dataframe(raw_df, file.name)
        if processed_df is not None:
            all_dfs.append(processed_df)
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return None

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

def detect_language_code(url):
    """Rileva lingua da URL."""
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path.lower()
    
    if domain.endswith(".it"): return "it"
    if domain.endswith(".es"): return "es"
    if domain.endswith(".fr"): return "fr"
    if domain.endswith(".de"): return "de"
    if domain.endswith(".co.uk") or domain.endswith(".uk"): return "en"
    
    segments = path.split('/')
    if len(segments) > 1:
        first = segments[1]
        if first in ['it', 'it-it']: return "it"
        if first in ['en', 'en-us', 'en-gb', 'uk']: return "en"
        if first in ['es', 'es-es']: return "es"
        if first in ['fr', 'fr-fr']: return "fr"
        if first in ['de', 'de-de']: return "de"
    
    # Se non specificato, assumiamo sia il dominio principale (spesso EN o lingua base)
    # Restituiamo 'base' per gestirlo genericamente
    return "base"

def make_context_string(row, lang_code):
    tag = f"LANGUAGE_{lang_code.upper()}"
    ctx = f"[{tag}] URL: {row['url']} | TITLE: {row['title']} | H1: {row['h1']}"
    if row['body_text']:
        ctx += f" | CONTENT: {row['body_text'][:800]}"
    return ctx

# --- INTERFACCIA ---

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Vecchi Siti")
    old_files = st.file_uploader("Sorgente (CSV/Excel)", type=["csv", "xlsx"], accept_multiple_files=True, key="old")
with col2:
    st.subheader("2. Nuovi Siti")
    new_files = st.file_uploader("Destinazione (CSV/Excel)", type=["csv", "xlsx"], accept_multiple_files=True, key="new")

if old_files and new_files:
    df_old = load_multiple_files(old_files)
    df_new = load_multiple_files(new_files)
    
    if df_old is not None and df_new is not None:
        
        st.info(f"📚 **Analisi**: {len(df_old)} URL Sorgente vs {len(df_new)} URL Destinazione.")

        if st.button("🚀 Avvia Matching Intelligente"):
            
            if not openai_api_key:
                st.error("Inserisci API Key.")
            else:
                status = st.status("Analisi Struttura e Lingue...", expanded=True)
                
                # 1. RILEVAMENTO LINGUE
                df_old['lang'] = df_old['url'].apply(detect_language_code)
                df_new['lang'] = df_new['url'].apply(detect_language_code)
                
                # --- IDENTIFICAZIONE HOMEPAGE PER OGNI LINGUA ---
                # Logica: La URL più corta per ogni codice lingua è probabilmente la Home.
                language_homes = {}
                unique_langs = df_new['lang'].unique()
                
                for lang in unique_langs:
                    # Filtra URL di quella lingua
                    lang_urls = df_new[df_new['lang'] == lang]['url'].tolist()
                    if lang_urls:
                        # Trova la più corta
                        shortest = min(lang_urls, key=len)
                        language_homes[lang] = shortest
                
                status.write(f"🏠 Homepage Rilevate: {language_homes}")
                
                # 2. CONTEXT & EMBEDDING
                status.write("🧠 Generazione Embeddings...")
                client = OpenAI(api_key=openai_api_key)
                
                df_old['ctx'] = df_old.apply(lambda r: make_context_string(r, r['lang']), axis=1)
                df_new['ctx'] = df_new.apply(lambda r: make_context_string(r, r['lang']), axis=1)
                
                # Embedding Old
                emb_old = []
                batch_s = 100
                prog = status.progress(0)
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

                # 3. MATCHING LOGIC
                status.write("🔍 Ricerca Corrispondenze & Catch-All...")
                results = []
                
                if emb_old and emb_new:
                    mat_old = np.array(emb_old)
                    mat_new = np.array(emb_new)
                    sims = cosine_similarity(mat_old, mat_new)
                    
                    eng_indices = df_new.index[df_new['lang'] == 'en'].tolist()
                    
                    for i, vector_idx in enumerate(df_old.index):
                        row_old = df_old.loc[vector_idx]
                        old_lang = row_old['lang']
                        
                        best_match_url = ""
                        best_score = 0.0
                        method = "Nessuno (404)"
                        
                        # --- A: MATCH CONTENUTO (AI) ---
                        # Filtra solo target stessa lingua
                        same_lang_indices = df_new.index[df_new['lang'] == old_lang].tolist()
                        
                        if same_lang_indices:
                            scores_same = sims[i, same_lang_indices]
                            if len(scores_same) > 0:
                                local_idx = scores_same.argmax()
                                local_score = scores_same[local_idx]
                                if local_score >= threshold_primary:
                                    best_match_url = df_new.loc[same_lang_indices[local_idx], 'url']
                                    best_score = local_score
                                    method = f"AI Match ({old_lang})"

                        # --- B: FALLBACK CONTENUTO EN (AI) ---
                        if not best_match_url and eng_indices and old_lang != 'en':
                            scores_eng = sims[i, eng_indices]
                            if len(scores_eng) > 0:
                                local_idx = scores_eng.argmax()
                                local_score = scores_eng[local_idx]
                                if local_score >= threshold_fallback:
                                    best_match_url = df_new.loc[eng_indices[local_idx], 'url']
                                    best_score = local_score
                                    method = "AI Fallback EN"
                        
                        # --- C: AUTO-HOME CATCH-ALL (SAFETY NET) ---
                        if not best_match_url and use_auto_home:
                            # Cerca la home della lingua di origine
                            if old_lang in language_homes:
                                best_match_url = language_homes[old_lang]
                                method = f"Auto-Home ({old_lang})"
                                best_score = 0.1 # Basso score convenzionale
                            
                            # Se non trova la lingua (es. lingua vecchia rimossa), prova la Home Base/Inglese
                            elif 'base' in language_homes:
                                best_match_url = language_homes['base']
                                method = "Auto-Home (Base)"
                            elif 'en' in language_homes:
                                best_match_url = language_homes['en']
                                method = "Auto-Home (EN)"

                        results.append({
                            "Old URL": row_old['url'],
                            "New URL": best_match_url,
                            "Confidence": round(best_score * 100, 1),
                            "Method": method,
                            "Detected Lang": old_lang
                        })

                # --- EXPORT ---
                final_df = pd.DataFrame(results)
                status.update(label="Fatto!", state="complete", expanded=False)
                
                st.subheader("🎯 Report Finale")
                
                # Metrics
                total = len(final_df)
                matched = len(final_df[final_df['Method'].str.contains("AI Match") | final_df['Method'].str.contains("Fallback")])
                homes = len(final_df[final_df['Method'].str.contains("Auto-Home")])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Totale URL", total)
                c2.metric("Redirect Specifici (AI)", matched)
                c3.metric("Redirect Generici (Home)", homes)
                
                st.dataframe(final_df.head(50))
                
                export_df = final_df[final_df['New URL'] != ""][['Old URL', 'New URL']]
                excel_data = to_excel(export_df)
                
                st.download_button(
                    label="📥 Scarica Excel Unificato",
                    data=excel_data,
                    file_name="redirect_map_final.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
