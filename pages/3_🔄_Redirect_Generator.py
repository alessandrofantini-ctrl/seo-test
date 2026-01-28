import streamlit as st
import pandas as pd
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="AI Redirect Mapper", layout="wide")

st.title("🔄 Screaming Frog AI Redirect Mapper")
st.markdown("""
Questo strumento utilizza l'Intelligenza Artificiale (Embedding) per mappare i redirect.
Carica i file **CSV o Excel** esportati da **Screaming Frog** (Vecchio Sito e Nuovo Sito).
L'AI leggerà URL, Title e H1 per trovare la corrispondenza semantica perfetta.
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configurazione")
    openai_api_key = st.text_input("OpenAI Key", type="password")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    
    threshold = st.slider("Soglia di Confidenza Minima", 0.0, 1.0, 0.75, help="Sotto questo valore, il redirect non verrà suggerito.")

# --- FUNZIONI ---

def load_sf_data(uploaded_file):
    """Carica dati da CSV o Excel (Screaming Frog) in modo universale."""
    df = None
    filename = uploaded_file.name
    
    # 1. CARICAMENTO DATI
    try:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            # Caricamento EXCEL
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            # Caricamento CSV (Tentativi multipli di encoding)
            encodings_to_try = ['utf-8', 'ISO-8859-1', 'cp1252', 'utf-16']
            for encoding in encodings_to_try:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception:
                    break
    except Exception as e:
        st.error(f"Errore lettura file: {e}")
        return None

    if df is None:
        st.error("Impossibile leggere il file. Verifica che sia un CSV o Excel valido.")
        return None

    # 2. NORMALIZZAZIONE E PULIZIA
    try:
        # Pulisce nomi colonne
        df.columns = df.columns.str.strip()
        cols_lower = df.columns.str.lower()
        
        # Cerca le colonne chiave di Screaming Frog (indipendentemente dalla lingua)
        col_url_idx = next((i for i, c in enumerate(cols_lower) if "address" in c or "indirizzo" in c), None)
        # Cerca Title e H1 (esclude colonne tipo 'Title 1 Length')
        col_title_idx = next((i for i, c in enumerate(cols_lower) if ("title 1" in c or "titolo 1" in c) and "length" not in c), None)
        col_h1_idx = next((i for i, c in enumerate(cols_lower) if "h1-1" in c and "length" not in c), None)
        
        if col_url_idx is None:
            st.error(f"Colonna 'Address' non trovata nel file {filename}.")
            return None
            
        # Crea DataFrame pulito
        clean_df = pd.DataFrame()
        clean_df['url'] = df.iloc[:, col_url_idx]
        clean_df['title'] = df.iloc[:, col_title_idx].fillna("") if col_title_idx is not None else ""
        clean_df['h1'] = df.iloc[:, col_h1_idx].fillna("") if col_h1_idx is not None else ""
        
        # Filtro status code 200 (se presente)
        col_status_idx = next((i for i, c in enumerate(cols_lower) if "status code" in c), None)
        if col_status_idx is not None:
            status_series = pd.to_numeric(df.iloc[:, col_status_idx], errors='coerce')
            clean_df = clean_df[status_series == 200]
            
        return clean_df

    except Exception as e:
        st.error(f"Errore elaborazione dati: {e}")
        return None

def get_embedding_batch(text_list, client):
    """Genera embedding in batch per risparmiare tempo e chiamate API."""
    try:
        # Tronca testi troppo lunghi per evitare errori API (limite token)
        clean_texts = [str(t)[:8000] for t in text_list] 
        response = client.embeddings.create(
            input=clean_texts,
            model="text-embedding-3-small"
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        st.error(f"Errore API OpenAI: {e}")
        return []

# --- INTERFACCIA UTENTE ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Vecchio Sito (Sorgente)")
    # Accetta sia CSV che XLSX
    old_file = st.file_uploader("Carica Export Vecchio (CSV/Excel)", type=["csv", "xlsx"], key="old")

with col2:
    st.subheader("2. Nuovo Sito (Destinazione)")
    # Accetta sia CSV che XLSX
    new_file = st.file_uploader("Carica Export Nuovo (CSV/Excel)", type=["csv", "xlsx"], key="new")

# LOGICA PRINCIPALE
if old_file and new_file and openai_api_key:
    # Caricamento dati con la nuova funzione universale
    df_old = load_sf_data(old_file)
    df_new = load_sf_data(new_file)
    
    if df_old is not None and df_new is not None:
        st.info(f"Dati caricati correttamente: {len(df_old)} URL vecchi e {len(df_new)} URL nuovi.")
        
        if st.button("🚀 Avvia Matching AI (Embeddings)"):
            client = OpenAI(api_key=openai_api_key)
            status = st.status("Elaborazione in corso...", expanded=True)
            
            # 1. PREPARAZIONE TESTO COMBINATO
            status.write("🧠 Preparazione contesti semantici...")
            df_old['combined_text'] = "URL: " + df_old['url'].astype(str) + " TITLE: " + df_old['title'].astype(str) + " H1: " + df_old['h1'].astype(str)
            df_new['combined_text'] = "URL: " + df_new['url'].astype(str) + " TITLE: " + df_new['title'].astype(str) + " H1: " + df_new['h1'].astype(str)
            
            # 2. GENERAZIONE EMBEDDINGS (BATCH PROCESSING)
            batch_size = 100
            
            # Vecchio Sito
            status.write(f"Vettorializzazione Vecchio Sito ({len(df_old)} URL)...")
            embeddings_old = []
            prog_bar = status.progress(0)
            for i in range(0, len(df_old), batch_size):
                batch = df_old['combined_text'].iloc[i:i+batch_size].tolist()
                emb = get_embedding_batch(batch, client)
                embeddings_old.extend(emb)
                prog_bar.progress(min((i + batch_size) / len(df_old), 1.0))
            
            # Nuovo Sito
            status.write(f"Vettorializzazione Nuovo Sito ({len(df_new)} URL)...")
            embeddings_new = []
            prog_bar.progress(0)
            for i in range(0, len(df_new), batch_size):
                batch = df_new['combined_text'].iloc[i:i+batch_size].tolist()
                emb = get_embedding_batch(batch, client)
                embeddings_new.extend(emb)
                prog_bar.progress(min((i + batch_size) / len(df_new), 1.0))

            # 3. CALCOLO SIMILARITÀ E MATCHING
            status.write("📐 Calcolo matrici di similarità...")
            if embeddings_old and embeddings_new:
                matrix_old = np.array(embeddings_old)
                matrix_new = np.array(embeddings_new)
                
                # Cosine Similarity
                similarity_matrix = cosine_similarity(matrix_old, matrix_new)
                
                matches = []
                for idx, row in enumerate(similarity_matrix):
                    best_match_idx = row.argmax()
                    score = row[best_match_idx]
                    
                    old_url_val = df_old.iloc[idx]['url']
                    
                    # Applica soglia di confidenza
                    if score >= threshold:
                        new_url_val = df_new.iloc[best_match_idx]['url']
                        new_title_val = df_new.iloc[best_match_idx]['title']
                    else:
                        new_url_val = ""
                        new_title_val = ""
                        
                    matches.append({
                        "Old URL": old_url_val,
                        "Old Title": df_old.iloc[idx]['title'],
                        "Suggested URL": new_url_val,
                        "New Title": new_title_val,
                        "Confidence": round(score * 100, 2)
                    })
                
                result_df = pd.DataFrame(matches)
                status.update(label="Mapping Completato!", state="complete", expanded=False)
                
                # --- VISUALIZZAZIONE RISULTATI ---
                st.subheader("🎯 Risultati Mapping")
                
                matched_count = len(result_df[result_df['Suggested URL'] != ""])
                st.metric("URL Mappati con Successo", f"{matched_count} / {len(df_old)}", help=f"Match con confidenza > {threshold*100}%")
                
                st.dataframe(result_df)
                
                # DOWNLOAD CSV
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Scarica CSV Mapping", csv, "ai_redirect_mapping.csv", "text/csv")
                
                # GENERAZIONE HTACCESS
                st.markdown("---")
                st.subheader("📝 Codice .htaccess (Anteprima)")
                htaccess_code = "# AI Generated Redirects 301\nRewriteEngine On\n\n"
                
                from urllib.parse import urlparse
                for _, row in result_df.iterrows():
                    if row['Suggested URL']:
                        try:
                            # Usa path relativi per evitare errori su domini diversi
                            old_path = urlparse(str(row['Old URL'])).path
                            new_path = urlparse(str(row['Suggested URL'])).path
                            # Pulisce spazi vuoti
                            if old_path and new_path:
                                htaccess_code += f"Redirect 301 {old_path} {new_path}\n"
                        except:
                            pass
                
                with st.expander("Visualizza/Copia Codice .htaccess"):
                    st.code(htaccess_code, language="apache")

            else:
                st.error("Errore critico: generazione embeddings fallita.")

elif not openai_api_key:
    st.warning("Inserisci la OpenAI API Key nella sidebar per iniziare.")
