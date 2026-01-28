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
Carica i file CSV esportati da **Screaming Frog** (Vecchio Sito e Nuovo Sito).
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

def load_sf_csv(uploaded_file):
    """Carica e pulisce il CSV di Screaming Frog."""
    try:
        # Screaming Frog spesso esporta con la prima riga inutile o header complessi
        # Proviamo a leggere standard, se fallisce cerca colonne specifiche
        df = pd.read_csv(uploaded_file, low_memory=False)
        
        # Standardizzazione nomi colonne (Screaming Frog in diverse lingue/versioni)
        # Cerchiamo colonne che contengono "Address", "Title", "H1"
        cols = df.columns.str.lower()
        
        col_url = next((c for c in df.columns if "address" in c.lower() or "indirizzo" in c.lower()), None)
        col_title = next((c for c in df.columns if "title 1" in c.lower() or "titolo 1" in c.lower()), None)
        col_h1 = next((c for c in df.columns if "h1-1" in c.lower()), None)
        
        if not col_url:
            st.error("Non trovo la colonna 'Address' nel CSV.")
            return None
            
        # Creiamo un DF pulito
        clean_df = pd.DataFrame()
        clean_df['url'] = df[col_url]
        clean_df['title'] = df[col_title].fillna("") if col_title else ""
        clean_df['h1'] = df[col_h1].fillna("") if col_h1 else ""
        
        # Filtriamo solo pagine HTML (status 200) se presente colonna status code
        col_status = next((c for c in df.columns if "status code" in c.lower()), None)
        if col_status:
            clean_df = clean_df[df[col_status] == 200]
            
        return clean_df
    except Exception as e:
        st.error(f"Errore lettura CSV: {e}")
        return None

def get_embedding_batch(text_list, client):
    """Genera embedding in batch per risparmiare tempo."""
    # OpenAI supporta batch, ma per semplicità usiamo liste.
    # Modello: text-embedding-3-small (economico e performante)
    try:
        # Pulizia base: tronca testi troppo lunghi
        clean_texts = [str(t)[:8000] for t in text_list] 
        response = client.embeddings.create(
            input=clean_texts,
            model="text-embedding-3-small"
        )
        # Ordina gli embedding in base all'indice per sicurezza
        embeddings = [data.embedding for data in response.data]
        return embeddings
    except Exception as e:
        st.error(f"Errore API OpenAI: {e}")
        return []

# --- INTERFACCIA ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Vecchio Sito (Sorgente)")
    old_file = st.file_uploader("Carica CSV Screaming Frog VECCHIO", type=["csv"], key="old")

with col2:
    st.subheader("2. Nuovo Sito (Destinazione)")
    new_file = st.file_uploader("Carica CSV Screaming Frog NUOVO", type=["csv"], key="new")

if old_file and new_file and openai_api_key:
    df_old = load_sf_csv(old_file)
    df_new = load_sf_csv(new_file)
    
    if df_old is not None and df_new is not None:
        st.info(f"Caricati: {len(df_old)} URL vecchi e {len(df_new)} URL nuovi.")
        
        if st.button("🚀 Avvia Matching AI (Embeddings)"):
            client = OpenAI(api_key=openai_api_key)
            status = st.status("Elaborazione in corso...", expanded=True)
            
            # 1. PREPARAZIONE DATI (Stringa Combinata)
            # Uniamo URL + Title + H1 per dare contesto all'AI
            status.write("🧠 Preparazione contesti semantici...")
            df_old['combined_text'] = "URL: " + df_old['url'] + " TITLE: " + df_old['title'] + " H1: " + df_old['h1']
            df_new['combined_text'] = "URL: " + df_new['url'] + " TITLE: " + df_new['title'] + " H1: " + df_new['h1']
            
            # 2. GENERAZIONE EMBEDDINGS (VECCHIO SITO)
            status.write(f"Vettorializzazione Vecchio Sito ({len(df_old)} URL)...")
            # Facciamo batch di 100 per non intasare
            batch_size = 100
            embeddings_old = []
            
            prog_bar = status.progress(0)
            for i in range(0, len(df_old), batch_size):
                batch = df_old['combined_text'].iloc[i:i+batch_size].tolist()
                emb = get_embedding_batch(batch, client)
                embeddings_old.extend(emb)
                prog_bar.progress(min((i + batch_size) / len(df_old), 1.0))
            
            # 3. GENERAZIONE EMBEDDINGS (NUOVO SITO)
            status.write(f"Vettorializzazione Nuovo Sito ({len(df_new)} URL)...")
            embeddings_new = []
            prog_bar.progress(0)
            for i in range(0, len(df_new), batch_size):
                batch = df_new['combined_text'].iloc[i:i+batch_size].tolist()
                emb = get_embedding_batch(batch, client)
                embeddings_new.extend(emb)
                prog_bar.progress(min((i + batch_size) / len(df_new), 1.0))

            # 4. CALCOLO SIMILARITÀ (COSINE SIMILARITY)
            status.write("📐 Calcolo matrici di similarità...")
            if embeddings_old and embeddings_new:
                matrix_old = np.array(embeddings_old)
                matrix_new = np.array(embeddings_new)
                
                # Calcola la similarità tra TUTTI i vecchi e TUTTI i nuovi
                similarity_matrix = cosine_similarity(matrix_old, matrix_new)
                
                # Trova il match migliore per ogni URL vecchio
                matches = []
                for idx, row in enumerate(similarity_matrix):
                    best_match_idx = row.argmax()
                    score = row[best_match_idx]
                    
                    old_url_val = df_old.iloc[idx]['url']
                    
                    if score >= threshold:
                        new_url_val = df_new.iloc[best_match_idx]['url']
                    else:
                        new_url_val = "" # Nessun match sopra la soglia
                        
                    matches.append({
                        "Old URL": old_url_val,
                        "Old Title": df_old.iloc[idx]['title'],
                        "Suggested URL": new_url_val,
                        "New Title": df_new.iloc[best_match_idx]['title'] if score >= threshold else "",
                        "Confidence": round(score * 100, 2)
                    })
                
                result_df = pd.DataFrame(matches)
                status.update(label="Mapping Completato!", state="complete", expanded=False)
                
                # --- RISULTATI ---
                st.subheader("🎯 Risultati Mapping")
                
                # Metriche
                matched_count = len(result_df[result_df['Suggested URL'] != ""])
                st.metric("URL Mappati con Successo", f"{matched_count} / {len(df_old)}", help=f"Match con confidenza > {threshold*100}%")
                
                st.dataframe(result_df)
                
                # DOWNLOAD
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Scarica CSV Mapping", csv, "ai_redirect_mapping.csv", "text/csv")
                
                # GENERATORE HTACCESS RAPIDO
                st.markdown("---")
                st.subheader("📝 Codice .htaccess Rapido")
                htaccess_code = "# AI Generated Redirects\nRewriteEngine On\n\n"
                for _, row in result_df.iterrows():
                    if row['Suggested URL']:
                        try:
                            # Estrae path relativi per pulizia
                            from urllib.parse import urlparse
                            old_path = urlparse(row['Old URL']).path
                            new_path = urlparse(row['Suggested URL']).path
                            htaccess_code += f"Redirect 301 {old_path} {new_path}\n"
                        except:
                            pass
                
                with st.expander("Visualizza Codice .htaccess"):
                    st.code(htaccess_code, language="apache")

            else:
                st.error("Errore nella generazione degli embeddings.")

elif not openai_api_key:
    st.warning("Inserisci la OpenAI API Key nella sidebar per iniziare.")
