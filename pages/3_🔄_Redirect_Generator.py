def load_sf_data(uploaded_file):
    """Carica dati da CSV o Excel (Screaming Frog) in modo universale."""
    df = None
    
    # 1. RILEVAMENTO TIPO FILE
    filename = uploaded_file.name
    
    try:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            # Caricamento EXCEL
            # engine='openpyxl' è necessario per .xlsx
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            
        else:
            # Caricamento CSV (Logica robusta con tentativi di encoding)
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

    # 2. NORMALIZZAZIONE COLONNE (Identica per CSV e Excel)
    try:
        # Pulisce nomi colonne
        df.columns = df.columns.str.strip()
        cols_lower = df.columns.str.lower()
        
        # Cerca colonne chiave
        col_url_idx = next((i for i, c in enumerate(cols_lower) if "address" in c or "indirizzo" in c), None)
        col_title_idx = next((i for i, c in enumerate(cols_lower) if ("title 1" in c or "titolo 1" in c) and "length" not in c), None)
        col_h1_idx = next((i for i, c in enumerate(cols_lower) if "h1-1" in c and "length" not in c), None)
        
        if col_url_idx is None:
            st.error(f"Colonna 'Address' non trovata nel file {filename}.")
            return None
            
        # Crea DF pulito
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
