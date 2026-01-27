import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import time

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="SEO Outline Generator", layout="centered")

# --- TITOLO ---
st.title("SEO Header Analyzer")
st.markdown("Genera una struttura di header tag ottimizzata analizzando la SERP e i competitor.")

# --- SIDEBAR PER LE API KEY (Per sicurezza su Streamlit Cloud) ---
with st.sidebar:
    st.header("Configurazione API")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    serp_api_key = st.text_input("SerpApi Key", type="password")
    
    # Se le chiavi sono salvate nei secrets di Streamlit, usale come fallback
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    if not serp_api_key and "SERP_API_KEY" in st.secrets:
        serp_api_key = st.secrets["SERP_API_KEY"]

# --- INPUT UTENTE ---
keyword = st.text_input("Inserisci keyword...", placeholder="Es. manutenzione cambio automatico")

# --- FUNZIONI DI UTILITÀ ---

def get_serp_data(query, api_key):
    """Recupera i dati da SerpApi."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "hl": "it",
        "gl": "it" # Geolocalizzazione Italia
    }
    
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Errore SerpApi: {e}")
        return None

def scrape_competitor_structure(url):
    """Estrae H1, H2, H3 da un URL."""
    headers_list = []
    
    # User Agent per simulare un browser reale (come nel codice C#)
    ua = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        # Timeout breve per non bloccare l'app troppo a lungo
        resp = requests.get(url, headers=ua, timeout=10)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Trova tutti gli header h1, h2, h3
        elements = soup.find_all(['h1', 'h2', 'h3'])
        
        if elements:
            headers_list.append(f"\n--- STRUTTURA SITO: {url} ---")
            # Prendi solo i primi 15 header come nel codice originale
            for tag in elements[:15]:
                text = tag.get_text(strip=True)
                if text:
                    headers_list.append(f"[{tag.name.upper()}] {text}")
                    
        return headers_list
    except Exception:
        # Ignora errori su singoli siti (timeout, 403, ecc.)
        return []

# --- LOGICA PRINCIPALE ---

if st.button("Genera Scaletta"):
    if not keyword:
        st.warning("Inserisci una keyword per procedere.")
    elif not openai_api_key or not serp_api_key:
        st.error("Mancano le API Key. Inseriscile nella sidebar.")
    else:
        # Container per mostrare lo stato di avanzamento
        status_text = st.empty()
        
        try:
            # 1. RECUPERO DATI DA SERPAPI
            status_text.info("🔍 Analisi SERP in corso (SerpApi)...")
            
            serp_data = get_serp_data(keyword, serp_api_key)
            
            if serp_data:
                info_per_ai = []
                urls_to_scrape = []
                
                # People Also Ask
                if "related_questions" in serp_data:
                    paa = [q.get("question") for q in serp_data["related_questions"]]
                    info_per_ai.append(f"DOMANDE UTENTI (PAA): {', '.join(paa)}")
                
                # Ricerche Correlate
                if "related_searches" in serp_data:
                    related = [r.get("query") for r in serp_data["related_searches"]]
                    info_per_ai.append(f"RICERCHE CORRELATE: {', '.join(related)}")
                
                # Risultati Organici (Prendiamo i primi 5 link)
                if "organic_results" in serp_data:
                    urls_to_scrape = [res.get("link") for res in serp_data["organic_results"][:5]]

                # 2. SCRAPING COMPETITOR
                status_text.info(f"🕷️ Scraping di {len(urls_to_scrape)} siti competitor...")
                
                strutture_competitor = []
                progress_bar = st.progress(0)
                
                for i, url in enumerate(urls_to_scrape):
                    # Aggiorna progress bar
                    progress_bar.progress((i + 1) / len(urls_to_scrape))
                    
                    extracted_headers = scrape_competitor_structure(url)
                    strutture_competitor.extend(extracted_headers)
                    time.sleep(0.5) # Piccola pausa per cortesia
                
                progress_bar.empty()

                # 3. GENERAZIONE CON OPENAI
                status_text.info("🤖 L'AI sta creando la tua scaletta ottimizzata...")
                
                client = OpenAI(api_key=openai_api_key)
                
                sito_target = "https://www.dieselcarbyfinazzi.it/"
                testo_domande = "\n".join(info_per_ai)
                testo_competitor = "\n".join(strutture_competitor)
                
                prompt_sistema = "Sei un SEO Specialist esperto di copywriting e ottimizzazione contenuti. Il tuo obiettivo è creare una struttura di header tag dettagliata per il sito https://www.dieselcarbyfinazzi.it/."
                
                prompt_utente = f"""### CONTESTO SEO
Obiettivo: Generare la struttura degli Header Tag (H1-H4) per un nuovo articolo.
Keyword Principale: "{keyword}"
Sito Target: {sito_target}

### ANALISI DATI SERP
Di seguito trovi le domande frequenti e le tendenze estratte da Google:
{testo_domande}

### ANALISI COMPETITOR
Ecco la struttura degli header dei siti attualmente posizionati:
{testo_competitor}

### DIRETTIVE DI SCRITTURA
- Crea un H1 magnetico che includa la keyword principale.
- Organizza i paragrafi H2 in modo logico per rispondere all'intento di ricerca.
- Usa H3 e H4 per approfondire dettagli tecnici o rispondere a FAQ specifiche.
- Assicurati che la struttura sia superiore a quella dei competitor analizzati.

### FORMATO OUTPUT
Restituisci solo la struttura gerarchica (H1, H2, H3, H4) e una breve conclusione.
"""

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": prompt_sistema},
                        {"role": "user", "content": prompt_utente}
                    ]
                )
                
                risultato_ai = response.choices[0].message.content
                
                # Pulizia UI finale
                status_text.success("Generazione completata!")
                
                # Output
                st.markdown("### 📝 Scaletta Generata")
                st.markdown("---")
                st.markdown(risultato_ai)
                
            else:
                status_text.error("Nessun dato ricevuto da SerpApi.")
                
        except Exception as e:
            status_text.error(f"Si è verificato un errore generale: {str(e)}")
