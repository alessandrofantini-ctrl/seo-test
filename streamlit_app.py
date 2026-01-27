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

# --- SIDEBAR PER LE API KEY ---
with st.sidebar:
    st.header("Configurazione API")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    serp_api_key = st.text_input("SerpApi Key", type="password")
    
    # Fallback sui Secrets di Streamlit
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    if not serp_api_key and "SERP_API_KEY" in st.secrets:
        serp_api_key = st.secrets["SERP_API_KEY"]

# --- INPUT UTENTE ---
keyword = st.text_input("Inserisci keyword...", placeholder="Es. riorganizzazione aziendale")

# --- FUNZIONI DI UTILITÀ ---

def get_serp_data(query, api_key):
    """Recupera i dati da SerpApi."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "hl": "it",
        "gl": "it"
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Errore durante la chiamata a SerpApi: {e}")
        return None

def scrape_competitor_structure(url):
    """Estrae H1, H2, H3 da un URL."""
    headers_list = []
    ua = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=ua, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        elements = soup.find_all(['h1', 'h2', 'h3'])
        if elements:
            headers_list.append(f"\n--- STRUTTURA SITO: {url} ---")
            for tag in elements[:15]: # Prendi max 15 header per sito
                text = tag.get_text(strip=True)
                if text:
                    headers_list.append(f"[{tag.name.upper()}] {text}")
        return headers_list
    except Exception:
        return []

# --- LOGICA PRINCIPALE ---

if st.button("Genera Scaletta"):
    if not keyword:
        st.warning("Inserisci una keyword per procedere.")
    elif not openai_api_key or not serp_api_key:
        st.error("Mancano le API Key. Inseriscile nella sidebar o nei Secrets.")
    else:
        # Placeholder per lo stato
        status_box = st.status("Inizio analisi...", expanded=True)
        
        try:
            # 1. SERPAPI
            status_box.write("🔍 Interrogazione Google (SerpApi)...")
            serp_data = get_serp_data(keyword, serp_api_key)
            
            if not serp_data:
                status_box.update(label="Errore SerpApi", state="error")
                st.stop()

            # Preparazione dati
            info_per_ai = []
            urls_to_scrape = []
            
            if "related_questions" in serp_data:
                paa = [q.get("question") for q in serp_data["related_questions"]]
                info_per_ai.append(f"DOMANDE PAA: {', '.join(paa)}")
            
            if "organic_results" in serp_data:
                urls_to_scrape = [res.get("link") for res in serp_data["organic_results"][:5]]

            # 2. SCRAPING
            status_box.write(f"🕷️ Scraping di {len(urls_to_scrape)} competitor...")
            strutture_competitor = []
            
            my_bar = status_box.empty() # Barra di progresso interna allo status
            prog_bar = my_bar.progress(0)

            for i, url in enumerate(urls_to_scrape):
                prog_bar.progress((i + 1) / len(urls_to_scrape))
                extracted = scrape_competitor_structure(url)
                strutture_competitor.extend(extracted)
                time.sleep(0.2)
            
            my_bar.empty() # Rimuovi barra progresso

            # 3. OPENAI
            status_box.write("🤖 Generazione scaletta con AI...")
            
            # Preparazione Prompt
            sito_target = "https://www.dieselcarbyfinazzi.it/"
            testo_domande = "\n".join(info_per_ai)
            testo_competitor_raw = "\n".join(strutture_competitor)
            
            # --- PROTEZIONE ERRORI DI CONNESSIONE ---
            # Se il testo è troppo lungo, OpenAI va in timeout o errore. Tagliamolo a 15k caratteri.
            if len(testo_competitor_raw) > 15000:
                testo_competitor = testo_competitor_raw[:15000] + "\n...[Testo troncato per limiti di lunghezza]..."
            else:
                testo_competitor = testo_competitor_raw

            prompt_sistema = "Sei un SEO Specialist esperto. Crea una struttura H1-H4 dettagliata."
            prompt_utente = f"""### CONTESTO
Keyword: "{keyword}"
Sito Target: {sito_target}

### DATI SERP
{testo_domande}

### COMPETITOR (Struttura H-TAG)
{testo_competitor}

### OBIETTIVO
Crea una scaletta ottimizzata (H1, H2, H3, H4) superiore ai competitor.
Restituisci solo la struttura e una breve conclusione.
"""

            # Chiamata specifica a OpenAI con gestione errori dedicata
            try:
                client = OpenAI(api_key=openai_api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": prompt_sistema},
                        {"role": "user", "content": prompt_utente}
                    ],
                    timeout=45 # Timeout esplicito aumentato
                )
                
                risultato_ai = response.choices[0].message.content
                
                status_box.update(label="Analisi Completata!", state="complete", expanded=False)
                
                st.markdown("### 📝 Scaletta Generata")
                st.markdown("---")
                st.markdown(risultato_ai)

            except Exception as e_ai:
                status_box.update(label="Errore AI", state="error")
                st.error(f"❌ Errore durante la comunicazione con OpenAI: {e_ai}")
                st.info("Suggerimento: Controlla che la API Key sia corretta e di avere credito sufficiente.")

        except Exception as e_gen:
            status_box.update(label="Errore Generale", state="error")
            st.error(f"Errore imprevisto: {e_gen}")
