import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import time
from docx import Document
from io import BytesIO

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="SEO Content Strategist Pro", layout="wide")

# --- DIZIONARIO MERCATI ---
# Qui puoi aggiungere tutti i paesi che vuoi
MARKETS = {
    "🇮🇹 Italia": {"gl": "it", "hl": "it", "domain": "google.it"},
    "🇺🇸 USA (English)": {"gl": "us", "hl": "en", "domain": "google.com"},
    "🇬🇧 UK": {"gl": "uk", "hl": "en", "domain": "google.co.uk"},
    "🇪🇸 Spagna": {"gl": "es", "hl": "es", "domain": "google.es"},
    "🇫🇷 Francia": {"gl": "fr", "hl": "fr", "domain": "google.fr"},
    "🇩🇪 Germania": {"gl": "de", "hl": "de", "domain": "google.de"},
}

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ SEO Settings")
    
    openai_api_key = st.text_input("OpenAI Key", type="password")
    serp_api_key = st.text_input("SerpApi Key", type="password")
    
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    if not serp_api_key and "SERP_API_KEY" in st.secrets:
        serp_api_key = st.secrets["SERP_API_KEY"]

    st.markdown("---")
    st.subheader("🌍 Impostazioni Mercato")
    # SELETTORE MERCATO
    selected_market_label = st.selectbox("Seleziona Mercato Target", list(MARKETS.keys()))
    market_params = MARKETS[selected_market_label] # Recupera i parametri (es. gl='us')

    st.markdown("---")
    st.subheader("🎯 Target Cliente")
    client_url = st.text_input("URL Sito Cliente (Opzionale)", placeholder="https://www.tuosito.it")
    custom_usp = st.text_area("USP / Punti di Forza", placeholder="Es. Spedizione in 24h...", height=100)
    tone_of_voice = st.selectbox("Tono di Voce", ["Autorevole & Tecnico", "Empatico & Problem Solving", "Diretto & Commerciale"])

# --- MAIN PAGE ---
st.title("🚀 SEO Brief Generator Multi-Country")
st.markdown(f"Analisi impostata su: **{selected_market_label}**")

col1, col2 = st.columns([3, 1])
with col1:
    keyword = st.text_input("Keyword Principale", placeholder="Es. best automated gearbox repair")
with col2:
    target_intent = st.selectbox("Intento", ["Informativo", "Commerciale", "Navigazionale"])

# --- FUNZIONI ---

# 1. MODIFICATA: Ora accetta gl, hl e domain come argomenti
def get_serp_data(query, api_key, gl, hl, domain):
    params = {
        "engine": "google", 
        "q": query, 
        "api_key": api_key, 
        "hl": hl,           # Lingua dinamica
        "gl": gl,           # Paese dinamico
        "google_domain": domain # Dominio dinamico
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

def scrape_site_content(url, is_client=False):
    # Usiamo un User-Agent generico internazionale
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    data = {"url": url, "headers": [], "text_sample": "", "title": ""}
    try:
        resp = requests.get(url, headers=ua, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        data["title"] = soup.title.string.strip() if soup.title else "N/A"
        elements = soup.find_all(['h1', 'h2', 'h3'])
        for tag in elements[:15]:
            data["headers"].append(f"[{tag.name.upper()}] {tag.get_text(strip=True)}")
        paragraphs = soup.find_all('p')
        text_content = " ".join([p.get_text(strip=True) for p in paragraphs[:10]])
        data["text_sample"] = text_content[:1500]
        return data
    except Exception:
        return None

def create_docx(content, kw):
    doc = Document()
    doc.add_heading(f'SEO Brief: {kw}', 0)
    doc.add_paragraph(content)
    bio = BytesIO()
    doc.save(bio)
    return bio

# --- LOGICA ---
if st.button("Avvia Analisi Completa"):
    if not keyword or not openai_api_key or not serp_api_key:
        st.error("Inserisci Keyword e API Keys.")
    else:
        status = st.status(f"Avvio scansione su {selected_market_label}...", expanded=True)
        try:
            # 1. ANALISI CLIENTE
            client_context_str = "Nessun sito cliente fornito. (Generico)"
            if client_url:
                status.write("🏢 Analisi identità cliente...")
                client_data = scrape_site_content(client_url, is_client=True)
                if client_data:
                    client_context_str = f"SITO CLIENTE: {client_url}\nMETA: {client_data['title']}\nTESTO: {client_data['text_sample']}"
            if custom_usp:
                client_context_str += f"\nUSP MANUALI: {custom_usp}"

            # 2. SERP (Passiamo i parametri dinamici)
            status.write(f"🔍 Analisi SERP ({market_params['domain']})...")
            
            # CHIAMATA AGGIORNATA
            serp = get_serp_data(
                keyword, 
                serp_api_key, 
                gl=market_params['gl'], 
                hl=market_params['hl'], 
                domain=market_params['domain']
            )
            
            if serp and "organic_results" in serp:
                urls = [res["link"] for res in serp["organic_results"][:4]]
                
                paa = [q.get("question") for q in serp.get("related_questions", [])]
                related_searches = [r.get("query") for r in serp.get("related_searches", [])]
                
                # 3. COMPETITOR
                status.write("⚔️ Spionaggio Competitor...")
                competitor_text = ""
                bar = status.empty()
                prog = bar.progress(0)
                for i, url in enumerate(urls):
                    prog.progress((i+1)/len(urls))
                    c_data = scrape_site_content(url)
                    if c_data:
                        competitor_text += f"\n--- COMPETITOR: {url} ---\n{c_data['title']}\n" + "\n".join(c_data['headers'])
                    time.sleep(0.1)
                bar.empty()
                
                # 4. AI STRATEGY
                status.write("🧠 Elaborazione Mappa Semantica & Brief...")
                
                system_prompt = "Sei un Senior SEO Strategist Internazionale. Crei contenuti 'Skyscraper' superiori ai competitor."
                
                # Aggiorniamo il prompt per informare l'AI del mercato
                user_prompt = f"""
                OBIETTIVO: Creare il Brief SEO definitivo per la keyword: "{keyword}".
                MERCATO TARGET: {selected_market_label} (Lingua: {market_params['hl']}).
                INTENTO: {target_intent}.
                TONO: {tone_of_voice}.
                
                IMPORTANTE: 
                - Analizza i dati forniti che sono nella lingua del mercato target.
                - Restituisci il Brief in ITALIANO (per il consulente) ma suggerisci H1/H2 e keyword nella LINGUA TARGET ({market_params['hl']}).
                
                ### DATI DI ANALISI (INPUT)
                
                1. IL NOSTRO BRAND (Contesto):
                {client_context_str}
                
                2. I COMPETITOR (SERP {market_params['gl'].upper()}):
                {competitor_text[:8000]}
                
                3. BISOGNI UTENTE (PAA):
                {", ".join(paa) if paa else "Nessuna domanda specifica rilevata."}
                
                4. ARGOMENTI CORRELATI:
                {", ".join(related_searches) if related_searches else "Nessuna correlata rilevata."}
                
                ### TASK: GENERA IL BRIEF
                
                Restituisci un output strutturato in Markdown:
                
                **SEZIONE A: ANALISI SEMANTICA**
                - **Primary Keyword**: {keyword}
                - **Keywords Secondarie (in {market_params['hl']})**: Elenca 5-10 termini.
                - **Gap Analysis**: Cosa manca ai competitor in questo mercato specifico?
                
                **SEZIONE B: STRUTTURA DEL CONTENUTO**
                Crea una scaletta. 
                - **H1 (in {market_params['hl']})**: Titolo ottimizzato.
                - **H2/H3 (in {market_params['hl']})**: Usa PAA e Correlate.
                - **Istruzioni Copy (in Italiano)**: Spiega cosa scrivere per ogni paragrafo.
                """
                
                client = OpenAI(api_key=openai_api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    temperature=0.7
                )
                output = resp.choices[0].message.content
                
                status.update(label="Strategia Pronta!", state="complete", expanded=False)
                st.markdown(output)
                
                st.session_state['ultimo_brief'] = output
                st.session_state['client_url_session'] = client_url 
                
                docx = create_docx(output, keyword)
                st.download_button("📥 Scarica Brief .docx", docx, f"brief_{keyword.replace(' ','_')}.docx")
            
            else:
                status.update(label="Errore SerpApi", state="error")
                st.error("Nessun dato trovato da Google. Verifica la SerpApi Key.")

        except Exception as e:
            status.update(label="Errore", state="error")
            st.error(f"Errore: {e}")
