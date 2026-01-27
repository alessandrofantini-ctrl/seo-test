import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import time
from docx import Document
from io import BytesIO

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="SEO Content Strategist Pro", layout="wide")

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
    st.subheader("🎯 Target Cliente")
    client_url = st.text_input("URL Sito Cliente (Opzionale)", placeholder="https://www.tuosito.it")
    custom_usp = st.text_area("USP / Punti di Forza", placeholder="Es. Officina autorizzata Bosch...", height=100)
    tone_of_voice = st.selectbox("Tono di Voce", ["Autorevole & Tecnico", "Empatico & Problem Solving", "Diretto & Commerciale"])

# --- MAIN PAGE ---
st.title("🚀 SEO Brief Generator")

col1, col2 = st.columns([3, 1])
with col1:
    keyword = st.text_input("Keyword Principale", placeholder="Es. manutenzione cambio automatico")
with col2:
    target_intent = st.selectbox("Intento", ["Informativo", "Commerciale", "Navigazionale"])

# --- FUNZIONI ---
def get_serp_data(query, api_key):
    params = {"engine": "google", "q": query, "api_key": api_key, "hl": "it", "gl": "it"}
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

def scrape_site_content(url, is_client=False):
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
        status = st.status("Avvio motori di ricerca...", expanded=True)
        try:
            # 1. ANALISI CLIENTE
            client_context_str = "Nessun sito cliente fornito."
            if client_url:
                status.write("🏢 Analisi identità cliente...")
                client_data = scrape_site_content(client_url, is_client=True)
                if client_data:
                    client_context_str = f"SITO CLIENTE: {client_url}\nMETA: {client_data['title']}\nTESTO: {client_data['text_sample']}"
            if custom_usp:
                client_context_str += f"\nUSP MANUALI: {custom_usp}"

            # 2. SERP
            status.write("🔍 Analisi SERP Google...")
            serp = get_serp_data(keyword, serp_api_key)
            
            if serp and "organic_results" in serp:
                urls = [res["link"] for res in serp["organic_results"][:4]]
                paa = [q["question"] for q in serp.get("related_questions", [])]
                
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
                
                # 4. AI
                status.write("🧠 Elaborazione Brief Strategico...")
                system_prompt = "Sei un Head of SEO. Crei brief editoriali che posizionano E convertono."
                user_prompt = f"""
                OBIETTIVO: SEO Brief per "{keyword}". Intento: {target_intent}. Tono: {tone_of_voice}.
                CLIENTE: {client_context_str}
                COMPETITOR: {competitor_text[:10000]}
                PAA: {", ".join(paa)}
                
                OUTPUT:
                1. Concept Strategico & Gap Analysis.
                2. Target.
                3. Struttura H1, H2, H3 con direttive copywriter e USP cliente.
                Usa Markdown.
                """
                
                client = OpenAI(api_key=openai_api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                )
                output = resp.choices[0].message.content
                
                status.update(label="Strategia Pronta!", state="complete", expanded=False)
                st.markdown(output)
                
                # --- SALVATAGGIO PER PAGINA 2 ---
                st.session_state['ultimo_brief'] = output
                st.success("✅ Brief salvato! Vai alla pagina 'Redattore Articoli' per generare il testo.")
                
                docx = create_docx(output, keyword)
                st.download_button("📥 Scarica Brief .docx", docx, f"brief_{keyword.replace(' ','_')}.docx")
            
            else:
                status.update(label="Errore SerpApi", state="error")
                st.error("Nessun dato trovato da Google.")

        except Exception as e:
            status.update(label="Errore", state="error")
            st.error(f"Errore: {e}")
