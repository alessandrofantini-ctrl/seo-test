import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from docx import Document
from io import BytesIO

st.set_page_config(page_title="Redattore AI - Brand Voice", layout="wide")

# --- FUNZIONE DI SCRAPING (Necessaria anche qui per analizzare il tono) ---
def scrape_tone_sample(url):
    """Scarica un campione di testo dal sito per analizzare il Tono di Voce."""
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    try:
        resp = requests.get(url, headers=ua, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Prende i paragrafi per capire come scrivono
        paragraphs = soup.find_all('p')
        text_sample = " ".join([p.get_text(strip=True) for p in paragraphs[:15]])
        return text_sample[:2000] # Limitiamo a 2000 caratteri per il prompt
    except Exception:
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Configurazione")
    openai_api_key = st.text_input("OpenAI Key", type="password")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    
    st.markdown("---")
    st.subheader("🎨 Brand Voice")
    
    # Recupera URL dalla sessione (se impostato nella Pagina 1)
    default_url = st.session_state.get('client_url_session', '')
    client_url_input = st.text_input("Sito Cliente per Tono di Voce", value=default_url, placeholder="https://www.lumicompany.it")
    
    lunghezza_label = st.select_slider(
        "Lunghezza Articolo", 
        options=["Breve", "Standard", "Long Form (Approfondito)"], 
        value="Standard"
    )
    
    creativita = st.slider("Livello Creatività", 0.0, 1.0, 0.6)

st.title("✍️ Redattore Articoli: Ghostwriter Mode")
st.markdown("Questo tool analizza il sito del cliente per **clonare il suo stile di scrittura** e produrre contenuti indistinguibili dall'originale.")

# Recupera brief dalla memoria
brief_default = ""
if 'ultimo_brief' in st.session_state:
    st.info("💡 Brief importato dall'Analisi SEO.")
    brief_default = st.session_state['ultimo_brief']

brief_input = st.text_area("Brief SEO / Scaletta", value=brief_default, height=350)

if st.button("🚀 Scrivi Articolo (Copia Stile Cliente)"):
    if not brief_input or not openai_api_key:
        st.error("⚠️ Manca il Brief o la API Key.")
    else:
        status = st.status("Avvio procedura Ghostwriter...", expanded=True)
        
        try:
            client = OpenAI(api_key=openai_api_key)
            
            # 1. ANALISI TONO DI VOCE (GHOSTWRITER)
            tone_instruction = ""
            client_sample = ""
            
            if client_url_input:
                status.write(f"🕵️ Analisi stile di scrittura su: {client_url_input}...")
                client_sample = scrape_tone_sample(client_url_input)
                
                if client_sample:
                    tone_instruction = f"""
                    ### PROTOCOLLO GHOSTWRITER ATTIVO
                    Devi imitare ESATTAMENTE il "Tone of Voice" del cliente.
                    Ecco un campione di come scrivono nel loro sito:
                    
                    "{client_sample}..."
                    
                    ANALISI STILE DA REPLICARE:
                    - Analizza il lessico (tecnico vs semplice).
                    - Analizza la lunghezza delle frasi (brevi e punchy vs lunghe e descrittive).
                    - Analizza il calore (distaccato/istituzionale vs amichevole/empatico).
                    - SCRIVI L'ARTICOLO COME SE FOSSI LORO.
                    """
                    status.write("✅ Stile clonato con successo.")
                else:
                    status.warning("⚠️ Impossibile leggere il sito cliente. Userò uno stile standard professionale.")
            
            # 2. DEFINIZIONE LUNGHEZZA
            prompt_lunghezza = ""
            if lunghezza_label == "Breve":
                prompt_lunghezza = "Circa 600-800 parole. Conciso."
            elif lunghezza_label == "Standard":
                prompt_lunghezza = "Circa 1200 parole. Ben argomentato."
            else:
                prompt_lunghezza = "Long Form (2000+ parole). Esaustivo, vietato riassumere."

            # 3. PROMPT GENERAZIONE
            status.write("✍️ Scrittura articolo in corso...")
            
            system_prompt = f"""
            Sei il Senior Copywriter del brand. Non sei un'AI esterna.
            
            {tone_instruction}
            
            REGOLE SEO & E-E-A-T:
            1. Usa Markdown (H1, H2, H3, **grassetti**).
            2. Inserisci 2-3 link esterni a fonti autorevoli (non competitor).
            3. Evita frasi fatte ("Nel panorama odierno...", "È importante sottolineare..."). Vai dritto al punto come farebbe un esperto.
            """
            
            user_prompt = f"""
            Scrivi l'articolo completo basandoti su questo BRIEF:
            
            {brief_input}
            
            Target Lunghezza: {prompt_lunghezza}
            """

            resp = client.chat.completions.create(
                model="gpt-4o", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=creativita,
                max_tokens=4000
            )
            
            articolo_finale = resp.choices[0].message.content
            
            status.update(label="Articolo Completato!", state="complete", expanded=False)
            
            st.markdown("### 📄 Anteprima Articolo (Stile Brand)")
            st.markdown(articolo_finale)
            
            # DOCX Export
            doc = Document()
            doc.add_heading('Articolo SEO - Brand Voice', 0)
            doc.add_paragraph(articolo_finale)
            bio = BytesIO()
            doc.save(bio)
            
            st.download_button(
                label="📥 Scarica Articolo (.docx)",
                data=bio,
                file_name="articolo_brand_voice.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
                
        except Exception as e:
            status.update(label="Errore", state="error")
            st.error(f"Errore: {e}")
