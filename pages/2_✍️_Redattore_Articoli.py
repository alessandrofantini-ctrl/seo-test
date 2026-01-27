import streamlit as st
from openai import OpenAI
from docx import Document
from io import BytesIO

st.set_page_config(page_title="Redattore AI", layout="wide")

# --- SIDEBAR (Configurazione) ---
with st.sidebar:
    st.title("⚙️ Configurazione")
    # Recupera le chiavi (usa st.secrets o input manuale come nell'altra pagina)
    openai_api_key = st.text_input("OpenAI Key", type="password")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]

    lunghezza = st.select_slider("Lunghezza Articolo", options=["Breve (500 parole)", "Standard (1000 parole)", "Long Form (2000 parole)"], value="Standard (1000 parole)")
    creativita = st.slider("Livello Creatività", 0.0, 1.0, 0.7)

# --- MAIN ---
st.title("✍️ Redattore Articoli AI")
st.markdown("Trasforma il brief SEO in un articolo completo e formattato.")

# Recupera il brief dalla memoria (se proveniamo dall'altra pagina)
brief_default = ""
if 'ultimo_brief' in st.session_state:
    st.info("💡 Ho trovato un brief generato dall'Analisi SEO. L'ho caricato automaticamente qui sotto.")
    brief_default = st.session_state['ultimo_brief']

# Input Area
brief_input = st.text_area("Incolla qui il Brief SEO o la Scaletta", value=brief_default, height=300, placeholder="Incolla qui la struttura H1, H2, H3...")

if st.button("Scrivi Articolo Completo"):
    if not brief_input or not openai_api_key:
        st.error("Manca il brief o la API Key.")
    else:
        status = st.status("Writing in progress...", expanded=True)
        
        try:
            client = OpenAI(api_key=openai_api_key)
            
            # Prompt specializzato nella scrittura estesa
            system_prompt = "Sei un Senior Copywriter italiano. Scrivi articoli coinvolgenti, ottimizzati SEO e pronti per la pubblicazione. Usa formattazione Markdown (grassetti, elenchi)."
            
            user_prompt = f"""
            Scrivi un articolo completo basato su questo BRIEF SEO:
            
            {brief_input}
            
            IMPOSTAZIONI:
            - Lunghezza target: {lunghezza}
            - Stile: Fluido, professionale ma non noioso.
            - Formattazione: Usa H1, H2, H3 correttamente. Usa grassetti per i concetti chiave.
            
            Non ripetere il brief, scrivi direttamente il contenuto finale.
            """
            
            status.write("🧠 Elaborazione del testo...")
            
            response = client.chat.completions.create(
                model="gpt-4o", # GPT-4o è essenziale per articoli lunghi di qualità
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=creativita
            )
            
            articolo_finale = response.choices[0].message.content
            status.update(label="Articolo Pronto!", state="complete", expanded=False)
            
            # Output
            st.markdown("### 📄 Articolo Generato")
            st.markdown(articolo_finale)
            
            # Export Word
            doc = Document()
            doc.add_heading('Articolo SEO', 0)
            doc.add_paragraph(articolo_finale) # Nota: python-docx incolla il markdown come testo grezzo. Per formattazione perfetta serve logica complessa, ma questo basta per il copia-incolla.
            bio = BytesIO()
            doc.save(bio)
            
            st.download_button(
                label="📥 Scarica Articolo (.docx)",
                data=bio,
                file_name="articolo_seo.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            
        except Exception as e:
            st.error(f"Errore: {e}")
