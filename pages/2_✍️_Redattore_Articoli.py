import streamlit as st
from openai import OpenAI
from docx import Document
from io import BytesIO

st.set_page_config(page_title="Redattore AI", layout="wide")

with st.sidebar:
    st.title("⚙️ Configurazione")
    openai_api_key = st.text_input("OpenAI Key", type="password")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    
    lunghezza = st.select_slider("Lunghezza", options=["Breve", "Standard", "Long Form"], value="Standard")
    creativita = st.slider("Creatività", 0.0, 1.0, 0.7)

st.title("✍️ Redattore Articoli AI")

# Recupera brief dalla pagina 1
brief_default = ""
if 'ultimo_brief' in st.session_state:
    st.info("💡 Brief importato dall'Analisi SEO.")
    brief_default = st.session_state['ultimo_brief']

brief_input = st.text_area("Brief SEO", value=brief_default, height=300)

if st.button("Scrivi Articolo"):
    if not brief_input or not openai_api_key:
        st.error("Manca Brief o API Key")
    else:
        with st.spinner("Scrittura in corso..."):
            try:
                client = OpenAI(api_key=openai_api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Sei un Senior Copywriter. Scrivi in Markdown."},
                        {"role": "user", "content": f"Scrivi articolo basato su: {brief_input}. Lunghezza: {lunghezza}."}
                    ],
                    temperature=creativita
                )
                articolo = resp.choices[0].message.content
                st.markdown(articolo)
                
                doc = Document()
                doc.add_paragraph(articolo)
                bio = BytesIO()
                doc.save(bio)
                st.download_button("📥 Scarica Articolo", bio, "articolo.docx")
            except Exception as e:
                st.error(f"Errore: {e}")
