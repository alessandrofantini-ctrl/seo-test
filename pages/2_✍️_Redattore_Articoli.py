import streamlit as st
from openai import OpenAI
from docx import Document
from io import BytesIO

st.set_page_config(page_title="Redattore AI Pro", layout="wide")

with st.sidebar:
    st.title("⚙️ Configurazione")
    openai_api_key = st.text_input("OpenAI Key", type="password")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    
    # Mappiamo le etichette su istruzioni concrete per l'AI
    lunghezza_label = st.select_slider(
        "Lunghezza Articolo", 
        options=["Breve", "Standard", "Long Form (Approfondito)"], 
        value="Standard"
    )
    
    creativita = st.slider("Livello Creatività", 0.0, 1.0, 0.6)

st.title("✍️ Redattore Articoli AI Pro")
st.markdown("Genera contenuti **E-E-A-T** ottimizzati, con link esterni autorevoli e struttura profonda.")

# Recupera brief dalla memoria
brief_default = ""
if 'ultimo_brief' in st.session_state:
    st.info("💡 Brief importato dall'Analisi SEO.")
    brief_default = st.session_state['ultimo_brief']

brief_input = st.text_area("Brief SEO / Scaletta", value=brief_default, height=400, placeholder="Incolla qui la struttura H1, H2, H3...")

if st.button("🚀 Scrivi Articolo Completo"):
    if not brief_input or not openai_api_key:
        st.error("⚠️ Manca il Brief o la API Key.")
    else:
        with st.spinner("✍️ Sto scrivendo un contenuto di alta qualità... (potrebbe richiedere circa 60-90 secondi)"):
            try:
                client = OpenAI(api_key=openai_api_key)
                
                # --- LOGICA DI LUNGHEZZA ---
                prompt_lunghezza = ""
                if lunghezza_label == "Breve":
                    prompt_lunghezza = "Scrivi circa 600-800 parole. Sii conciso ma esaustivo."
                elif lunghezza_label == "Standard":
                    prompt_lunghezza = "Scrivi circa 1200-1500 parole. Approfondisci bene ogni paragrafo."
                else: # Long Form
                    prompt_lunghezza = "OBIETTIVO: LONG FORM (2000+ PAROLE). È vietato riassumere. Devi espandere ogni singolo punto della scaletta con esempi, dettagli tecnici e spiegazioni approfondite."

                # --- PROMPT DI SISTEMA (Il "Cervello" dell'AI) ---
                system_prompt = """
                Sei un Senior SEO Copywriter con esperienza in contenuti E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness).
                
                LE TUE REGOLE D'ORO:
                1. **PROFONDITÀ**: Non scrivere mai frasi generiche ("è importante fare questo"). Spiega IL PERCHÉ e IL COME.
                2. **STRUTTURA**: Usa Markdown. Grassetto per le parole chiave (**così**). Usa elenchi puntati per spezzare il testo.
                3. **LINK ESTERNI (CRUCIALE)**: Devi inserire nel testo almeno 2 o 3 link esterni verso fonti AUTOREVOLI per aumentare il Trust.
                   - Usa fonti come: Wikipedia, Siti Governativi (.gov.it), Grandi Testate (Il Sole 24 Ore, Ansa), o Brand Ufficiali del settore (es. sito ufficiale Bosch se parli di ricambi).
                   - NON linkare siti competitor diretti (altre agenzie o blog piccoli).
                   - Format: [Testo Ancoraggio](URL).
                4. **TONO**: Professionale, empatico, autorevole ma leggibile.
                5. **NO FLUFF**: Evita introduzioni lunghe e inutili. Vai dritto al punto.
                """
                
                # --- PROMPT UTENTE (L'Istruzione Specifica) ---
                user_prompt = f"""
                Scrivi l'articolo completo seguendo rigorosamente questo BRIEF:
                
                {brief_input}
                
                ---
                ISTRUZIONI DI SCRITTURA:
                - **Lunghezza richiesta**: {prompt_lunghezza}
                - Inserisci link esterni autorevoli dove ha senso (es. normative, definizioni tecniche).
                - Usa paragrafi brevi (max 3-4 righe) per la leggibilità.
                - Concludi con una Call to Action (CTA) che inviti a contattare l'azienda o richiedere un preventivo.
                """

                # Chiamata a GPT-4o
                resp = client.chat.completions.create(
                    model="gpt-4o", 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=creativita,
                    max_tokens=4000 # Aumentiamo il limite di output per permettere testi lunghi
                )
                
                articolo_finale = resp.choices[0].message.content
                
                st.success("✅ Articolo Generato con Successo!")
                st.markdown("---")
                st.markdown(articolo_finale)
                
                # Creazione DOCX
                doc = Document()
                doc.add_heading('Articolo SEO Ottimizzato', 0)
                # Nota: Inseriamo il testo grezzo. Per formattazione perfetta in Word servirebbe un parser Markdown->Word complesso,
                # ma questo mantiene il contenuto leggibile.
                doc.add_paragraph(articolo_finale) 
                
                bio = BytesIO()
                doc.save(bio)
                
                st.download_button(
                    label="📥 Scarica Articolo (.docx)",
                    data=bio,
                    file_name="articolo_seo_pro.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
                
            except Exception as e:
                st.error(f"Si è verificato un errore: {e}")
