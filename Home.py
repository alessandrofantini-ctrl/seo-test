import streamlit as st

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="Lumi Company SEO Suite",
    page_icon="✨",
    layout="centered"
)

# --- HEADER CON LOGO E TITOLO ---
st.image("https://www.lumicompany.it/_next/static/media/lumi_logo_black.b03993c0.svg", width=300)

st.title("✨ Lumi Company AI Suite")
st.markdown("### Gli strumenti ufficiali per la strategia SEO di [www.lumicompany.it](https://www.lumicompany.it)")

st.markdown("---")

# --- MENU DI NAVIGAZIONE VISUALE ---
st.markdown("""
Benvenuto nella dashboard operativa. Seleziona uno strumento dal menu laterale o usa i link rapidi qui sotto:

#### 1️⃣ 🔎 **[Analisi & Strategia SEO](/Analisi_SEO)**
_Il primo passo._ Analizza le keyword, studia i competitor e crea il **Brief Editoriale** perfetto per posizionarti su Google.

#### 2️⃣ ✍️ **[Redattore Articoli AI](/Redattore_Articoli)**
_Il secondo passo._ Trasforma il Brief generato in un **articolo completo**, formattato e ottimizzato per il blog di Lumi Company.

#### 3️⃣ 🔄 **[Redirect Assistant Pro](/Redirect_Generator)**
_Per le migrazioni._ Carica i CSV di Screaming Frog e usa l'AI per mappare automaticamente le vecchie URL sulle nuove, evitando errori 404.

---
""")

# --- FOOTER ---
st.info("💡 **Workflow Consigliato:** Analisi SEO -> Redazione Articolo. Usa il Redirect Assistant solo in fase di restyling o migrazione.")

st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: small;'>
        Developed for <b>Lumi Company</b> | Powered by OpenAI & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
