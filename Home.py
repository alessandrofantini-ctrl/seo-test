import streamlit as st

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="Lumi Company SEO Suite",
    page_icon="✨",
    layout="centered"
)

# --- HEADER CON LOGO E TITOLO ---
try:
    st.image("https://www.lumicompany.it/_next/static/media/lumi_logo_black.b03993c0.svg", width=300)
except Exception:
    st.header("Lumi Company")

st.title("✨ Lumi Company AI Suite")
st.markdown("### Gli strumenti ufficiali per la strategia SEO di [www.lumicompany.it](https://www.lumicompany.it)")
st.markdown("---")

# --- MENU DI NAVIGAZIONE VISUALE ---
st.markdown("""
Benvenuto nella dashboard operativa. Seleziona uno strumento dal menu laterale o usa i link rapidi qui sotto:

#### 0️⃣ 👤 **[Gestione Clienti](/Clienti)**
_Il punto di partenza._ Crea e gestisci i **profili dei tuoi clienti**: prodotti, servizi, USP e tono di voce.
Ogni profilo viene caricato automaticamente negli strumenti, senza dover reinserire i dati ogni volta.

#### 1️⃣ 🔎 **[Analisi & Strategia SEO](/Analisi_SEO)**
_Il primo passo._ Analizza le keyword, studia i competitor e crea il **Brief editoriale** per posizionarti su Google.

#### 2️⃣ ✍️ **[Redattore articoli AI](/Redattore_Articoli)**
_Il secondo passo._ Trasforma il brief in un **articolo completo**, formattato e ottimizzato.

#### 3️⃣ 🔗 **[Internal link assistant](/Internal_link_assistant)**
_Il terzo passo._ Inserisce **link interni in automatico** nel testo usando:
- export **Google Search Console** (priorità pagine)
- (opzionale) **crawl** tipo Screaming Frog (title/H1/meta)
- **AI** per trovare il match migliore tra paragrafi e pagine.

#### 4️⃣ 🔄 **[Redirect assistant pro](/Redirect_Generator)**
_Per le migrazioni._ Mappa redirect 301, gestisce le lingue e crea file Excel pronti per i dev.

---
""")

# --- FOOTER ---
st.info("💡 Workflow consigliato: Gestione Clienti → Analisi SEO → Redazione articolo → Internal linking. Usa il Redirect assistant solo in fase di restyling o migrazione.")
st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: small;'>
        Developed for <b>Lumi Company</b> | Powered by OpenAI & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
