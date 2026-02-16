import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from docx import Document
from io import BytesIO
import re
import json
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter, Retry

st.set_page_config(page_title="Redattore AI - Brand Voice", layout="wide")

# =========================
# HTTP SESSION + RETRY
# =========================
def build_session():
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

HTTP = build_session()
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# =========================
# SCRAPING TONE + STYLE DNA
# =========================
def safe_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s

def detect_main_container(soup: BeautifulSoup):
    for tag in ["article", "main"]:
        el = soup.find(tag)
        if el and len(el.get_text(" ", strip=True)) > 500:
            return el
    return soup.body if soup.body else soup

def scrape_tone_sample(url):
    """Scarica un campione di testo e segnali di stile dal sito per clonare il ToV."""
    try:
        resp = HTTP.get(url, headers=UA, timeout=15, allow_redirects=True)
        if resp.status_code >= 400 or not resp.text:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        main = detect_main_container(soup)

        paragraphs = main.find_all("p")
        text = " ".join([safe_text(p.get_text(" ", strip=True)) for p in paragraphs[:25]])
        text = safe_text(text)[:2500]

        # segnali base (euristici) per istruire meglio il modello
        avg_sentence_len = None
        sentences = re.split(r"[.!?]\s+", text)
        sentences = [s for s in sentences if len(s.split()) >= 4]
        if sentences:
            avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)

        # presenza elenchi
        bullets = main.find_all(["ul", "ol"])
        bullet_count = len(bullets)

        return {
            "text_sample": text,
            "avg_sentence_len": avg_sentence_len,
            "bullet_count": bullet_count,
        }
    except Exception:
        return None

# =========================
# DOCX (migliorato)
# =========================
def add_markdownish_to_docx(doc: Document, md: str):
    lines = md.splitlines()
    for line in lines:
        l = line.rstrip()
        if not l.strip():
            doc.add_paragraph("")
            continue

        if l.startswith("### "):
            doc.add_heading(l.replace("### ", ""), level=3)
        elif l.startswith("## "):
            doc.add_heading(l.replace("## ", ""), level=2)
        elif l.startswith("# "):
            doc.add_heading(l.replace("# ", ""), level=1)
        else:
            doc.add_paragraph(l)

def create_docx(md: str, title: str):
    doc = Document()
    doc.add_heading(title, 0)
    add_markdownish_to_docx(doc, md)
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

# =========================
# SIDEBAR (richiesta chiavi invariata)
# =========================
with st.sidebar:
    st.title("⚙️ Configurazione")

    openai_api_key = st.text_input("OpenAI Key", type="password")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]

    st.markdown("---")
    st.subheader("🎨 Brand Voice")

    default_url = st.session_state.get("client_url_session", "")
    client_url_input = st.text_input(
        "Sito Cliente per Tono di Voce",
        value=default_url,
        placeholder="https://www.lumicompany.it"
    )

    lunghezza_label = st.select_slider(
        "Lunghezza Articolo",
        options=["Breve", "Standard", "Long Form (Approfondito)"],
        value="Standard"
    )

    creativita = st.slider("Livello Creatività", 0.0, 1.0, 0.55)

    st.markdown("---")
    st.subheader("🔧 Qualità contenuto")
    seo_depth = st.select_slider(
        "Densità informativa",
        options=["Media", "Alta", "Molto alta (LLM-first)"],
        value="Alta"
    )
    add_tables = st.toggle("Aggiungi tabelle di confronto", value=True)
    add_examples = st.toggle("Aggiungi esempi pratici / mini-casi", value=True)
    add_faq = st.toggle("Includi FAQ (schema-ready)", value=True)
    add_summary = st.toggle("Aggiungi TL;DR iniziale", value=True)

st.title("✍️ Redattore Articoli: Ghostwriter Mode")
st.markdown(
    "Questo tool analizza il sito del cliente per **replicare lo stile** e produrre contenuti più completi, "
    "più utili per la SEO moderna e più leggibili anche da modelli LLM."
)

# Recupera brief dalla memoria
brief_default = ""
if "ultimo_brief" in st.session_state:
    st.info("💡 Brief importato dall'Analisi SEO.")
    brief_default = st.session_state["ultimo_brief"]

brief_input = st.text_area("Brief SEO / Scaletta", value=brief_default, height=350)

# =========================
# HELPERS: estrazione lingua target dal brief (best-effort)
# =========================
def infer_target_language_from_brief(brief: str):
    # prova a trovare "Lingua target: xx" o "(Lingua target: xx)" o "(Lingua: xx)"
    m = re.search(r"lingua\s*target\s*:\s*([a-z]{2})", brief, flags=re.I)
    if m:
        return m.group(1).lower()
    m = re.search(r"\(lingua\s*:\s*([a-z]{2})\)", brief, flags=re.I)
    if m:
        return m.group(1).lower()
    return None

def length_targets(label: str):
    # range parole + struttura consigliata
    if label == "Breve":
        return {"words": "800–1100", "h2": "5–7", "faq": "4–6"}
    if label == "Standard":
        return {"words": "1400–1900", "h2": "7–10", "faq": "6–10"}
    return {"words": "2200–3200", "h2": "10–14", "faq": "8–12"}

def depth_rules(depth: str):
    if depth == "Media":
        return "Approfondimento medio: copri i punti essenziali, ma senza eccesso di dettagli."
    if depth == "Alta":
        return "Approfondimento alto: spiega il perché, includi criteri decisionali, errori comuni, checklist, esempi."
    return (
        "Approfondimento molto alto (LLM-first): definizioni chiare, concetti/entità esplicite, "
        "passaggi step-by-step, confronti, limiti/casi particolari, FAQ complete. "
        "Scrivi come una guida di riferimento."
    )

# =========================
# GENERAZIONE ARTICOLO (2 PASSI)
# =========================
if st.button("🚀 Scrivi Articolo (Copia Stile Cliente)"):
    if not brief_input or not openai_api_key:
        st.error("⚠️ Manca il Brief o la API Key.")
    else:
        status = st.status("Avvio procedura Ghostwriter...", expanded=True)

        try:
            client = OpenAI(api_key=openai_api_key)

            # 1) BRAND VOICE DNA
            tone_instruction = ""
            style_dna = {}
            if client_url_input:
                status.write(f"🕵️ Analisi stile di scrittura su: {client_url_input}...")
                sample = scrape_tone_sample(client_url_input)

                if sample and sample.get("text_sample"):
                    style_dna = sample
                    avg_len = sample.get("avg_sentence_len")
                    bullet_count = sample.get("bullet_count", 0)

                    tone_instruction = f"""
### protocollo ghostwriter attivo
Devi imitare ESATTAMENTE il tone of voice del cliente.
Campione reale (estratto):
\"\"\"{sample["text_sample"]}\"\"\"

Indicatori stile (euristici):
- lunghezza media frase: {round(avg_len,1) if avg_len else "n/d"} parole
- presenza liste (ul/ol) nella pagina: {bullet_count}

Istruzioni di stile:
- replica ritmo, livello di formalità, uso di tecnicismi, energia e “calore”
- evita frasi generiche e costruzioni da AI; scrivi come un autore interno al brand
"""
                    status.write("✅ Stile clonato con successo.")
                else:
                    status.warning("⚠️ Impossibile leggere il sito cliente. Userò uno stile professionale coerente e naturale.")

            # 2) TARGETS
            targets = length_targets(lunghezza_label)
            depth = depth_rules(seo_depth)

            # best-effort: lingua target dal brief
            target_lang = infer_target_language_from_brief(brief_input) or "auto"

            # 3) STEP A: outline + piano contenuti (strutturato)
            status.write("🧱 Step 1/2: costruzione outline dettagliata...")
            system_a = f"""
Sei un Senior SEO Editor e ghostwriter del brand. Non sei un'AI esterna.
Obiettivo: produrre un articolo che sia ottimo per SEO moderna e comprensibile anche a modelli LLM.

{tone_instruction}

Vincoli qualità:
- niente riempitivi ("nel panorama odierno", "è importante sottolineare", ecc.)
- ogni sezione deve aggiungere informazione nuova (criteri, esempi, dati, checklist)
- struttura chiara, con definizioni esplicite e termini coerenti
"""

            user_a = f"""
Usa questo brief come fonte unica (non inventare servizi/claim non presenti nel brief):
{brief_input}

Output richiesto (in Markdown + un blocco JSON finale):
1) una outline completa con:
   - H1
   - H2/H3
   - per ogni H2: obiettivo sezione, punti chiave (bullet), esempi/asset da inserire, possibile snippet/box (se utile)
2) linee guida E-E-A-T (dove mettere prove, dati, citazioni, disclaimer)
3) elenco "entity/termini" che userai (in lingua target del brief se presente)
4) alla fine un blocco ```json con:
{{
  "target_language": "{target_lang}",
  "word_target": "{targets["words"]}",
  "sections": [{{"h2":"...","h3":["..."],"key_points":["..."],"examples":["..."]}}],
  "eeat_notes": ["..."],
  "entities": ["..."]
}}

Regole:
- mantieni H1/H2/H3 nella lingua target se il brief lo richiede
- istruzioni e note possono essere in italiano
"""

            outline_resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_a}, {"role": "user", "content": user_a}],
                temperature=max(0.2, min(creativita, 0.6)),
                max_tokens=2500
            )
            outline_md = outline_resp.choices[0].message.content

            st.markdown("### 🧱 Outline (step 1)")
            st.markdown(outline_md)

            # 4) STEP B: stesura articolo (con vincoli SEO/LLM-first)
            status.write("✍️ Step 2/2: stesura articolo completo...")

            extra_blocks = []
            if add_summary:
                extra_blocks.append("- apri con un TL;DR di 5–7 bullet che anticipa i punti chiave (non generici)")
            if add_tables:
                extra_blocks.append("- inserisci almeno 1 tabella utile (confronto, checklist, pro/contro, criteri)")
            if add_examples:
                extra_blocks.append("- inserisci esempi pratici o mini-casi (anche anonimi) quando chiariscono decisioni")
            if add_faq:
                extra_blocks.append(f"- includi una sezione FAQ ({targets['faq']}) con domande in lingua target e risposte concise")

            extra_blocks_txt = "\n".join(extra_blocks) if extra_blocks else "- nessun blocco extra"

            system_b = f"""
Sei il Senior Copywriter del brand. Non sei un'AI esterna.

{tone_instruction}

Regole SEO & LLM-first:
- usa Markdown con gerarchia corretta (#, ##, ###)
- integra definizioni e concetti in modo esplicito (aiuta retrieval e comprensione)
- scrivi denso: ogni paragrafo deve portare valore (criteri, passaggi, esempi, numeri quando sensati)
- evita keyword stuffing: usa varianti naturali e sinonimi
- inserisci 2-3 link esterni a fonti autorevoli (non competitor) come riferimenti (testo ancora, non URL nudi)
- niente conclusioni "acqua": chiudi con una sintesi operativa e next step

Profilo approfondimento:
{depth}
"""

            user_b = f"""
Stendi l'articolo completo usando questa outline (non devi mostrare il JSON):
{outline_md}

Vincoli:
- lunghezza target: {targets["words"]} parole (non scrivere corto)
- numero H2 indicativo: {targets["h2"]}
- {extra_blocks_txt}

Nota:
- se nel brief H1/H2/H3 devono essere in una lingua specifica, rispettala.
- le CTA possono essere nella lingua più adatta al mercato indicato nel brief.
"""

            # max_tokens: più alto per evitare tagli (se il modello taglia, puoi aumentare ulteriormente)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_b}, {"role": "user", "content": user_b}],
                temperature=creativita,
                max_tokens=6500
            )

            articolo_finale = resp.choices[0].message.content

            status.update(label="Articolo completato!", state="complete", expanded=False)

            st.markdown("### 📄 Anteprima articolo (stile brand)")
            st.markdown(articolo_finale)

            # DOCX export (migliorato)
            docx = create_docx(articolo_finale, "Articolo SEO - Brand Voice")
            st.download_button(
                label="📥 Scarica Articolo (.docx)",
                data=docx,
                file_name="articolo_brand_voice.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

        except Exception as e:
            status.update(label="Errore", state="error")
            st.error(f"Errore: {e}")
