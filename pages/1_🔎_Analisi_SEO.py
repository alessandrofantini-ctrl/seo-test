import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from docx import Document
from io import BytesIO
import json
import re
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter, Retry

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="SEO Content Strategist Pro", layout="wide")

# --- DIZIONARIO MERCATI ---
MARKETS = {
    "🇮🇹 Italia": {"gl": "it", "hl": "it", "domain": "google.it"},
    "🇺🇸 USA (English)": {"gl": "us", "hl": "en", "domain": "google.com"},
    "🇬🇧 UK": {"gl": "uk", "hl": "en", "domain": "google.co.uk"},
    "🇪🇸 Spagna": {"gl": "es", "hl": "es", "domain": "google.es"},
    "🇫🇷 Francia": {"gl": "fr", "hl": "fr", "domain": "google.fr"},
    "🇩🇪 Germania": {"gl": "de", "hl": "de", "domain": "google.de"},
}

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

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ SEO Settings")

    # >>>>>>>>>>>> QUESTO BLOCCO RESTA INVARIATO (richiesta chiavi) <<<<<<<<<<<<
    openai_api_key = st.text_input("OpenAI Key", type="password")
    serp_api_key = st.text_input("SerpApi Key", type="password")

    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    if not serp_api_key and "SERP_API_KEY" in st.secrets:
        serp_api_key = st.secrets["SERP_API_KEY"]
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    st.markdown("---")
    st.subheader("🌍 Impostazioni Mercato")
    selected_market_label = st.selectbox("Seleziona Mercato Target", list(MARKETS.keys()))
    market_params = MARKETS[selected_market_label]

    st.markdown("---")
    st.subheader("🎯 Target Cliente")
    client_url = st.text_input("URL Sito Cliente (Opzionale)", placeholder="https://www.tuosito.it")
    custom_usp = st.text_area("USP / Punti di Forza", placeholder="Es. Spedizione in 24h...", height=100)
    tone_of_voice = st.selectbox("Tono di Voce", ["Autorevole & Tecnico", "Empatico & Problem Solving", "Diretto & Commerciale"])

    st.markdown("---")
    st.subheader("🔧 Modalità Analisi")
    deep_mode = st.toggle("Deep mode (più competitor + più segnali)", value=True)
    max_competitors = st.slider("Competitor da analizzare", min_value=2, max_value=10, value=6 if deep_mode else 3)
    max_workers = st.slider("Parallelismo scraping", min_value=2, max_value=10, value=6)
    include_schema = st.toggle("Estrai segnali schema (JSON-LD)", value=True)
    include_meta = st.toggle("Estrai meta (title/description/canonical)", value=True)

# --- MAIN PAGE ---
st.title("🚀 SEO Brief Generator Multi-Country")
st.markdown(f"Analisi impostata su: **{selected_market_label}**")

col1, col2 = st.columns([3, 1])
with col1:
    keyword = st.text_input("Keyword Principale", placeholder="Es. best automated gearbox repair")
with col2:
    target_intent = st.selectbox("Intento", ["Informativo", "Commerciale", "Navigazionale"])

# =========================
# UTILS
# =========================
def safe_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""

def extract_json_ld(soup: BeautifulSoup):
    scripts = soup.find_all("script", type="application/ld+json")
    out = []
    for sc in scripts[:10]:
        txt = sc.get_text(strip=True)
        if not txt:
            continue
        try:
            data = json.loads(txt)
            out.append(data)
        except Exception:
            # a volte è JSON-LD non valido (multipli oggetti, ecc.)
            pass
    return out

def detect_main_container(soup: BeautifulSoup):
    # euristica semplice: prova <article>, poi <main>, poi body
    for tag in ["article", "main"]:
        el = soup.find(tag)
        if el and len(el.get_text(" ", strip=True)) > 500:
            return el
    return soup.body if soup.body else soup

# =========================
# CACHING
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_serp_data(query, api_key, gl, hl, domain):
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "hl": hl,
        "gl": gl,
        "google_domain": domain
    }
    r = HTTP.get("https://serpapi.com/search", params=params, timeout=25)
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=60 * 60)
def scrape_site_content(url, include_meta=True, include_schema=True):
    data = {
        "url": url,
        "domain": domain_of(url),
        "title": "",
        "meta_description": "",
        "canonical": "",
        "headers": [],
        "h1": "",
        "h2": [],
        "h3": [],
        "word_count": 0,
        "text_sample": "",
        "lang": "",
        "schema_types": [],
        "has_faq_schema": False,
    }

    try:
        resp = HTTP.get(url, headers=UA, timeout=15, allow_redirects=True)
        if resp.status_code >= 400 or not resp.text:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # lang
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            data["lang"] = safe_text(html_tag.get("lang"))

        # meta
        if include_meta:
            if soup.title and soup.title.string:
                data["title"] = safe_text(soup.title.string)
            md = soup.find("meta", attrs={"name": "description"})
            if md and md.get("content"):
                data["meta_description"] = safe_text(md.get("content"))

            canon = soup.find("link", rel=lambda x: x and "canonical" in x.lower())
            if canon and canon.get("href"):
                data["canonical"] = safe_text(canon.get("href"))

        # contenuto principale (euristica)
        main = detect_main_container(soup)

        # headers
        elements = main.find_all(["h1", "h2", "h3"])
        for tag in elements[:30]:
            txt = safe_text(tag.get_text(" ", strip=True))
            if not txt:
                continue
            data["headers"].append(f"[{tag.name.upper()}] {txt}")
            if tag.name == "h1" and not data["h1"]:
                data["h1"] = txt
            elif tag.name == "h2" and len(data["h2"]) < 15:
                data["h2"].append(txt)
            elif tag.name == "h3" and len(data["h3"]) < 20:
                data["h3"].append(txt)

        # testo
        paragraphs = main.find_all("p")
        text_content = " ".join([safe_text(p.get_text(" ", strip=True)) for p in paragraphs])
        text_content = safe_text(text_content)
        data["word_count"] = len(text_content.split()) if text_content else 0
        data["text_sample"] = text_content[:1800]

        # schema
        if include_schema:
            jlds = extract_json_ld(soup)
            types = set()
            has_faq = False
            for item in jlds:
                # può essere dict o list
                items = item if isinstance(item, list) else [item]
                for it in items:
                    if isinstance(it, dict) and "@type" in it:
                        t = it["@type"]
                        if isinstance(t, list):
                            for tt in t:
                                types.add(str(tt))
                        else:
                            types.add(str(t))
                        if "FAQPage" in str(t):
                            has_faq = True
            data["schema_types"] = sorted(types)[:20]
            data["has_faq_schema"] = has_faq

        return data
    except Exception:
        return None

def summarize_competitor_for_prompt(comp):
    # summary compatto, più segnale e meno token
    parts = []
    parts.append(f"URL: {comp['url']} (dominio: {comp.get('domain','')})")
    if comp.get("title"):
        parts.append(f"Title: {comp['title']}")
    if comp.get("meta_description"):
        parts.append(f"Meta description: {comp['meta_description']}")
    if comp.get("h1"):
        parts.append(f"H1: {comp['h1']}")
    if comp.get("h2"):
        parts.append("H2: " + " | ".join(comp["h2"][:10]))
    if comp.get("word_count"):
        parts.append(f"Word count stimata: {comp['word_count']}")
    if comp.get("lang"):
        parts.append(f"Lang: {comp['lang']}")
    if comp.get("schema_types"):
        parts.append("Schema types: " + ", ".join(comp["schema_types"][:10]))
    if comp.get("has_faq_schema"):
        parts.append("FAQ schema: sì")
    # micro-estratto testo per contesto (limitato)
    if comp.get("text_sample"):
        parts.append(f"Estratto: {comp['text_sample']}")
    return "\n".join(parts)

def build_serp_snapshot(serp_json, max_items):
    snapshot = {
        "organic": [],
        "paa": [],
        "related_searches": [],
        "features": [],
    }

    if not serp_json:
        return snapshot

    # organic
    for res in serp_json.get("organic_results", [])[:max_items]:
        snapshot["organic"].append({
            "position": res.get("position"),
            "title": res.get("title"),
            "link": res.get("link"),
            "snippet": res.get("snippet") or res.get("snippet_highlighted_words"),
            "source": res.get("source"),
        })

    # people also ask / related questions
    for q in serp_json.get("related_questions", [])[:20]:
        if q.get("question"):
            snapshot["paa"].append(q.get("question"))

    # related searches
    for r in serp_json.get("related_searches", [])[:20]:
        if r.get("query"):
            snapshot["related_searches"].append(r.get("query"))

    # features euristiche (non sempre presenti in serpapi)
    if serp_json.get("answer_box"):
        snapshot["features"].append("answer_box")
    if serp_json.get("knowledge_graph"):
        snapshot["features"].append("knowledge_graph")
    if serp_json.get("shopping_results"):
        snapshot["features"].append("shopping_results")
    if serp_json.get("local_results"):
        snapshot["features"].append("local_results")
    if serp_json.get("top_stories"):
        snapshot["features"].append("top_stories")
    if serp_json.get("inline_videos"):
        snapshot["features"].append("inline_videos")

    return snapshot

def create_docx_from_markdownish(content_md: str, kw: str):
    # docx più leggibile: heading + paragrafi, senza parser full markdown (semplice e robusto)
    doc = Document()
    doc.add_heading(f"SEO Brief: {kw}", 0)

    lines = content_md.splitlines()
    for line in lines:
        l = line.strip()
        if not l:
            doc.add_paragraph("")
            continue

        # headings markdown
        if l.startswith("### "):
            doc.add_heading(l.replace("### ", ""), level=3)
        elif l.startswith("## "):
            doc.add_heading(l.replace("## ", ""), level=2)
        elif l.startswith("# "):
            doc.add_heading(l.replace("# ", ""), level=1)
        else:
            doc.add_paragraph(line)

    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

# =========================
# UI ACTION
# =========================
if st.button("Avvia Analisi Completa"):
    if not keyword or not openai_api_key or not serp_api_key:
        st.error("Inserisci Keyword e API Keys.")
    else:
        status = st.status(f"Avvio scansione su {selected_market_label}...", expanded=True)
        try:
            # 1) ANALISI CLIENTE
            client_context_str = "Nessun sito cliente fornito. (Generico)"
            client_data = None
            if client_url:
                status.write("🏢 Analisi identità cliente...")
                client_data = scrape_site_content(
                    client_url,
                    include_meta=include_meta,
                    include_schema=include_schema
                )
                if client_data:
                    client_context_str = (
                        f"SITO CLIENTE: {client_url}\n"
                        f"TITLE: {client_data.get('title','')}\n"
                        f"META DESC: {client_data.get('meta_description','')}\n"
                        f"H1: {client_data.get('h1','')}\n"
                        f"H2: {', '.join(client_data.get('h2',[])[:10])}\n"
                        f"TESTO (estratto): {client_data.get('text_sample','')}"
                    )
                else:
                    client_context_str = f"SITO CLIENTE: {client_url}\n(Non leggibile / errore scraping)"

            if custom_usp:
                client_context_str += f"\nUSP MANUALI: {custom_usp}"

            # 2) SERP
            status.write(f"🔍 Analisi SERP ({market_params['domain']})...")
            serp = get_serp_data(
                keyword,
                serp_api_key,
                gl=market_params["gl"],
                hl=market_params["hl"],
                domain=market_params["domain"]
            )

            if not serp or "organic_results" not in serp:
                status.update(label="Errore SerpApi", state="error")
                st.error("Nessun dato trovato da Google. Verifica la SerpApi Key.")
                st.stop()

            serp_snapshot = build_serp_snapshot(serp, max_competitors)
            organic_urls = [x["link"] for x in serp_snapshot["organic"] if x.get("link")][:max_competitors]

            with st.expander("🔎 SERP snapshot (debug utile)", expanded=False):
                st.json(serp_snapshot)

            # 3) SCRAPING COMPETITOR (PARALLELO)
            status.write("⚔️ Spionaggio Competitor (scraping parallelo)...")
            competitor_results = []
            progress = status.progress(0)

            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for u in organic_urls:
                    futures.append(ex.submit(scrape_site_content, u, include_meta, include_schema))

                done = 0
                for fut in as_completed(futures):
                    done += 1
                    progress.progress(done / max(1, len(futures)))
                    comp = fut.result()
                    if comp:
                        competitor_results.append(comp)

            progress.empty()

            if not competitor_results:
                status.update(label="Errore scraping competitor", state="error")
                st.error("Non sono riuscito a leggere i competitor (403/timeout). Prova a ridurre parallelismo o competitor.")
                st.stop()

            # 4) COMPRESS / SUMMARIZE PER PROMPT
            status.write("🧩 Sintesi segnali competitor...")
            competitor_summaries = []
            for c in competitor_results[:max_competitors]:
                competitor_summaries.append(summarize_competitor_for_prompt(c))

            competitor_block = "\n\n---\n\n".join(competitor_summaries)
            competitor_block = competitor_block[:12000]  # safety cap

            paa = serp_snapshot.get("paa", [])
            related_searches = serp_snapshot.get("related_searches", [])
            serp_features = serp_snapshot.get("features", [])

            # 5) AI STRATEGY
            status.write("🧠 Elaborazione mappa semantica & brief...")

            system_prompt = (
                "Sei un Senior SEO Strategist internazionale. "
                "Crei brief operativi per contenuti 'skyscraper' superiori ai competitor, "
                "con struttura chiara e indicazioni copy azionabili."
            )

            # Output: Markdown + un blocco JSON riutilizzabile (senza obbligare parsing complesso)
            user_prompt = f"""
OBIETTIVO: Creare il Brief SEO definitivo per la keyword: "{keyword}".
MERCATO TARGET: {selected_market_label} (Lingua target: {market_params['hl']}).
INTENTO: {target_intent}.
TONO: {tone_of_voice}.

REGOLE IMPORTANTI:
- Analizza i dati forniti (competitor e SERP) che sono nella lingua del mercato target.
- Restituisci il brief in ITALIANO (per il consulente), ma suggerisci H1/H2, meta title/description e keyword nella LINGUA TARGET ({market_params['hl']}).
- Sii specifico, niente frasi generiche. Evidenzia gap e opportunità reali.

### DATI DI ANALISI

1) BRAND (contesto):
{client_context_str}

2) SERP SNAPSHOT:
- Features SERP rilevate: {", ".join(serp_features) if serp_features else "Nessuna feature specifica rilevata."}
- PAA (People also ask): {", ".join(paa) if paa else "Nessuna domanda specifica rilevata."}
- Related searches: {", ".join(related_searches) if related_searches else "Nessuna correlata rilevata."}

3) COMPETITOR (sintesi per URL):
{competitor_block}

### TASK: GENERA IL BRIEF

Restituisci un output strutturato in Markdown con queste sezioni:

## sezione a: analisi semantica
- primary keyword
- keywords secondarie (in {market_params['hl']}) 10-15
- entità/concetti obbligatori 10 (in {market_params['hl']})
- termini/angoli da evitare 5 (in {market_params['hl']} o IT se concetto)
- gap analysis: cosa manca ai competitor in questo mercato specifico (bullet concreti)
- intent check: l’intento scelto è coerente con la SERP? se no, proponi correzione

## sezione b: struttura del contenuto
- meta title (in {market_params['hl']}, <= 60 caratteri)
- meta description (in {market_params['hl']}, <= 155 caratteri)
- h1 (in {market_params['hl']})
- outline h2/h3 (in {market_params['hl']}) integrando PAA e correlate
- per ogni H2: istruzioni copy in italiano (cosa scrivere, esempi di contenuti, prove/claim da includere)

## sezione c: eeat e conversione
- prove di autorevolezza da inserire (es. certificazioni, casi, dati, testimonianze) e dove metterle
- faq consigliate 5-8 (in {market_params['hl']}) con risposta breve (in {market_params['hl']})
- CTA e microconversioni (in italiano) adatte all’intento

## sezione d: blocco json riutilizzabile
Alla fine, aggiungi un blocco JSON in un code fence ```json con:
{{
  "market": "...",
  "language": "...",
  "primary_keyword": "...",
  "secondary_keywords": [...],
  "entities": [...],
  "meta_title": "...",
  "meta_description": "...",
  "h1": "...",
  "outline": [{{"h2": "...", "h3": ["..."]}}],
  "faqs": [{{"q":"...","a":"..."}}],
  "notes_it": "..."
}}

Non aggiungere testo dopo il blocco JSON.
"""

            client = OpenAI(api_key=openai_api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6
            )
            output = resp.choices[0].message.content

            status.update(label="Strategia pronta!", state="complete", expanded=False)

            # Render
            st.markdown(output)

            # Store
            st.session_state["ultimo_brief"] = output
            st.session_state["client_url_session"] = client_url
            st.session_state["serp_snapshot"] = serp_snapshot
            st.session_state["competitors"] = competitor_results

            # Download docx
            docx = create_docx_from_markdownish(output, keyword)
            st.download_button(
                "📥 Scarica Brief .docx",
                docx,
                f"brief_{keyword.replace(' ','_')}.docx"
            )

            # Download JSON estratto (best-effort)
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", output, flags=re.S)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    st.download_button(
                        "📥 Scarica Brief JSON",
                        data=json.dumps(parsed, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name=f"brief_{keyword.replace(' ','_')}.json",
                        mime="application/json"
                    )
                except Exception:
                    pass

        except Exception as e:
            status.update(label="Errore", state="error")
            st.error(f"Errore: {e}")
