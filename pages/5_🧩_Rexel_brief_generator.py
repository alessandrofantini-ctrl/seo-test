import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter, Retry
import re
import json
import numpy as np
from io import BytesIO

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="B2B category brief (batch + Rexel facets)", layout="wide")
st.title("🧩 B2B category brief (batch + Rexel facets)")
st.caption("Output ridotto: H1, lunghezza consigliata, outline H2/H3, FAQ (solo domande). Supporta batch da Excel (URL + query).")

# =========================
# HTTP session with retry
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
# UTILS
# =========================
def safe_text(s) -> str:
    if s is None:
        return ""
    try:
        if pd.isna(s):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(s)).strip()

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "").lower()
    except Exception:
        return ""

def uniq_keep_order(items):
    out = []
    for x in items:
        x = safe_text(x)
        if x and x not in out:
            out.append(x)
    return out

def clamp(s: str, n: int) -> str:
    s = safe_text(s)
    return s[:n] if len(s) > n else s

def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))

def normalize_lines(text_area_value: str):
    return [safe_text(x) for x in (text_area_value or "").splitlines() if safe_text(x)]

def to_excel_bytes(df: pd.DataFrame, sheet_name="output") -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name=sheet_name)
    return bio.getvalue()

# =========================
# REXEL FACETS (brands + filter names) from category URL
# =========================
def scrape_rexel_facets(category_url: str):
    """
    Estrae:
    - brands dal div id="collapseBrands" (se presente)
    - nomi filtri (euristica su righe con conteggio) nella finestra dopo 'Filtri'
    Ritorna anche una finestra testo per debug.
    """
    out = {
        "url": category_url,
        "http_status": None,
        "brands": [],
        "filters": [],
        "raw_lines_window": [],
        "error": "",
    }
    try:
        r = HTTP.get(category_url, headers=UA, timeout=18, allow_redirects=True)
        out["http_status"] = r.status_code
        if r.status_code >= 400 or not r.text:
            out["error"] = f"HTTP {r.status_code}"
            return out

        soup = BeautifulSoup(r.text, "html.parser")

        # BRANDS
        brands = []
        brands_div = soup.find(id="collapseBrands")
        if brands_div:
            for el in brands_div.find_all(["a", "label", "span"], limit=300):
                t = safe_text(el.get_text(" ", strip=True))
                if not t:
                    continue
                t = re.sub(r"\s*\(\s*[\d\.\,]+\s*\)\s*$", "", t).strip()
                if len(t) >= 2 and not t.lower().startswith("mostra"):
                    brands.append(t)
            brands = uniq_keep_order(brands)

        # FILTER NAMES (euristica)
        txt = soup.get_text("\n", strip=True)
        lines = [safe_text(x) for x in txt.split("\n") if safe_text(x)]

        start = None
        end = None
        for i, l in enumerate(lines):
            if l.lower() == "filtri":
                start = i
                break
        if start is not None:
            for j in range(start, min(len(lines), start + 500)):
                if "trovati" in lines[j].lower():
                    end = j
                    break
        if start is None:
            out["brands"] = brands[:60]
            out["filters"] = []
            return out

        window = lines[start:(end if end else min(len(lines), start + 260))]
        out["raw_lines_window"] = window[:120]

        filters_found = []
        stop_values = {
            "filtri", "filtri attivi", "mostra", "ordina per", "rilevanza",
            "prezzo listino", "risultati", "filtra prodotti", "categoria",
            "normalmente disponibile", "ordinabile a fornitore"
        }

        for l in window:
            ll = l.lower()
            if ll in stop_values:
                continue
            m = re.match(r"^(.+?)\s*\(\s*[\d\.\,]+\s*\)\s*$", l)
            if not m:
                continue
            name = m.group(1).strip()
            nlow = name.lower()
            if nlow in stop_values:
                continue
            if len(name) < 6:
                continue
            # evita che brand finiscano tra i filtri (capita se layout cambia)
            if any(nlow == b.lower() for b in brands[:30]):
                continue
            filters_found.append(name)

        out["brands"] = brands[:60]
        out["filters"] = uniq_keep_order(filters_found)[:40]
        return out

    except Exception as e:
        out["error"] = str(e)
        return out

# =========================
# COMPETITOR SCRAPING (for word-count + outline hints)
# =========================
def remove_boilerplate(soup: BeautifulSoup):
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe"]):
        tag.decompose()
    for selector in ["nav", "header", "footer", "aside", "form"]:
        for tag in soup.select(selector):
            tag.decompose()
    for cls in ["cookie", "cookies", "cookie-banner", "newsletter", "modal", "popup"]:
        for tag in soup.select(f".{cls}"):
            tag.decompose()
    return soup

def detect_main_container(soup: BeautifulSoup):
    for tag in ["article", "main"]:
        el = soup.find(tag)
        if el:
            txt = el.get_text(" ", strip=True)
            if txt and len(txt) > 600:
                return el
    return soup.body if soup.body else soup

def scrape_competitor_page(url: str, max_text_chars=14000):
    data = {
        "url": url,
        "domain": domain_of(url),
        "title": "",
        "h1": "",
        "h2": [],
        "h3": [],
        "bullets": [],
        "text_sample": "",
        "word_count": 0,
        "http_status": None,
        "error": "",
    }
    try:
        r = HTTP.get(url, headers=UA, timeout=18, allow_redirects=True)
        data["http_status"] = r.status_code
        if r.status_code >= 400 or not r.text:
            data["error"] = f"HTTP {r.status_code}"
            return data

        soup = BeautifulSoup(r.text, "html.parser")
        soup = remove_boilerplate(soup)
        if soup.title and soup.title.string:
            data["title"] = safe_text(soup.title.string)

        main = detect_main_container(soup)

        for tag in main.find_all(["h1", "h2", "h3"])[:110]:
            txt = safe_text(tag.get_text(" ", strip=True))
            if not txt:
                continue
            if tag.name == "h1" and not data["h1"]:
                data["h1"] = txt
            elif tag.name == "h2" and len(data["h2"]) < 30:
                data["h2"].append(txt)
            elif tag.name == "h3" and len(data["h3"]) < 45:
                data["h3"].append(txt)

        for li in main.find_all("li")[:160]:
            t = safe_text(li.get_text(" ", strip=True))
            if t and 18 <= len(t) <= 200:
                data["bullets"].append(t)
        data["bullets"] = data["bullets"][:45]

        paragraphs = main.find_all("p")
        p_text = " ".join([safe_text(p.get_text(" ", strip=True)) for p in paragraphs])
        b_text = " ".join(data["bullets"])
        text = safe_text((p_text + " " + b_text).strip())
        if len(text) > max_text_chars:
            text = text[:max_text_chars]
        data["text_sample"] = text[:3200]
        data["word_count"] = word_count(text)

        return data
    except Exception as e:
        data["error"] = str(e)
        return data

def competitor_block_for_prompt(comps):
    out = []
    for c in comps:
        out.append({
            "url": c.get("url"),
            "domain": c.get("domain"),
            "h1": clamp(c.get("h1"), 120),
            "h2": [clamp(x, 110) for x in (c.get("h2") or [])[:10]],
            "bullets": [clamp(x, 160) for x in (c.get("bullets") or [])[:14]],
            "word_count": c.get("word_count", 0),
            "text_sample": clamp(c.get("text_sample"), 900),
            "error": c.get("error", ""),
        })
    return out

def build_word_target(avg_wc: int, margin_pct: int, fallback_range: str):
    if avg_wc and avg_wc > 200:
        lo = int(avg_wc * (1 + max(0, margin_pct - 10) / 100))
        hi = int(avg_wc * (1 + (margin_pct + 10) / 100))
        lo = max(300, lo)
        hi = max(lo + 150, hi)
        return avg_wc, f"{lo}–{hi}"
    return 0, fallback_range

# =========================
# SERP (optional) via SerpApi
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 60)
def serpapi_search(query: str, api_key: str, gl="it", hl="it", domain="google.it"):
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

def pick_competitor_urls_from_serp(serp_json, preferred_domains, max_urls=10, exclude_domains=None):
    if not serp_json:
        return []
    exclude_domains = exclude_domains or []
    preferred, others = [], []
    for res in serp_json.get("organic_results", [])[:25]:
        u = res.get("link")
        if not u:
            continue
        d = domain_of(u)
        if any(ex in d for ex in exclude_domains):
            continue
        if any(pd in d for pd in preferred_domains):
            preferred.append(u)
        else:
            others.append(u)

    out = []
    for u in preferred + others:
        if u not in out:
            out.append(u)
        if len(out) >= max_urls:
            break
    return out

def serp_snapshot_compact(serp_json, max_items=10):
    if not serp_json:
        return None
    return {
        "organic": [
            {
                "position": r.get("position"),
                "title": r.get("title"),
                "link": r.get("link"),
                "source": r.get("source"),
                "snippet": r.get("snippet"),
            }
            for r in serp_json.get("organic_results", [])[:max_items]
        ],
        "paa": [x.get("question") for x in serp_json.get("related_questions", [])[:20] if x.get("question")],
        "related": [x.get("query") for x in serp_json.get("related_searches", [])[:20] if x.get("query")],
    }

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Configurazione")

    openai_api_key = st.text_input("OpenAI key", type="password")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]

    st.markdown("---")
    st.subheader("SERP (opzionale)")
    use_serp = st.toggle("Usa SerpApi per trovare competitor in SERP", value=True)
    serp_api_key = st.text_input("SerpApi key", type="password") if use_serp else ""
    serp_max_urls = st.slider("Competitor dalla SERP", 3, 15, 10)

    st.markdown("---")
    st.subheader("Competitor da privilegiare")
    preferred_domains_text = st.text_area(
        "Domini prioritari (uno per riga)",
        value="\n".join([
            "rs-online.com",
            "sacchi.it",
            "sonepar.it",
            "marchiol.com",
            "emmstore.it",
            "comet.it",
        ]),
        height=120
    )

    st.markdown("---")
    st.subheader("Lunghezza")
    margin_pct = st.slider("Margine su media competitor (%)", 0, 60, 20, step=5)
    fallback_range = st.selectbox("Range default se non ho competitor utili", ["450–750", "550–900", "700–1100"], index=1)

    st.markdown("---")
    st.subheader("Struttura")
    max_h2 = st.slider("Massimo H2", 4, 10, 8)

    st.markdown("---")
    st.subheader("Brand")
    brand_name = st.text_input("Nome brand", value="Rexel")
    site_url = st.text_input("Sito", value="https://rexel.it/")
    exclude_own_domain = st.toggle("Escludi dominio brand dalla SERP", value=True)

# =========================
# INPUT: SINGLE OR BATCH
# =========================
tab1, tab2 = st.tabs(["Singola categoria", "Batch da Excel"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        page_url = st.text_input(
            "URL categoria (opzionale, utile su rexel.it per estrarre marche/filtri)",
            placeholder="https://rexel.it/categoria/…"
        )
    with col2:
        query = st.text_input("Query obiettivo", placeholder="contattori e teleruttori")

    st.markdown("### URL competitor (opzionale)")
    manual_competitor_urls = st.text_area("Uno per riga (anche vuoto)", height=110, placeholder="Incolla URL competitor…")

    run_single = st.button("🚀 Genera output (singolo)")

with tab2:
    st.markdown("Carica un Excel con **due colonne**: una per le **URL** e una per le **query**.")
    batch_file = st.file_uploader("Excel (xlsx)", type=["xlsx"])
    st.caption("Le colonne possono chiamarsi anche diversamente: il tool prova ad auto-rilevarle (url/address/pagina, query/keyword).")
    st.markdown("### URL competitor (opzionale, usate per tutte le righe)")
    batch_manual_competitor_urls = st.text_area("Competitor globali (uno per riga)", height=110, placeholder="Incolla URL competitor…")
    run_batch = st.button("🚀 Genera output (batch)")

# =========================
# CORE: build prompt + call AI
# =========================
def build_prompts_for_row(
    brand_name: str,
    site_url: str,
    page_url: str,
    query: str,
    max_h2: int,
    avg_wc: int,
    target_range: str,
    brands_final: list,
    filters_final: list,
    competitor_block: list
):
    system_prompt = """
Sei un senior SEO e-commerce B2B specializzato in materiale elettrico.
Crei spunti operativi per testi di categoria (category page) destinati a buyer tecnici.

Regole editoriali:
- Output in italiano.
- Stile tecnico e concreto, niente fuffa.
- Maiuscole: sentence case (solo prima lettera maiuscola), evita title case.
- Niente numeri inventati e niente claim non verificabili.
- Poco informativo: evita sezioni lunghe "cos'è".
"""

    user_prompt = f"""
Brand: {brand_name}
Sito: {site_url}
URL pagina (se disponibile): {page_url if page_url else "non fornita"}
Query obiettivo: "{query}"

Dati pagina (se disponibili, estratti automaticamente da Rexel):
- marche principali: {brands_final if brands_final else "non disponibili"}
- filtri (nomi facet): {filters_final if filters_final else "non disponibili"}

Dati competitor (se disponibili):
- media parole competitor stimata: {avg_wc if avg_wc else "non disponibile"}
- lunghezza consigliata: {target_range}
- estratti competitor (json): {json.dumps(competitor_block, ensure_ascii=False)}

VINCOLI OUTPUT:
- Voglio solo le sezioni richieste sotto, niente meta title/description.
- H1: deve corrispondere alla query obiettivo. Usa la query come base e rendila in sentence case senza cambiare il significato.
- Lunghezza consigliata: mostra solo il range, senza motivazione.
- Outline: formato H2/H3 con Nota (IT) come guida pratica.
- FAQ: solo domande, senza risposte.

FORMATO OBBLIGATORIO:

## h1
- (una sola riga)

## lunghezza consigliata
- {target_range}

## outline (H2/H3)
Massimo {max_h2} H2.
Per ogni H2: 2–4 H3 + una Nota (IT) su cosa scrivere (criteri scelta, compatibilità, differenze tra varianti, errori comuni).
Se sono presenti filtri reali, usali per orientare l’outline.

## faq (solo domande)
5–8 domande che un buyer tecnico farebbe.
Niente risposte.
"""
    return system_prompt, user_prompt

def call_openai_brief(openai_api_key: str, system_prompt: str, user_prompt: str):
    client = OpenAI(api_key=openai_api_key)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.35
    )
    return resp.choices[0].message.content

def parse_output_sections(text: str):
    """
    Estrae sezioni in modo best-effort per esportare in Excel.
    """
    t = text or ""
    def grab(section_name):
        m = re.search(rf"^##\s+{re.escape(section_name)}\s*$([\s\S]*?)(?=^##\s+|\Z)", t, flags=re.M)
        return safe_text(m.group(1)) if m else ""

    h1 = grab("h1")
    length = grab("lunghezza consigliata")
    outline = grab("outline (H2/H3)")
    faq = grab("faq (solo domande)")

    # pulizia rapida (togli bullet leader "- " se presente)
    h1 = re.sub(r"^\-\s*", "", h1).strip()
    length = re.sub(r"^\-\s*", "", length).strip()

    return h1, length, outline, faq

# =========================
# SHARED: competitor collection
# =========================
def build_competitor_data(query: str, manual_urls: list, preferred_domains: list, own_domain: str):
    serp_snap = None
    urls = manual_urls[:]

    if use_serp:
        if not serp_api_key:
            return urls, None, []  # handled upstream
        serp_json = serpapi_search(query, serp_api_key, gl="it", hl="it", domain="google.it")
        serp_snap = serp_snapshot_compact(serp_json, max_items=10) if serp_json else None
        serp_urls = pick_competitor_urls_from_serp(
            serp_json,
            preferred_domains=preferred_domains,
            max_urls=serp_max_urls,
            exclude_domains=[own_domain] if own_domain else []
        ) if serp_json else []
        for u in serp_urls:
            if u not in urls:
                urls.append(u)

    # scrape (max 12)
    urls_to_scrape = urls[:12]
    results = []
    if urls_to_scrape:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(scrape_competitor_page, u) for u in urls_to_scrape]
            for f in as_completed(futs):
                results.append(f.result())

    # sort by preferred domain then wc
    def pref_rank(d):
        d = (d or "").lower()
        for i, pdm in enumerate(preferred_domains):
            if pdm in d:
                return 0, i
        return 1, 999

    results_sorted = sorted(results, key=lambda x: (pref_rank(x.get("domain")), -(x.get("word_count") or 0)))
    return urls, serp_snap, results_sorted

# =========================
# RUN: SINGLE
# =========================
def run_for_one(page_url: str, query: str, competitor_urls_text: str):
    if not openai_api_key:
        st.error("Inserisci OpenAI key.")
        return

    q = safe_text(query)
    if not q:
        st.error("Inserisci la query obiettivo.")
        return

    preferred_domains = normalize_lines(preferred_domains_text)
    own_domain = domain_of(site_url) if exclude_own_domain else ""

    # Rexel facets
    rexel_debug = None
    brands_final, filters_final = [], []
    if page_url and "rexel.it" in domain_of(page_url):
        status = st.status("🧲 Estrazione marche/filtri da Rexel…", expanded=True)
        rexel_debug = scrape_rexel_facets(page_url)
        brands_final = rexel_debug.get("brands", []) or []
        filters_final = rexel_debug.get("filters", []) or []
        status.update(label="Estrazione Rexel pronta", state="complete", expanded=False)

        with st.expander("Debug Rexel (marche/filtri trovati)", expanded=True):
            st.write(f"HTTP: {rexel_debug.get('http_status')} — URL: {rexel_debug.get('url')}")
            if rexel_debug.get("error"):
                st.error(rexel_debug["error"])
            st.markdown("**Marche trovate**")
            st.write(brands_final if brands_final else "Nessuna marca trovata")
            st.markdown("**Filtri trovati**")
            st.write(filters_final if filters_final else "Nessun filtro trovato")
            st.markdown("**Finestra testo (debug)**")
            st.code("\n".join((rexel_debug.get("raw_lines_window") or [])[:80]))

    # competitor data
    manual_urls = normalize_lines(competitor_urls_text)
    status = st.status("🕷️ Lettura competitor…", expanded=True)
    all_urls, serp_snap, comp_results = build_competitor_data(q, manual_urls, preferred_domains, own_domain)
    status.update(label="Competitor pronti", state="complete", expanded=False)

    if serp_snap:
        with st.expander("SERP snapshot (debug)", expanded=False):
            st.json(serp_snap, expanded=2)

    with st.expander("Competitor snapshot (debug)", expanded=False):
        for c in comp_results:
            label = f"{c.get('domain','')} — {clamp(c.get('h1') or c.get('title') or c.get('url'), 80)}"
            with st.expander(label, expanded=False):
                st.json({
                    "url": c.get("url"),
                    "domain": c.get("domain"),
                    "http_status": c.get("http_status"),
                    "h1": c.get("h1"),
                    "h2": c.get("h2", [])[:18],
                    "bullets": c.get("bullets", [])[:20],
                    "word_count": c.get("word_count"),
                    "text_sample": c.get("text_sample", ""),
                    "error": c.get("error", ""),
                }, expanded=2)

    wc_list = [r.get("word_count") for r in comp_results if isinstance(r.get("word_count"), int) and r.get("word_count") and r.get("word_count") > 0]
    avg_wc = int(np.mean(wc_list)) if wc_list else 0
    avg_wc, target_range = build_word_target(avg_wc, margin_pct, fallback_range)

    st.markdown("### 📏 Lunghezza consigliata")
    st.write(f"**{target_range} parole**")

    comp_block = competitor_block_for_prompt(comp_results[:10])
    system_prompt, user_prompt = build_prompts_for_row(
        brand_name=brand_name,
        site_url=site_url,
        page_url=page_url,
        query=q,
        max_h2=max_h2,
        avg_wc=avg_wc,
        target_range=target_range,
        brands_final=brands_final,
        filters_final=filters_final,
        competitor_block=comp_block
    )

    status = st.status("🧠 Generazione output…", expanded=True)
    try:
        out = call_openai_brief(openai_api_key, system_prompt, user_prompt)
        status.update(label="Output pronto", state="complete", expanded=False)
        st.markdown("## ✅ Output")
        st.markdown(out)
    except Exception as e:
        status.update(label="Errore", state="error", expanded=False)
        st.error(f"Errore OpenAI: {e}")

# =========================
# RUN: BATCH
# =========================
def detect_url_query_columns(df: pd.DataFrame):
    cols = [c for c in df.columns]
    cols_l = {str(c).strip().lower(): c for c in cols}

    def pick(candidates):
        for cand in candidates:
            for lc, orig in cols_l.items():
                if cand == lc or cand in lc:
                    return orig
        return None

    url_col = pick(["url", "address", "pagina", "page", "link", "indirizzo"])
    query_col = pick(["query", "keyword", "kw", "search", "ricerca"])

    return url_col, query_col

def run_batch_from_excel(file, competitor_urls_text: str):
    if not openai_api_key:
        st.error("Inserisci OpenAI key.")
        return

    try:
        df = pd.read_excel(file, engine="openpyxl")
    except Exception as e:
        st.error(f"Non riesco a leggere l’Excel: {e}")
        return

    if df is None or df.empty:
        st.error("Excel vuoto.")
        return

    url_col, query_col = detect_url_query_columns(df)
    if not url_col or not query_col:
        st.error("Non riesco a trovare le colonne. Serve una colonna URL e una colonna query.")
        st.write("Colonne trovate:", list(df.columns))
        return

    df_work = df[[url_col, query_col]].copy()
    df_work.columns = ["url", "query"]
    df_work["url"] = df_work["url"].astype(str).map(safe_text)
    df_work["query"] = df_work["query"].astype(str).map(safe_text)
    df_work = df_work[(df_work["url"] != "") & (df_work["query"] != "")].reset_index(drop=True)

    if df_work.empty:
        st.error("Nessuna riga valida (URL/query).")
        return

    preferred_domains = normalize_lines(preferred_domains_text)
    own_domain = domain_of(site_url) if exclude_own_domain else ""
    manual_urls_global = normalize_lines(competitor_urls_text)

    st.info(f"Righe da processare: {len(df_work)}")
    prog = st.progress(0.0)
    results_rows = []

    for i, row in df_work.iterrows():
        page_url = row["url"]
        q = row["query"]

        # Rexel facets per URL
        brands_final, filters_final = [], []
        rexel_http = None
        rexel_err = ""
        if page_url and "rexel.it" in domain_of(page_url):
            rexel_debug = scrape_rexel_facets(page_url)
            rexel_http = rexel_debug.get("http_status")
            rexel_err = rexel_debug.get("error", "")
            brands_final = (rexel_debug.get("brands") or [])[:60]
            filters_final = (rexel_debug.get("filters") or [])[:40]

        # competitor data per query
        all_urls, serp_snap, comp_results = build_competitor_data(q, manual_urls_global, preferred_domains, own_domain)
        wc_list = [r.get("word_count") for r in comp_results if isinstance(r.get("word_count"), int) and r.get("word_count") and r.get("word_count") > 0]
        avg_wc = int(np.mean(wc_list)) if wc_list else 0
        avg_wc, target_range = build_word_target(avg_wc, margin_pct, fallback_range)
        comp_block = competitor_block_for_prompt(comp_results[:10])

        system_prompt, user_prompt = build_prompts_for_row(
            brand_name=brand_name,
            site_url=site_url,
            page_url=page_url,
            query=q,
            max_h2=max_h2,
            avg_wc=avg_wc,
            target_range=target_range,
            brands_final=brands_final,
            filters_final=filters_final,
            competitor_block=comp_block
        )

        try:
            out = call_openai_brief(openai_api_key, system_prompt, user_prompt)
            h1, length, outline, faq = parse_output_sections(out)

            results_rows.append({
                "url": page_url,
                "query": q,
                "h1": h1,
                "lunghezza_consigliata": length if length else target_range,
                "outline": outline,
                "faq_domande": faq,
                "debug_rexel_http": rexel_http,
                "debug_rexel_brands_count": len(brands_final),
                "debug_rexel_filters_count": len(filters_final),
                "raw_output": out
            })

        except Exception as e:
            results_rows.append({
                "url": page_url,
                "query": q,
                "h1": "",
                "lunghezza_consigliata": target_range,
                "outline": "",
                "faq_domande": "",
                "debug_rexel_http": rexel_http,
                "debug_rexel_brands_count": len(brands_final),
                "debug_rexel_filters_count": len(filters_final),
                "raw_output": f"ERROR: {e}"
            })

        prog.progress((i + 1) / len(df_work))

    prog.empty()

    out_df = pd.DataFrame(results_rows)

    st.success("Batch completato.")
    st.dataframe(out_df[["url", "query", "h1", "lunghezza_consigliata", "debug_rexel_brands_count", "debug_rexel_filters_count"]], use_container_width=True)

    with st.expander("Debug (prime 3 righe: outline + faq)", expanded=False):
        for r in out_df.head(3).to_dict(orient="records"):
            st.markdown(f"### {r['query']}")
            st.write(r["url"])
            st.markdown("**H1**")
            st.write(r["h1"])
            st.markdown("**Lunghezza**")
            st.write(r["lunghezza_consigliata"])
            st.markdown("**Outline**")
            st.write(r["outline"])
            st.markdown("**FAQ (domande)**")
            st.write(r["faq_domande"])
            st.markdown("---")

    st.download_button(
        "📥 Scarica output batch (xlsx)",
        data=to_excel_bytes(out_df, sheet_name="brief"),
        file_name="b2b_category_brief_batch.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =========================
# DISPATCH
# =========================
if run_single:
    run_for_one(page_url=page_url, query=query, competitor_urls_text=manual_competitor_urls)

if run_batch:
    if batch_file is None:
        st.error("Carica un Excel.")
    else:
        run_batch_from_excel(batch_file, competitor_urls_text=batch_manual_competitor_urls)
