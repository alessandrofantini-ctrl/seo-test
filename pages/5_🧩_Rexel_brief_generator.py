import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter, Retry
import re
import json
import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="B2B category brief (e-commerce materiale elettrico)", layout="wide")
st.title("🧩 B2B category brief (e-commerce materiale elettrico)")
st.caption("Brief tecnici per copywriter: struttura categoria, criteri di scelta, parametri e filtri. Poco informativo, molto operativo.")

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
def safe_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "").lower()
    except Exception:
        return ""

def remove_boilerplate(soup: BeautifulSoup):
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe"]):
        tag.decompose()
    for selector in ["nav", "header", "footer", "aside", "form"]:
        for tag in soup.select(selector):
            tag.decompose()
    # banner/popups frequenti
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

def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))

def clamp(s: str, n: int) -> str:
    s = safe_text(s)
    return s[:n] if len(s) > n else s

def normalize_lines(text_area_value: str):
    return [safe_text(x) for x in (text_area_value or "").splitlines() if safe_text(x)]

def uniq_keep_order(items):
    out = []
    for x in items:
        x = safe_text(x)
        if x and x not in out:
            out.append(x)
    return out

# =========================
# REXEL FACETS (brands + filters) FROM CATEGORY URL
# =========================
def scrape_rexel_facets(category_url: str):
    """
    Estrae:
    - brand (nomi) dal blocco id="collapseBrands" se presente, altrimenti euristica testuale
    - nomi filtri (facet name) dalla sezione "Filtri" (euristica su testo)
    Mostra anche raw_lines utili per debug.
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

        # 1) BRAND: prova dal div specifico (come da tua indicazione)
        brands = []
        brands_div = soup.find(id="collapseBrands")
        if brands_div:
            # spesso i brand sono <a> o <label> con testo tipo "SCHNEIDER ELECTRIC (1.967)"
            for el in brands_div.find_all(["a", "label", "span"], limit=250):
                t = safe_text(el.get_text(" ", strip=True))
                if not t:
                    continue
                # pulisci conteggi
                t = re.sub(r"\s*\(\s*[\d\.\,]+\s*\)\s*$", "", t).strip()
                # evita stringhe troppo corte o generiche
                if len(t) >= 2 and not t.lower().startswith("mostra"):
                    brands.append(t)
            brands = uniq_keep_order(brands)

        # fallback brand: euristica sulla zona testo tra "Brand" e "Filtri"
        if not brands:
            txt = soup.get_text("\n", strip=True)
            lines = [safe_text(x) for x in txt.split("\n") if safe_text(x)]
            try:
                i_brand = next(i for i, l in enumerate(lines) if l.lower() == "brand")
                i_filtri = next(i for i, l in enumerate(lines) if l.lower() == "filtri")
                window = lines[i_brand:i_filtri]
                # trova righe con conteggio
                for l in window:
                    m = re.match(r"^(.+?)\s*\(\s*[\d\.\,]+\s*\)\s*$", l)
                    if m:
                        nm = m.group(1).strip()
                        if len(nm) >= 2:
                            brands.append(nm)
                brands = uniq_keep_order(brands)
            except Exception:
                pass

        # 2) FILTRI: euristica sul blocco testuale dopo "Filtri" e prima di "Trovati"
        filters_found = []
        txt = soup.get_text("\n", strip=True)
        lines = [safe_text(x) for x in txt.split("\n") if safe_text(x)]

        # prova a delimitare la finestra utile
        start = None
        end = None
        for i, l in enumerate(lines):
            if l.lower() == "filtri":
                start = i
                break
        if start is not None:
            for j in range(start, min(len(lines), start + 400)):
                if "trovati" in lines[j].lower():
                    end = j
                    break
        if start is None:
            # se non troviamo "Filtri", niente
            out["brands"] = brands
            out["filters"] = []
            return out

        window = lines[start:(end if end else min(len(lines), start + 220))]

        # salva una finestra ridotta per debug
        out["raw_lines_window"] = window[:120]

        # nella finestra ci sono sia nomi filtri che valori. Teniamo:
        # - righe che non sono "Filtri", "Filtri attivi", ecc.
        # - righe che NON sembrano un valore (es. "Normalmente disponibile") in base a stoplist
        stop_values = {
            "normalmente disponibile",
            "ordinabile a fornitore",
            "filtri attivi",
            "mostra",
            "ordina per",
            "rilevanza",
            "prezzo listino",
            "risultati",
            "filtra prodotti",
            "categoria",
        }

        # pattern tipico: "nome filtro (1.886)"
        for l in window:
            ll = l.lower()
            if ll in stop_values:
                continue
            if ll.startswith("agg."):
                continue
            # nomi filtri spesso NON hanno conteggio? su Rexel sì.
            m = re.match(r"^(.+?)\s*\(\s*[\d\.\,]+\s*\)\s*$", l)
            if m:
                name = m.group(1).strip()
                nlow = name.lower()
                if nlow in stop_values:
                    continue
                # escludi righe molto “valore”
                if nlow in {"schneider electric", "abb", "lovato", "eaton"}:
                    continue
                # evita valori troppo brevi
                if len(name) < 6:
                    continue
                filters_found.append(name)

        # de-dup e taglio
        filters_found = uniq_keep_order(filters_found)[:40]

        out["brands"] = brands[:60]
        out["filters"] = filters_found
        return out

    except Exception as e:
        out["error"] = str(e)
        return out

# =========================
# SCRAPING competitor pages
# =========================
def scrape_page(url: str, max_text_chars=14000):
    data = {
        "url": url,
        "domain": domain_of(url),
        "title": "",
        "meta_description": "",
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
        md = soup.find("meta", attrs={"name": "description"})
        if md and md.get("content"):
            data["meta_description"] = safe_text(md.get("content"))

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

# =========================
# SERP via SerpApi (optional)
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

    preferred = []
    others = []
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
    st.subheader("SERP (opzionale ma consigliato)")
    use_serp = st.toggle("Usa SerpApi per trovare competitor in SERP", value=True)
    serp_api_key = st.text_input("SerpApi key", type="password") if use_serp else ""
    serp_max_urls = st.slider("Numero risultati SERP da considerare", 3, 15, 10)

    st.markdown("---")
    st.subheader("Competitor da privilegiare")
    st.caption("Uno per riga: domini che vuoi prioritari in SERP (es. rs-online.com).")
    preferred_domains_text = st.text_area(
        "Domini prioritari",
        value="\n".join([
            "rs-online.com",
            "sacchi.it",
            "sonepar.it",
            "marchiol.com",
            "emmstore.it",
            "comet.it",
        ]),
        height=130
    )

    st.markdown("---")
    st.subheader("Word count target")
    margin_pct = st.slider("Margine su media competitor (%)", 0, 60, 20, step=5)
    fallback_range = st.selectbox("Range default se non ho competitor utili", ["450–750", "550–900", "700–1100"], index=1)

    st.markdown("---")
    st.subheader("Stile output")
    tone = st.selectbox("Tono", ["Tecnico B2B (consigliato)", "Tecnico + commerciale", "Catalogo sintetico"], index=0)
    max_h2 = st.slider("Massimo H2", 4, 10, 8)
    include_microcopy = st.toggle("Includi microcopy + avvertenze", value=True)

    st.markdown("---")
    st.subheader("Brand")
    brand_name = st.text_input("Nome azienda/brand", value="Rexel")
    site_url = st.text_input("Sito", value="https://rexel.it/")
    exclude_own_domain = st.toggle("Escludi il dominio del brand dalla SERP", value=True)

# =========================
# MAIN INPUTS
# =========================
col1, col2 = st.columns([2, 1])
with col1:
    category_name = st.text_input("Nome categoria (es. Contattori)", placeholder="Contattori")
with col2:
    primary_kw = st.text_input("Keyword principale", placeholder="contattori")

st.markdown("### Auto-estrazione da URL categoria Rexel (opzionale)")
rexel_category_url = st.text_input(
    "URL categoria su rexel.it (serve per estrarre marche e filtri automaticamente)",
    placeholder="https://rexel.it/categoria/…"
)

st.markdown("### Extra (opzionale)")
col3, col4 = st.columns(2)
with col3:
    known_subcats = st.text_area("Sotto-categorie note (una per riga)", height=120, placeholder="Es.\nContattori modulari\nContattori tripolari\nContatti ausiliari")
with col4:
    known_brands = st.text_area("Marche principali (una per riga)", height=120, placeholder="Es.\nSchneider Electric\nSiemens\nABB")

known_filters = st.text_area(
    "Filtri reali del catalogo (opzionale, uno per riga)",
    height=90,
    placeholder="Es.\nNumero poli\nTensione bobina\nPotenza nominale\nMontaggio"
)

st.markdown("---")
st.markdown("### URL competitor (opzionale)")
st.caption("Se li inserisci, il tool li userà sempre. Se non li inserisci, può recuperarli dalla SERP (se attivo).")
manual_competitor_urls = st.text_area(
    "Uno per riga (anche vuoto)",
    height=110,
    placeholder="Incolla qui URL di pagine categoria competitor"
)

# =========================
# GENERATION HELPERS
# =========================
def build_word_target(avg_wc: int, margin_pct: int, fallback_range: str):
    if avg_wc and avg_wc > 200:
        lo = int(avg_wc * (1 + max(0, margin_pct - 10) / 100))
        hi = int(avg_wc * (1 + (margin_pct + 10) / 100))
        lo = max(350, lo)
        hi = max(lo + 150, hi)
        return avg_wc, f"{lo}–{hi}"
    return 0, fallback_range

def competitor_block_for_prompt(comps):
    out = []
    for c in comps:
        out.append({
            "url": c.get("url"),
            "domain": c.get("domain"),
            "h1": clamp(c.get("h1"), 140),
            "h2": [clamp(x, 120) for x in (c.get("h2") or [])[:12]],
            "bullets": [clamp(x, 160) for x in (c.get("bullets") or [])[:18]],
            "word_count": c.get("word_count", 0),
            "text_sample": clamp(c.get("text_sample"), 1200),
            "error": c.get("error", ""),
        })
    return out

# =========================
# RUN BUTTON
# =========================
if st.button("🚀 Genera brief per categoria"):
    if not openai_api_key:
        st.error("Inserisci OpenAI key.")
        st.stop()

    if not (primary_kw or category_name):
        st.error("Inserisci almeno nome categoria o keyword principale.")
        st.stop()

    # inputs
    cat = safe_text(category_name) or safe_text(primary_kw)
    kw = safe_text(primary_kw) or safe_text(category_name)

    preferred_domains = normalize_lines(preferred_domains_text)
    own_domain = domain_of(site_url) if exclude_own_domain else ""
    exclude_domains = [own_domain] if own_domain else []

    # 1) AUTO-ESTRAZIONE REXEL (brands + filters)
    rexel_debug = None
    auto_brands = []
    auto_filters = []
    if rexel_category_url and "rexel.it" in domain_of(rexel_category_url):
        status = st.status("🧲 Estrazione marche/filtri da Rexel…", expanded=True)
        rexel_debug = scrape_rexel_facets(rexel_category_url)
        auto_brands = rexel_debug.get("brands", []) or []
        auto_filters = rexel_debug.get("filters", []) or []
        status.update(label="Estrazione Rexel pronta", state="complete", expanded=False)

        with st.expander("Debug Rexel (marche/filtri trovati)", expanded=True):
            st.write(f"HTTP: {rexel_debug.get('http_status')} — URL: {rexel_debug.get('url')}")
            if rexel_debug.get("error"):
                st.error(rexel_debug["error"])
            st.markdown("**Marche trovate**")
            st.write(auto_brands if auto_brands else "Nessuna marca trovata")
            st.markdown("**Filtri trovati**")
            st.write(auto_filters if auto_filters else "Nessun filtro trovato")
            st.markdown("**Finestra testo (debug)**")
            st.code("\n".join((rexel_debug.get("raw_lines_window") or [])[:80]))

    # 2) Merge: se l’utente non ha compilato marche/filtri, usiamo quelli auto
    subcats = normalize_lines(known_subcats)

    brands_manual = normalize_lines(known_brands)
    filters_manual = normalize_lines(known_filters)

    brands_final = brands_manual if brands_manual else auto_brands
    filters_final = filters_manual if filters_manual else auto_filters

    # 3) competitor URLs
    manual_urls = normalize_lines(manual_competitor_urls)

    serp_json = None
    serp_snap = None
    serp_urls = []

    if use_serp:
        if not serp_api_key:
            st.error("SerpApi attiva: inserisci SerpApi key.")
            st.stop()

        status = st.status("🔍 Analisi SERP…", expanded=True)
        serp_json = serpapi_search(kw, serp_api_key, gl="it", hl="it", domain="google.it")
        serp_snap = serp_snapshot_compact(serp_json, max_items=10) if serp_json else None
        serp_urls = pick_competitor_urls_from_serp(
            serp_json,
            preferred_domains=preferred_domains,
            max_urls=serp_max_urls,
            exclude_domains=exclude_domains
        ) if serp_json else []
        status.update(label="SERP pronta", state="complete", expanded=False)

    competitor_urls = []
    for u in manual_urls + serp_urls:
        if u and u not in competitor_urls:
            competitor_urls.append(u)

    if not competitor_urls:
        st.error("Nessun competitor URL disponibile: inserisci URL manuali o attiva SERP.")
        st.stop()

    if serp_snap:
        with st.expander("SERP snapshot (debug)", expanded=False):
            st.json(serp_snap, expanded=2)

    # 4) scrape competitor
    status = st.status("🕷️ Lettura competitor (contenuti)…", expanded=True)
    results = []
    max_scrape = min(len(competitor_urls), 12)
    urls_to_scrape = competitor_urls[:max_scrape]

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(scrape_page, u) for u in urls_to_scrape]
        prog = status.progress(0.0)
        done = 0
        for f in as_completed(futs):
            done += 1
            prog.progress(done / max(1, len(futs)))
            results.append(f.result())
    prog.empty()
    status.update(label="Competitor letti", state="complete", expanded=False)

    # competitor debug
    with st.expander("Competitor snapshot (debug)", expanded=False):
        def pref_rank(d):
            d = (d or "").lower()
            for i, pdm in enumerate(preferred_domains):
                if pdm in d:
                    return 0, i
            return 1, 999

        results_sorted = sorted(results, key=lambda x: (pref_rank(x.get("domain")), -(x.get("word_count") or 0)))
        for c in results_sorted:
            label = f"{c.get('domain','')} — {clamp(c.get('h1') or c.get('title') or c.get('url'), 80)}"
            with st.expander(label, expanded=False):
                st.json({
                    "url": c.get("url"),
                    "domain": c.get("domain"),
                    "http_status": c.get("http_status"),
                    "title": c.get("title"),
                    "meta_description": c.get("meta_description"),
                    "h1": c.get("h1"),
                    "h2": c.get("h2", [])[:20],
                    "h3": c.get("h3", [])[:20],
                    "bullets": c.get("bullets", [])[:25],
                    "word_count": c.get("word_count"),
                    "text_sample": c.get("text_sample", ""),
                    "error": c.get("error", ""),
                }, expanded=2)

    # 5) word count target
    wc_list = [r.get("word_count") for r in results if isinstance(r.get("word_count"), int) and r.get("word_count") and r.get("word_count") > 0]
    avg_wc = int(np.mean(wc_list)) if wc_list else 0
    avg_wc, target_range = build_word_target(avg_wc, margin_pct, fallback_range)

    st.markdown("### 📏 Lunghezza consigliata")
    if avg_wc:
        st.write(f"- Media competitor (stimata): **{avg_wc} parole**")
    else:
        st.write("- Media competitor non disponibile (scraping limitato o pagine non leggibili).")
    st.write(f"- Target consigliato: **{target_range} parole**")

    # 6) build prompt block
    comp_block = competitor_block_for_prompt(results_sorted[:10])

    client = OpenAI(api_key=openai_api_key)

    system_prompt = """
Sei un senior SEO e-commerce B2B specializzato in materiale elettrico.
Crei brief operativi per testi di categoria (category page) destinati a buyer tecnici.

Obiettivo:
- contenuto utile alla scelta e all'acquisto (specifiche, compatibilità, criteri, filtri)
- poco informativo, niente storia/definizioni scolastiche

Regole editoriali:
- frasi brevi, lessico tecnico, orientato ai filtri
- maiuscole: sentence case (solo prima lettera maiuscola), evita title case
- niente numeri inventati e niente claim non verificabili
- evita fuffa e ripetizioni
- output in italiano
"""

    user_prompt = f"""
Brand: {brand_name}
Sito: {site_url}
Categoria: "{cat}"
Keyword principale: "{kw}"

Contesto (se disponibile):
- sotto-categorie note: {subcats if subcats else "non fornite"}
- marche principali (auto o manuali): {brands_final if brands_final else "non fornite"}
- filtri catalogo (auto o manuali): {filters_final if filters_final else "non forniti"}

SERP/competitor (segnali estratti, possono essere incompleti):
- domini prioritari: {preferred_domains if preferred_domains else "non forniti"}
- media parole competitor stimata: {avg_wc if avg_wc else "non disponibile"}
- target parole consigliato: {target_range}
- estratti competitor (json): {json.dumps(comp_block, ensure_ascii=False)}

Vincoli:
- Non scrivere un articolo informativo: evita sezioni lunghe "cos'è".
- Struttura da category page B2B: breve intro, guida alla scelta, varianti, compatibilità, accessori, note pratiche.
- Inserisci solo parametri tecnici plausibili per la categoria (non inventare specifiche strane).
- Niente title case.

Output richiesto (formato obbligatorio):

## meta
- meta title (3 varianti, <= 60 caratteri):
  - una variante con pattern: "{kw} | {brand_name}"
  - due varianti tecniche orientate a selezione/assortimento (no slogan)
- meta description (3 varianti, <= 155 caratteri):
  - orientate a specifiche, assortimento, filtri, uso B2B
  - niente promesse vaghe

## h1
- 1 proposta (sentence case)

## lunghezza consigliata
- se la media competitor è disponibile: conferma target = media + 15–25%
- se non disponibile: conferma target = {fallback_range}
- motivazione: 2 righe pratiche (solo utilità e-commerce)

## outline (H2/H3)
Massimo {max_h2} H2.
Per ogni H2:
- 2–4 H3
- una Nota (IT) pratica su cosa scrivere con focus su: criteri di scelta, compatibilità/installazione, differenze tra varianti, errori comuni.
Evita H2 generici tipo "cos'è" (se serve, massimo 3 righe in intro).

## parametri tecnici da coprire (checklist)
Lista 12–20 punti tecnici tipici della categoria.

## filtri consigliati per e-commerce
Suggerisci 10–15 filtri/facet realistici per la categoria.
Se sono stati forniti filtri reali, usali come base e completa solo dove serve.

{"## microcopy utile\n- 6 micro-frasi per aiutare la selezione\n- 6 avvertenze/compatibilità\n" if include_microcopy else ""}

## faq tecniche (5)
Domande che un buyer B2B fa davvero. Risposte 1–2 frasi, pratiche.

## note per copywriter
5 regole pratiche per scrivere la categoria:
- tono
- lunghezza paragrafi
- uso bullet
- densità keyword naturale (senza ripetizioni)
- parole/claim da evitare

Non aggiungere altre sezioni.
"""

    status = st.status("🧠 Generazione brief…", expanded=True)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.35
        )
        brief = resp.choices[0].message.content
        status.update(label="Brief pronto", state="complete", expanded=False)

        st.markdown("## ✅ Brief per copywriter")
        st.markdown(brief)

        st.download_button(
            "📥 Scarica brief (txt)",
            data=brief.encode("utf-8"),
            file_name=f"brief_categoria_{re.sub(r'[^a-z0-9]+','_',kw.lower()).strip('_')}.txt",
            mime="text/plain"
        )

    except Exception as e:
        status.update(label="Errore", state="error", expanded=False)
        st.error(f"Errore OpenAI: {e}")
