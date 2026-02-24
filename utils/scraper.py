import json
import re
import streamlit as st
from bs4 import BeautifulSoup
from collections import Counter
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.helpers import HTTP, UA, STOPWORDS, safe_text, truncate_chars, domain_of, tokenize, normalize_sentence_case


# =========================
# SOUP HELPERS
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


def extract_json_ld(soup: BeautifulSoup):
    scripts = soup.find_all("script", type="application/ld+json")
    out = []
    for sc in scripts[:12]:
        txt = sc.get_text(strip=True)
        if not txt:
            continue
        try:
            out.append(json.loads(txt))
        except Exception:
            pass
    return out


# =========================
# CORE SCRAPER
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 60)
def scrape_site_content(url, include_meta=True, include_schema=True, max_text_chars=9000):
    data = {
        "url": url,
        "domain": domain_of(url),
        "title": "",
        "meta_description": "",
        "canonical": "",
        "h1": "",
        "h2": [],
        "h3": [],
        "word_count": 0,
        "text_sample": "",
        "top_terms": [],
        "lang": "",
        "schema_types": [],
        "has_faq_schema": False,
        "question_headings": [],
    }

    try:
        resp = HTTP.get(url, headers=UA, timeout=18, allow_redirects=True)
        if resp.status_code >= 400 or not resp.text:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        soup = remove_boilerplate(soup)

        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            data["lang"] = safe_text(html_tag.get("lang"))

        if include_meta:
            if soup.title and soup.title.string:
                data["title"] = safe_text(soup.title.string)
            md = soup.find("meta", attrs={"name": "description"})
            if md and md.get("content"):
                data["meta_description"] = safe_text(md.get("content"))
            canon = soup.find("link", rel=lambda x: x and "canonical" in x.lower())
            if canon and canon.get("href"):
                data["canonical"] = safe_text(canon.get("href"))

        main = detect_main_container(soup)

        for tag in main.find_all(["h1", "h2", "h3"])[:80]:
            txt = safe_text(tag.get_text(" ", strip=True))
            if not txt:
                continue
            if tag.name == "h1" and not data["h1"]:
                data["h1"] = txt
            elif tag.name == "h2" and len(data["h2"]) < 30:
                data["h2"].append(txt)
            elif tag.name == "h3" and len(data["h3"]) < 45:
                data["h3"].append(txt)
            if "?" in txt:
                data["question_headings"].append(txt)

        paragraphs = main.find_all("p")
        lis = main.find_all("li")
        p_text = " ".join([safe_text(p.get_text(" ", strip=True)) for p in paragraphs])
        li_text = " ".join([safe_text(li.get_text(" ", strip=True)) for li in lis[:140]])
        text_content = safe_text((p_text + " " + li_text).strip())
        if len(text_content) > max_text_chars:
            text_content = text_content[:max_text_chars]

        data["word_count"] = len(text_content.split()) if text_content else 0
        data["text_sample"] = text_content[:2400]

        toks = tokenize(text_content)
        common = Counter(toks).most_common(25)
        data["top_terms"] = [t for t, _ in common]

        if include_schema:
            jlds = extract_json_ld(soup)
            types = set()
            has_faq = False
            for item in jlds:
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
            data["schema_types"] = sorted(types)[:25]
            data["has_faq_schema"] = has_faq

        return data
    except Exception:
        return None


# =========================
# SCRAPING PROFONDO CLIENTE
# =========================
def scrape_page_light(url: str) -> dict | None:
    """Scraping leggero di una singola pagina per profilazione cliente."""
    try:
        resp = HTTP.get(url, headers=UA, timeout=12, allow_redirects=True)
        if resp.status_code >= 400:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        soup = remove_boilerplate(soup)
        main = detect_main_container(soup)
        title = safe_text(soup.title.string) if soup.title else ""
        h1 = safe_text(main.find("h1").get_text()) if main.find("h1") else ""
        h2s = [safe_text(h.get_text()) for h in main.find_all("h2")[:10]]
        text = safe_text(main.get_text(" ", strip=True))[:3000]
        return {"url": url, "title": title, "h1": h1, "h2s": h2s, "text": text}
    except Exception:
        return None


def get_priority_links(base_url: str, soup: BeautifulSoup) -> list:
    """Trova link a pagine prodotti/servizi/chi siamo nello stesso dominio."""
    priority_keywords = [
        "prodott", "serviz", "soluzion", "chi-siamo", "about",
        "categor", "offert", "cosa-facciamo", "product", "service", "portfolio"
    ]
    base_domain = urlparse(base_url).netloc
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if urlparse(href).netloc != base_domain:
            continue
        if any(pk in href.lower() for pk in priority_keywords):
            links.append(href)
    return list(dict.fromkeys(links))


def scrape_client_deep(base_url: str, keyword: str, max_pages: int = 6) -> list:
    """
    Crawla il sito cliente su più pagine rilevanti.
    Restituisce lista di dict con contenuto pagine.
    """
    visited = set()
    pages_data = []

    # Homepage
    home = scrape_page_light(base_url)
    if not home:
        return []
    visited.add(base_url)
    pages_data.append(("homepage", home))

    # Link prioritari dalla homepage
    try:
        resp = HTTP.get(base_url, headers=UA, timeout=12)
        soup = BeautifulSoup(resp.text, "html.parser")
        priority_links = get_priority_links(base_url, soup)
    except Exception:
        priority_links = []

    # Scraping parallelo pagine prioritarie
    def score_page(pd):
        if not pd:
            return 0
        kw_tokens = tokenize(keyword)
        page_tokens = tokenize(pd.get("text", ""))
        return sum(1 for t in kw_tokens if t in page_tokens)

    futures_map = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for url in priority_links[:15]:
            if url not in visited and len(visited) < max_pages + 1:
                visited.add(url)
                futures_map[ex.submit(scrape_page_light, url)] = url

        scored = []
        for fut in as_completed(futures_map):
            pd = fut.result()
            if pd and pd.get("text") and len(pd["text"]) > 150:
                scored.append((score_page(pd), futures_map[fut], pd))

    scored.sort(key=lambda x: x[0], reverse=True)
    for score, url, pd in scored[:max_pages]:
        label = next(
            (pk for pk in ["prodott", "serviz", "soluzion", "about", "chi-siamo"] if pk in url.lower()),
            "pagina"
        )
        pages_data.append((label, pd))

    return pages_data


# =========================
# SERP + COMPETITOR
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


def build_serp_snapshot(serp_json, max_items):
    snapshot = {"organic": [], "paa": [], "related_searches": [], "features": []}
    if not serp_json:
        return snapshot

    for res in serp_json.get("organic_results", [])[:max_items]:
        snapshot["organic"].append({
            "position": res.get("position"),
            "title": res.get("title"),
            "link": res.get("link"),
            "snippet": res.get("snippet") or res.get("snippet_highlighted_words"),
            "source": res.get("source"),
        })
    for q in serp_json.get("related_questions", [])[:20]:
        if q.get("question"):
            snapshot["paa"].append(q.get("question"))
    for r in serp_json.get("related_searches", [])[:20]:
        if r.get("query"):
            snapshot["related_searches"].append(r.get("query"))

    for feature in ["answer_box", "knowledge_graph", "shopping_results",
                    "local_results", "top_stories", "inline_videos"]:
        if serp_json.get(feature):
            snapshot["features"].append(feature)

    return snapshot


def aggregate_competitor_insights(competitors, target_lang):
    h2_all, terms_all, q_all = [], [], []
    for c in competitors:
        for h2 in c.get("h2", [])[:30]:
            h = safe_text(h2)
            if h:
                h2_all.append(h.lower())
        for t in c.get("top_terms", [])[:25]:
            terms_all.append(t.lower())
        for q in c.get("question_headings", [])[:25]:
            qq = safe_text(q)
            if qq:
                q_all.append(qq.lower())

    def norm_heading(h):
        h = re.sub(r"\s+", " ", h)
        h = re.sub(r"[^\w\sàèéìòùäöüßñç-]", "", h)
        return h.strip()

    top_h2 = [normalize_sentence_case(x) for x, _ in Counter([norm_heading(x) for x in h2_all if x]).most_common(12)]
    top_terms = [x for x, _ in Counter([x for x in terms_all if x and x not in STOPWORDS]).most_common(20)]
    top_q = [normalize_sentence_case(x) for x, _ in Counter([norm_heading(x) for x in q_all if x]).most_common(8)]

    return {"top_h2": top_h2, "top_terms": top_terms, "top_questions": top_q}
