import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from urllib.parse import urlparse
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 120

DEFAULT_LINKS_PER_1000_WORDS = 4
DEFAULT_MIN_WORDS_PER_LINK = 180
DEFAULT_MAX_LINKS_PER_PARAGRAPH = 1
DEFAULT_MAX_SAME_URL = 1

# =========================
# PAGE
# =========================
st.set_page_config(page_title="Internal link assistant (GSC + crawl + AI)", layout="wide")
st.title("🔗 Internal link assistant (GSC + crawl + AI)")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Configurazione")

    openai_api_key = st.text_input("OpenAI key", type="password")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]

    st.markdown("---")
    st.subheader("Fonti URL interne")
    st.markdown(
        """
Carica almeno **GSC** (consigliato).  
Il **crawl** (Screaming Frog / simili) migliora moltissimo la qualità del matching perché aggiunge title/H1/estratto.

Il tool inserisce link in modo prudente: pochi link, niente ripetizioni, anchor naturali.
"""
    )

    st.markdown("---")
    st.subheader("Regole di inserimento")
    links_per_1000_words = st.slider("Link ogni 1000 parole", 1, 12, DEFAULT_LINKS_PER_1000_WORDS)
    min_words_per_link = st.slider("Minimo parole tra due link", 80, 400, DEFAULT_MIN_WORDS_PER_LINK, step=10)
    max_links_per_paragraph = st.slider("Max link per paragrafo", 1, 2, DEFAULT_MAX_LINKS_PER_PARAGRAPH)
    max_same_url = st.slider("Max volte stesso URL", 1, 3, DEFAULT_MAX_SAME_URL)

    st.markdown("---")
    st.subheader("Modalità")
    mode = st.selectbox("Inserimento", ["Safe (autoinserisce)", "Assistita (proposte)"], index=0)

    st.markdown("---")
    st.subheader("Lingua")
    st.markdown("Se hai la colonna Language nel crawl, filtriamo automaticamente per lingua dell’articolo.")
    manual_lang = st.selectbox("Lingua articolo (fallback)", ["auto", "it", "en", "es", "fr", "de"], index=0)

# =========================
# UPLOADS + INPUT
# =========================
col_a, col_b = st.columns(2)
with col_a:
    gsc_file = st.file_uploader("📈 Export Google Search Console (CSV/XLSX)", type=["csv", "xlsx"], key="gsc")
with col_b:
    crawl_file = st.file_uploader("🕷️ Crawl (Screaming Frog / CSV/XLSX) (opzionale)", type=["csv", "xlsx"], key="crawl")

st.markdown("---")
article_text = st.text_area(
    "📝 Testo articolo (Markdown o testo con paragrafi)",
    height=360,
    placeholder="Incolla qui il testo completo. Se usi Markdown, va benissimo."
)

# =========================
# HELPERS (I/O)
# =========================

def load_file(uploaded):
    """
    Ritorna:
    - DataFrame se CSV o XLSX con un solo sheet
    - dict {sheet_name: DataFrame} se XLSX con più sheet
    """
    if uploaded is None:
        return None

    name = (uploaded.name or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            uploaded.seek(0)
            xls = pd.ExcelFile(uploaded, engine="openpyxl")
            if len(xls.sheet_names) <= 1:
                return pd.read_excel(xls, sheet_name=xls.sheet_names[0])
            # più sheet: leggili tutti e fai scegliere al parser quello giusto
            sheets = {}
            for sh in xls.sheet_names:
                try:
                    sheets[sh] = pd.read_excel(xls, sheet_name=sh)
                except Exception:
                    continue
            return sheets

        # CSV
        for enc in ["utf-8", "utf-8-sig", "ISO-8859-1", "cp1252"]:
            try:
                uploaded.seek(0)
                return pd.read_csv(uploaded, encoding=enc, low_memory=False)
            except Exception:
                continue
        return None
    except Exception:
        return None


def parse_gsc(df_or_sheets):
    """
    Supporta:
    - CSV export GSC (un df)
    - XLSX export GSC con più fogli (dict di df): seleziona automaticamente il foglio "Pagine/Pages"
      oppure quello che contiene una colonna pagina.
    """
    if df_or_sheets is None:
        return None

    # Se è un dict di sheet, prova quelli più probabili prima
    if isinstance(df_or_sheets, dict):
        preferred = ["Pagine", "Pages", "Pagine principali", "Page", "Pagina"]
        # 1) tenta per nome sheet
        for sh in preferred:
            for sheet_name, df in df_or_sheets.items():
                if sh.lower() in sheet_name.lower():
                    out = parse_gsc(df)  # ricorsione su df singolo
                    if out is not None and not out.empty:
                        return out
        # 2) altrimenti prova tutti gli sheet e prendi il primo che funziona
        for sheet_name, df in df_or_sheets.items():
            out = parse_gsc(df)
            if out is not None and not out.empty:
                return out
        return None

    # Da qui in poi: df singolo
    df = df_or_sheets
    if df is None or df.empty:
        return None

    cols = {c.lower().strip(): c for c in df.columns}

    def find_col(candidates):
        for cand in candidates:
            for lc, orig in cols.items():
                if cand in lc:
                    return orig
        return None

    # Colonna URL: nei report GSC in italiano è spesso "Pagine principali"
    c_page = find_col(["page", "pagina", "pagine principali", "pagina principale", "url"])
    c_click = find_col(["click", "clic"])
    c_impr = find_col(["impression", "impressioni"])
    c_pos  = find_col(["position", "posizione"])

    if not c_page:
        return None

    out = pd.DataFrame()
    out["url"] = df[c_page].astype(str).str.strip()
    out["clicks"] = pd.to_numeric(df[c_click], errors="coerce") if c_click else 0
    out["impressions"] = pd.to_numeric(df[c_impr], errors="coerce") if c_impr else 0
    out["position"] = pd.to_numeric(df[c_pos], errors="coerce") if c_pos else np.nan

    out = out[out["url"].str.startswith("http", na=False)]
    out["domain"] = out["url"].astype(str).apply(lambda x: urlparse(x).netloc.replace("www.", "").lower() if x else "")

    out["score_gsc"] = (
        np.log1p(out["impressions"].fillna(0))
        + 1.5 * np.log1p(out["clicks"].fillna(0))
        - 0.02 * out["position"].fillna(10)
    )

    out = out.drop_duplicates(subset=["url"]).reset_index(drop=True)
    return out


# =========================
# EMBEDDINGS
# =========================
def get_embeddings(texts, client: OpenAI):
    clean = []
    for t in texts:
        t = "" if t is None else str(t)
        clean.append(t.replace("\n", " ").strip()[:8000])
    try:
        res = client.embeddings.create(input=clean, model=EMBED_MODEL)
        embs = [d.embedding for d in res.data]
        # safety
        if len(embs) != len(clean):
            embs = (embs + [None] * len(clean))[:len(clean)]
        return embs
    except Exception as e:
        st.error(f"OpenAI embeddings error: {e}")
        return [None] * len(clean)

def make_page_context(row):
    parts = []
    if row.get("title"):
        parts.append(f"Title: {row['title']}")
    if row.get("h1"):
        parts.append(f"H1: {row['h1']}")
    if row.get("meta"):
        parts.append(f"Description: {row['meta']}")
    # fallback slug
    toks = slug_tokens(row.get("url",""))
    if toks:
        parts.append("Slug: " + " ".join(toks[:12]))
    return " | ".join(parts)[:2000] if parts else ("Slug: " + " ".join(slug_tokens(row.get("url",""))[:12]))[:2000]

# =========================
# ARTICLE SEGMENTATION
# =========================
def detect_article_lang(text):
    # euristica minima (solo per fallback): se molte lettere accentate -> it/fr/es; se molte parole inglesi -> en
    t = (text or "").lower()
    if len(t) < 200:
        return ""
    it_markers = [" che ", " per ", " con ", " aziende", " guida", " vantaggi"]
    en_markers = [" the ", " and ", " guide ", " how to ", " benefits ", " business"]
    it_score = sum(t.count(m) for m in it_markers)
    en_score = sum(t.count(m) for m in en_markers)
    if it_score > en_score * 1.2:
        return "it"
    if en_score > it_score * 1.2:
        return "en"
    return ""

def split_blocks(text):
    """
    Ritorna una lista di blocchi: {type: 'heading'|'para', text: str}
    Preserva headings Markdown (#, ##, ###).
    """
    lines = (text or "").splitlines()
    blocks = []
    buf = []

    def flush_para():
        nonlocal buf
        if buf:
            para = "\n".join(buf).strip()
            if para:
                blocks.append({"type": "para", "text": para})
            buf = []

    for line in lines:
        if re.match(r"^\s{0,3}#{1,6}\s+\S+", line):
            flush_para()
            blocks.append({"type": "heading", "text": line.strip()})
        elif line.strip() == "":
            flush_para()
        else:
            buf.append(line)
    flush_para()
    return blocks

def word_count(s):
    return len(re.findall(r"\w+", s or ""))

# =========================
# LINK INSERTION
# =========================
def choose_anchor(row, fallback="approfondisci"):
    # preferisci title/h1 (più naturale), altrimenti slug
    t = safe_text(row.get("title",""))
    h1 = safe_text(row.get("h1",""))
    if t:
        return t[:70]
    if h1:
        return h1[:70]
    toks = slug_tokens(row.get("url",""))
    if toks:
        return " ".join(toks[:6])[:70]
    return fallback

def insert_link_end_of_paragraph(paragraph, anchor, url):
    # Inserisce alla fine del paragrafo in modo naturale e “safe”
    p = paragraph.rstrip()
    if p.endswith((".", "!", "?", "…")):
        return p + f" Approfondisci: [{anchor}]({url})."
    return p + f". Approfondisci: [{anchor}]({url})."

# =========================
# RUN
# =========================
if st.button("🚀 Genera internal link"):
    if not article_text.strip():
        st.error("Incolla il testo dell’articolo.")
        st.stop()
    if not gsc_file:
        st.error("Carica almeno l’export di Google Search Console.")
        st.stop()
    if not openai_api_key:
        st.error("Inserisci OpenAI key.")
        st.stop()

    df_gsc_raw = load_file(gsc_file)
    df_crawl_raw = load_file(crawl_file) if crawl_file else None

    gsc = parse_gsc(df_gsc_raw)
    if gsc is None or gsc.empty:
        st.error("Non riesco a leggere il file GSC. Assicurati che contenga una colonna Page/Pagina con le URL.")
        st.stop()

    crawl = parse_crawl(df_crawl_raw) if df_crawl_raw is not None else None

    # merge: base = GSC, arricchisci con crawl se disponibile
    pages = gsc.copy()
    if crawl is not None and not crawl.empty:
        pages = pages.merge(
            crawl[["url", "title", "h1", "meta", "lang"]],
            on="url",
            how="left"
        )
    else:
        pages["title"] = ""
        pages["h1"] = ""
        pages["meta"] = ""
        pages["lang"] = ""

    # determina lingua articolo
    lang_guess = detect_article_lang(article_text)
    lang_article = lang_guess
    if manual_lang != "auto":
        lang_article = manual_lang
    if not lang_article:
        lang_article = ""

    # filtra lingua solo se disponibile nei dati crawl (es. it-IT, en, ecc.)
    if lang_article and pages["lang"].astype(str).str.len().sum() > 0:
        pages["_lang2"] = pages["lang"].astype(str).str[:2].str.lower()
        pages_lang = pages[pages["_lang2"] == lang_article].copy()
        if len(pages_lang) >= 10:
            pages = pages_lang.drop(columns=["_lang2"], errors="ignore")
        else:
            pages = pages.drop(columns=["_lang2"], errors="ignore")

    # ordina per priorità GSC e tieni un massimo ragionevole (performance + costi embeddings)
    pages = pages.sort_values("score_gsc", ascending=False).head(1200).reset_index(drop=True)

    # debug snapshot
    with st.expander("📌 Candidate URL snapshot (debug)", expanded=False):
        st.write(f"URL candidate: {len(pages)}")
        st.dataframe(pages[["url", "clicks", "impressions", "position", "score_gsc"]].head(30), use_container_width=True)

    # embeddings pages
    client = OpenAI(api_key=openai_api_key)
    pages["ctx"] = pages.apply(lambda r: make_page_context(r), axis=1)

    st.info("🧠 Calcolo embeddings pagine interne…")
    emb_pages = []
    for i in range(0, len(pages), BATCH_SIZE):
        batch = pages["ctx"].iloc[i:i+BATCH_SIZE].tolist()
        emb_pages.extend(get_embeddings(batch, client))

    ok_idx = [i for i, e in enumerate(emb_pages) if e is not None]
    if not ok_idx:
        st.error("Embeddings non disponibili. Controlla la key.")
        st.stop()

    pages_ok = pages.iloc[ok_idx].reset_index(drop=True)
    mat_pages = np.array([emb_pages[i] for i in ok_idx])

    # segment article
    blocks = split_blocks(article_text)
    total_words = word_count(article_text)
    max_links_total = max(1, int(np.ceil(total_words / 1000 * links_per_1000_words)))

    # choose candidate paragraphs (non headings, abbastanza lunghi)
    para_indices = [i for i,b in enumerate(blocks) if b["type"]=="para" and word_count(b["text"]) >= 60]
    if not para_indices:
        st.warning("Non ho trovato paragrafi abbastanza lunghi per inserire link in modo sensato.")
        st.stop()

    # embeddings paragraphs
    para_texts = [blocks[i]["text"][:2000] for i in para_indices]
    st.info("🧠 Calcolo embeddings paragrafi…")
    emb_paras = []
    for i in range(0, len(para_texts), BATCH_SIZE):
        emb_paras.extend(get_embeddings(para_texts[i:i+BATCH_SIZE], client))

    para_ok = [i for i,e in enumerate(emb_paras) if e is not None]
    if not para_ok:
        st.error("Embeddings paragrafi non disponibili.")
        st.stop()

    mat_paras = np.array([emb_paras[i] for i in para_ok])
    sims = cosine_similarity(mat_paras, mat_pages)

    # greedy selection with rules
    used_urls = Counter()
    last_link_wordpos = -10**9
    inserted = []
    out_blocks = [b["text"] for b in blocks]

    # compute word positions per block (for spacing)
    running = 0
    block_wordpos = []
    for b in blocks:
        block_wordpos.append(running)
        running += word_count(b["text"])

    # helper: get best candidates for a paragraph row
    def top_candidates(sim_row, k=8):
        idxs = np.argsort(sim_row)[::-1][:k]
        return [(int(j), float(sim_row[j])) for j in idxs]

    # iterate paragraphs by their order in article (keeps narrative)
    links_made = 0
    for rank_i, para_local in enumerate(para_ok):
        if links_made >= max_links_total:
            break

        block_i = para_indices[para_local]
        para = blocks[block_i]["text"]
        current_wordpos = block_wordpos[block_i]

        # spacing rule
        if (current_wordpos - last_link_wordpos) < min_words_per_link:
            continue

        cand = top_candidates(sims[rank_i], k=10)

        chosen = None
        for j, score in cand:
            row = pages_ok.iloc[j].to_dict()
            url = row["url"]

            # no repeat too much
            if used_urls[url] >= max_same_url:
                continue

            # avoid self-link if the article URL equals candidate? (not available here)
            # confidence threshold: dynamic
            if score < 0.36:
                continue

            chosen = (row, score)
            break

        if not chosen:
            continue

        row, score = chosen
        anchor = choose_anchor(row)
        url = row["url"]

        # apply insertion (safe): end of paragraph
        updated = insert_link_end_of_paragraph(para, anchor, url)

        out_blocks[block_i] = updated
        used_urls[url] += 1
        links_made += 1
        last_link_wordpos = current_wordpos

        inserted.append({
            "block_index": block_i,
            "anchor": anchor,
            "url": url,
            "similarity": round(score, 4),
            "word_pos": current_wordpos
        })

    # compose output text with original structure
    # We lost blank lines; reconstruct with simple joining rules:
    final_lines = []
    for i, b in enumerate(blocks):
        txt = out_blocks[i].strip()
        if not txt:
            continue
        final_lines.append(txt)
        final_lines.append("")  # blank line between blocks
    updated_article = "\n".join(final_lines).strip()

    st.markdown("## ✅ Risultato")

    if mode.startswith("Assistita"):
        # In assistita mostriamo solo proposte e non sovrascriviamo l'articolo
        st.warning("Modalità assistita: qui sotto vedi le proposte. Se vuoi inserimento automatico, usa Safe.")
        st.dataframe(pd.DataFrame(inserted), use_container_width=True)
    else:
        st.markdown("### Articolo con link inseriti")
        st.text_area("Output (copia/incolla)", value=updated_article, height=360)

        st.markdown("### Audit link inseriti")
        audit_df = pd.DataFrame(inserted).sort_values("word_pos")
        st.dataframe(audit_df, use_container_width=True)

        # download
        st.download_button(
            "📥 Scarica output (txt)",
            data=updated_article.encode("utf-8"),
            file_name="article_with_internal_links.txt",
            mime="text/plain"
        )

    # Debug competitor-like view for pages
    with st.expander("🔎 Debug pagine candidate (top 20)", expanded=False):
        debug = pages_ok[["url", "title", "h1", "meta", "clicks", "impressions", "position", "score_gsc"]].head(20)
        st.dataframe(debug, use_container_width=True)
