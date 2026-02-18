import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse, urlunparse
from io import BytesIO
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG BASE ----------------
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
TOPK = 3
LOW_CONF_SCORE = 0.80
LOW_CONF_DELTA = 0.05

# ---------------- UI ----------------
st.set_page_config(page_title="AI Redirect Mapper", layout="wide")
st.title("🎯 AI Redirect Mapper")

with st.sidebar:
    st.header("Configurazione")
    openai_api_key = st.text_input("OpenAI key", type="password")

    st.markdown("---")
    default_lang_fallback = st.selectbox(
        "Lingua default se non rilevabile",
        ["base", "it", "en", "es", "fr", "de"],
        index=0
    )

    st.markdown("---")
    threshold_primary = st.slider("Soglia match stessa lingua", 0.0, 1.0, 0.82)
    threshold_fallback = st.slider("Soglia fallback EN", 0.0, 1.0, 0.78)

# ---------------- Helpers ----------------

def load_file(file):
    if file.name.endswith(("xlsx", "xls")):
        return pd.read_excel(file, engine="openpyxl")
    for enc in ["utf-8", "ISO-8859-1", "cp1252"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc, low_memory=False)
        except:
            continue
    return None

def norm_url(u):
    u = str(u).strip()
    if not u:
        return ""
    p = urlparse(u)
    scheme = p.scheme or "https"
    netloc = (p.netloc or "").lower()
    path = p.path or "/"
    return urlunparse((scheme, netloc, path, "", "", ""))

def detect_language(url, fallback="base", lang_hint=None):
    if isinstance(lang_hint, str) and lang_hint:
        m = re.match(r"^([a-z]{2})", lang_hint.lower())
        if m:
            return m.group(1)

    p = urlparse(url)
    domain = p.netloc.lower()
    path = p.path.lower()

    m = re.search(r"/([a-z]{2})(?:[-_][a-z0-9]{2,4})?(?:/|$)", path)
    if m:
        return m.group(1)

    if domain.endswith(".it"): return "it"
    if domain.endswith(".es"): return "es"
    if domain.endswith(".fr"): return "fr"
    if domain.endswith(".de"): return "de"
    if domain.endswith(".uk"): return "en"

    return fallback

def make_embedding_text(row):
    parts = []
    if row["title"]: parts.append(row["title"])
    if row["h1"]: parts.append(row["h1"])
    if row["meta"]: parts.append(row["meta"])
    return " | ".join(parts)[:6000]

def get_embeddings(texts, client):
    clean = [(t or "")[:8000].replace("\n", " ") for t in texts]
    try:
        res = client.embeddings.create(input=clean, model=EMBED_MODEL)
        return [d.embedding for d in res.data]
    except:
        return [None] * len(texts)

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# ---------------- Upload ----------------

col1, col2 = st.columns(2)

with col1:
    old_file = st.file_uploader("Vecchio sito (CSV/Excel)", type=["csv","xlsx"], key="old")

with col2:
    new_file = st.file_uploader("Nuovo sito (CSV/Excel)", type=["csv","xlsx"], key="new")

if old_file and new_file:
    df_old = load_file(old_file)
    df_new = load_file(new_file)

    # SOLO HTML + 200
    df_old = df_old[
        (df_old["Content Type"].str.contains("html", case=False, na=False)) &
        (df_old["Status Code"] == 200)
    ]

    df_new = df_new[
        (df_new["Content Type"].str.contains("html", case=False, na=False)) &
        (df_new["Status Code"] == 200)
    ]

    df_old_clean = pd.DataFrame({
        "url": df_old["Address"].map(norm_url),
        "title": df_old.get("Title 1", ""),
        "h1": df_old.get("H1-1", ""),
        "meta": df_old.get("Meta Description 1", ""),
        "lang_hint": df_old.get("Language", "")
    })

    df_new_clean = pd.DataFrame({
        "url": df_new["Address"].map(norm_url),
        "title": df_new.get("Title 1", ""),
        "h1": df_new.get("H1-1", ""),
        "meta": df_new.get("Meta Description 1", ""),
        "lang_hint": df_new.get("Language", "")
    })

    df_old_clean["lang"] = df_old_clean.apply(
        lambda r: detect_language(r["url"], default_lang_fallback, r["lang_hint"]),
        axis=1
    )

    df_new_clean["lang"] = df_new_clean.apply(
        lambda r: detect_language(r["url"], default_lang_fallback, r["lang_hint"]),
        axis=1
    )

    if st.button("Avvia matching"):
        client = OpenAI(api_key=openai_api_key)

        df_old_clean["text"] = df_old_clean.apply(make_embedding_text, axis=1)
        df_new_clean["text"] = df_new_clean.apply(make_embedding_text, axis=1)

        emb_old = get_embeddings(df_old_clean["text"].tolist(), client)
        emb_new = get_embeddings(df_new_clean["text"].tolist(), client)

        sims = cosine_similarity(np.array(emb_old), np.array(emb_new))

        results = []

        for i, row in df_old_clean.iterrows():
            old_lang = row["lang"]
            indices = df_new_clean.index[df_new_clean["lang"] == old_lang].tolist()

            if not indices:
                continue

            scores = sims[i, indices]
            sorted_idx = np.argsort(scores)[::-1][:TOPK]

            best_score = scores[sorted_idx[0]]
            best_url = df_new_clean.loc[indices[sorted_idx[0]], "url"]

            delta = 0
            if len(sorted_idx) > 1:
                delta = best_score - scores[sorted_idx[1]]

            method = "AI"
            if best_score < LOW_CONF_SCORE or delta < LOW_CONF_DELTA:
                method = "AI (Low confidence)"

            if best_score < threshold_primary:
                best_url = ""
                method = "Nessun match"

            results.append({
                "Old URL": row["url"],
                "New URL": best_url,
                "Confidence %": round(best_score * 100, 1),
                "Delta": round(delta, 4),
                "Method": method
            })

        final_df = pd.DataFrame(results)
        st.dataframe(final_df.head(50))

        st.download_button(
            "Scarica Excel",
            to_excel(final_df[final_df["New URL"] != ""][["Old URL","New URL"]]),
            "redirect_map.xlsx"
        )
