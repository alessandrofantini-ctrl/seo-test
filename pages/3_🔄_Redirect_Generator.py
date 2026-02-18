import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse, urlunparse
from io import BytesIO
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

# Output
TOPK = 3
LOW_CONF_SCORE = 0.80
LOW_CONF_DELTA = 0.05

# Matching URL-based (generico)
SLUG_JACCARD_STRONG = 0.70   # match quasi certo per slug
SLUG_JACCARD_WEAK = 0.45     # candidato forte, ma non definitivo

# Anti-collasso: limita riuso della stessa pagina target
MAX_REUSE_SAME_TARGET = 8
REUSE_SCORE_GAP_ALLOW = 0.02  # se score2 è vicino a score1, preferisci evitare il collasso

# =========================
# UI
# =========================
st.set_page_config(page_title="AI Redirect Mapper", layout="wide")
st.title("🎯 AI Redirect Mapper")

with st.sidebar:
    st.header("Configurazione")

    openai_api_key = st.text_input("OpenAI key", type="password")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]

    st.markdown("---")
    st.subheader("Lingua default se non rilevabile")
    st.markdown(
        """
Se non si riesce a capire la lingua dall'URL (es. /it/, /en/, /es/),
dal TLD (.it, .es, ecc.) o dal campo “Language”, verrà usato questo valore.
"""
    )
    default_lang_fallback = st.selectbox(
        "Lingua default",
        ["base", "it", "en", "es", "fr", "de"],
        index=0
    )

    st.markdown("---")
    st.subheader("Soglia match stessa lingua")
    st.markdown(
        """
Più alta = più prudente (meno errori, più “nessun match”).
Più bassa = più match (ma rischio maggiore).
"""
    )
    threshold_primary = st.slider("Soglia stessa lingua", 0.0, 1.0, 0.82)

    st.markdown("---")
    st.subheader("Soglia fallback EN")
    st.markdown(
        """
Se non trova match nella lingua originale, prova in inglese.
(utile se consolidate su EN o avete rimosso altre lingue).
"""
    )
    threshold_fallback = st.slider("Soglia fallback EN", 0.0, 1.0, 0.75)

st.caption("Filtri automatici: solo HTML + Status Code 200. Nessun controllo su noindex/indexability.")

# =========================
# Helpers (I/O)
# =========================
def load_file(file):
    try:
        name = (file.name or "").lower()
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file, engine="openpyxl")
        for enc in ["utf-8", "ISO-8859-1", "cp1252"]:
            try:
                file.seek(0)
                return pd.read_csv(file, encoding=enc, low_memory=False)
            except Exception:
                continue
        return None
    except Exception as e:
        st.error(f"Errore lettura file {getattr(file, 'name', '')}: {e}")
        return None

def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Redirects")
    return output.getvalue()

# =========================
# Helpers (URL normalizzazione)
# =========================
def norm_url(u: str) -> str:
    u = ("" if u is None else str(u)).strip()
    if not u:
        return ""
    p = urlparse(u)
    scheme = p.scheme or "https"
    netloc = (p.netloc or "").lower()
    path = p.path or "/"
    # normalizzazione minima: rimuove query/fragment
    return urlunparse((scheme, netloc, path, "", "", ""))

def get_path(u: str) -> str:
    try:
        p = urlparse(u)
        return p.path or "/"
    except Exception:
        return "/"

def is_home(u: str) -> bool:
    return get_path(u) == "/"

def normalize_path_for_match(path: str) -> str:
    """
    Normalizza path per confronti generici:
    - lowercase
    - rimuove estensione .html/.htm/.php/.asp/.aspx
    - underscore -> dash
    - collassa trattini multipli
    - trim slash
    """
    p = (path or "/").lower()
    # togli estensione nel solo ultimo segmento
    p = re.sub(r"(\.(html|htm|php|asp|aspx))$", "", p)
    p = p.replace("_", "-")
    p = re.sub(r"-{2,}", "-", p)
    # evita doppie slash
    p = re.sub(r"/{2,}", "/", p)
    return p

def normalized_slug_tokens(path: str) -> set:
    p = normalize_path_for_match(path).strip("/")
    if not p:
        return set()
    parts = [x for x in p.split("/") if x]
    joined = "-".join(parts)
    toks = [t for t in re.split(r"[^a-z0-9]+", joined) if t]
    # filtra token troppo brevi (rumore)
    toks = [t for t in toks if len(t) > 2]
    # stopword leggere (universali)
    stop = {
        "the","and","for","with","from","this","that",
        "una","uno","per","con","dal","dalla","delle","degli",
        "www","com","html","php"
    }
    return set(t for t in toks if t not in stop)

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

# =========================
# Helpers (lingua)
# =========================
def detect_language(url: str, fallback="base", lang_hint=None) -> str:
    # 1) hint Language (it-IT, es-ES, en)
    if isinstance(lang_hint, str) and lang_hint.strip():
        m = re.match(r"^\s*([a-z]{2})", lang_hint.strip().lower())
        if m:
            code = m.group(1)
            if code in {"it", "en", "es", "fr", "de"}:
                return code

    p = urlparse(url if isinstance(url, str) else str(url))
    domain = (p.netloc or "").lower()
    path = (p.path or "").lower()

    # 2) /it/ /en/ /es-mx/ /es-419/ ecc.
    m = re.search(r"/([a-z]{2})(?:[-_][a-z0-9]{2,4})?(?:/|$)", path)
    if m:
        code = m.group(1)
        if code in {"it", "en", "es", "fr", "de"}:
            return code

    # 3) TLD comuni
    if domain.endswith(".it"): return "it"
    if domain.endswith(".es"): return "es"
    if domain.endswith(".fr"): return "fr"
    if domain.endswith(".de"): return "de"
    if domain.endswith(".co.uk") or domain.endswith(".uk"): return "en"

    return fallback

# =========================
# Helpers (embedding)
# =========================
def make_embedding_text(row: pd.Series) -> str:
    def s(x):
        if x is None:
            return ""
        try:
            if pd.isna(x):
                return ""
        except Exception:
            pass
        return str(x).strip()

    title = s(row.get("title", ""))
    h1 = s(row.get("h1", ""))
    meta = s(row.get("meta", ""))

    parts = []
    if title: parts.append(f"Title: {title}")
    if h1: parts.append(f"H1: {h1}")
    if meta: parts.append(f"Description: {meta}")

    if not parts:
        return "pagina senza segnali testuali"
    return " | ".join(parts)[:6000]

def get_embeddings(texts, client: OpenAI):
    clean = []
    for t in texts:
        t = "" if t is None else str(t)
        clean.append(t.replace("\n", " ").strip()[:8000])

    try:
        res = client.embeddings.create(input=clean, model=EMBED_MODEL)
        embs = [d.embedding for d in res.data]
        if len(embs) != len(clean):
            embs = (embs + [None] * len(clean))[:len(clean)]
        return embs
    except Exception as e:
        st.error(f"OpenAI error (embeddings): {e}")
        return [None] * len(clean)

# =========================
# Parsing Screaming Frog
# =========================
REQUIRED_COLS = ["Address", "Content Type", "Status Code"]

def ensure_cols(df: pd.DataFrame, label: str) -> bool:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"{label}: mancano colonne obbligatorie: {', '.join(missing)}")
        return False
    return True

def filter_html_200(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Content Type"] = out["Content Type"].astype(str)
    out = out[out["Content Type"].str.contains("html", case=False, na=False)]
    out = out[out["Status Code"].astype(str).str.strip() == "200"]
    return out

def build_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    def col(name):
        return df[name] if name in df.columns else pd.Series([""] * len(df))

    clean = pd.DataFrame({
        "url": col("Address").map(norm_url),
        "title": col("Title 1"),
        "h1": col("H1-1"),
        "meta": col("Meta Description 1"),
        "lang_hint": col("Language"),
    })

    for c in ["url", "title", "h1", "meta", "lang_hint"]:
        clean[c] = clean[c].fillna("").astype(str)

    clean = clean[clean["url"] != ""].drop_duplicates(subset=["url"]).reset_index(drop=True)
    return clean

# =========================
# URL-based matching (universale)
# =========================
def build_new_indexes(df_new: pd.DataFrame):
    """
    Costruisce indici per matching generico:
    - home_url per ogni lingua
    - path_normalized -> url (se unico)
    - token signature -> lista url
    """
    home_by_lang = {}
    path_map = {}
    tokens_map = {}

    for _, r in df_new.iterrows():
        u = r["url"]
        lang = r["lang"]
        path = get_path(u)
        npath = normalize_path_for_match(path)

        if is_home(u):
            home_by_lang[lang] = u

        # path match: se duplicato, lo segniamo come non univoco
        if npath in path_map and path_map[npath] != u:
            path_map[npath] = None
        else:
            path_map[npath] = u

        tok = frozenset(normalized_slug_tokens(path))
        if tok:
            tokens_map.setdefault(tok, []).append(u)

    return home_by_lang, path_map, tokens_map

def url_match_candidate(old_url: str, old_lang: str, home_by_lang, path_map):
    """
    Regole universali:
    - home -> home (stessa lingua se disponibile)
    - match esatto path normalizzato
    """
    old_path = get_path(old_url)
    old_npath = normalize_path_for_match(old_path)

    # home -> home
    if old_npath == "/":
        if old_lang in home_by_lang:
            return home_by_lang[old_lang], "Home→Home"
        # fallback: qualsiasi home esistente
        if home_by_lang:
            return list(home_by_lang.values())[0], "Home→Home (fallback)"
        return "", "Home→Home (missing)"

    # exact path normalized match
    if old_npath in path_map and path_map[old_npath]:
        return path_map[old_npath], "Exact path match"

    return "", ""

def slug_similarity_candidates(old_url: str, old_lang: str, df_new: pd.DataFrame):
    """
    Candidati basati su somiglianza slug (Jaccard).
    Filtra per lingua uguale.
    """
    old_toks = normalized_slug_tokens(get_path(old_url))
    if not old_toks:
        return []

    candidates = []
    df_lang = df_new[df_new["lang"] == old_lang]
    for _, r in df_lang.iterrows():
        new_toks = normalized_slug_tokens(get_path(r["url"]))
        if not new_toks:
            continue
        score = jaccard(old_toks, new_toks)
        if score >= SLUG_JACCARD_WEAK:
            candidates.append((r["url"], score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:TOPK]

# =========================
# Upload
# =========================
col1, col2 = st.columns(2)
with col1:
    old_file = st.file_uploader("Vecchio sito (CSV/Excel)", type=["csv", "xlsx"], key="old")
with col2:
    new_file = st.file_uploader("Nuovo sito (CSV/Excel)", type=["csv", "xlsx"], key="new")

if old_file and new_file:
    df_old_raw = load_file(old_file)
    df_new_raw = load_file(new_file)

    if df_old_raw is None or df_new_raw is None:
        st.stop()

    if not ensure_cols(df_old_raw, "Vecchio sito") or not ensure_cols(df_new_raw, "Nuovo sito"):
        st.stop()

    df_old_raw = filter_html_200(df_old_raw)
    df_new_raw = filter_html_200(df_new_raw)

    df_old = build_clean_df(df_old_raw)
    df_new = build_clean_df(df_new_raw)

    if df_old.empty or df_new.empty:
        st.warning("Dopo i filtri (HTML + 200) non ci sono righe sufficienti per fare matching.")
        st.stop()

    df_old["lang"] = df_old.apply(lambda r: detect_language(r["url"], default_lang_fallback, r["lang_hint"]), axis=1)
    df_new["lang"] = df_new.apply(lambda r: detect_language(r["url"], default_lang_fallback, r["lang_hint"]), axis=1)

    st.info(f"📚 Analisi: {len(df_old)} URL sorgente vs {len(df_new)} URL destinazione")

    if st.button("🚀 Avvia matching"):
        if not openai_api_key:
            st.error("Inserisci OpenAI key.")
            st.stop()

        client = OpenAI(api_key=openai_api_key)
        status = st.status("Esecuzione…", expanded=True)

        # Indici per matching generico URL-based
        home_by_lang, path_map, _ = build_new_indexes(df_new)

        # Prepara embedding texts (solo se servono)
        df_old["text"] = df_old.apply(make_embedding_text, axis=1)
        df_new["text"] = df_new.apply(make_embedding_text, axis=1)

        # Embeddings new (una volta)
        status.write("🧠 Generazione embeddings (new)…")
        prog = status.progress(0.0)

        emb_new = []
        for i in range(0, len(df_new), BATCH_SIZE):
            emb_new.extend(get_embeddings(df_new["text"].iloc[i:i+BATCH_SIZE].tolist(), client))
            prog.progress(min(0.40, (i + BATCH_SIZE) / max(1, len(df_new)) * 0.40))

        # Embeddings old
        status.write("🧠 Generazione embeddings (old)…")
        emb_old = []
        for i in range(0, len(df_old), BATCH_SIZE):
            emb_old.extend(get_embeddings(df_old["text"].iloc[i:i+BATCH_SIZE].tolist(), client))
            prog.progress(0.40 + min(0.40, (i + BATCH_SIZE) / max(1, len(df_old)) * 0.40))

        # Filtra None embeddings
        old_ok = [i for i, e in enumerate(emb_old) if e is not None]
        new_ok = [j for j, e in enumerate(emb_new) if e is not None]
        if not old_ok or not new_ok:
            status.update(label="Errore", state="error", expanded=True)
            st.error("Embeddings non disponibili (tutti None). Controlla la key e riprova.")
            st.stop()

        mat_old = np.array([emb_old[i] for i in old_ok])
        mat_new = np.array([emb_new[j] for j in new_ok])

        # map indici locali -> indici df
        old_map = {local_i: df_old.index[i] for local_i, i in enumerate(old_ok)}
        new_map = {local_j: df_new.index[j] for local_j, j in enumerate(new_ok)}

        # Similarity matrix
        status.write("🔍 Calcolo similarità…")
        sims = cosine_similarity(mat_old, mat_new)
        prog.progress(0.90)

        # Indici new per lingua
        new_by_lang = {}
        for local_j, df_j in new_map.items():
            lang = df_new.loc[df_j, "lang"]
            new_by_lang.setdefault(lang, []).append(local_j)
        eng_indices = new_by_lang.get("en", [])

        # Anti-collasso usage count
        target_use = {}

        results = []

        for local_i in range(sims.shape[0]):
            df_i = old_map[local_i]
            old_url = df_old.loc[df_i, "url"]
            old_lang = df_old.loc[df_i, "lang"]

            best_url = ""
            best_score = 0.0
            method = "Nessuno (404)"
            delta = 0.0

            # 1) URL-based: home/exact path
            url_based, url_based_method = url_match_candidate(old_url, old_lang, home_by_lang, path_map)
            if url_based:
                best_url = url_based
                best_score = 1.0
                method = url_based_method
            else:
                # 2) URL-based: slug similarity (jaccard)
                slug_cands = slug_similarity_candidates(old_url, old_lang, df_new)
                if slug_cands and slug_cands[0][1] >= SLUG_JACCARD_STRONG:
                    best_url = slug_cands[0][0]
                    best_score = slug_cands[0][1]
                    method = "Slug match (strong)"
                    if len(slug_cands) > 1:
                        delta = slug_cands[0][1] - slug_cands[1][1]
                else:
                    # 3) AI matching (same lang, strict)
                    idx_list = new_by_lang.get(old_lang, [])
                    candidates = []
                    if idx_list:
                        scores = sims[local_i, idx_list]
                        order = np.argsort(scores)[::-1][:TOPK]
                        candidates = [(idx_list[k], float(scores[k])) for k in order]

                    chosen = None
                    if candidates and candidates[0][1] >= threshold_primary:
                        chosen = ("same", candidates)
                    else:
                        # 4) fallback EN
                        if eng_indices and old_lang != "en":
                            scores_en = sims[local_i, eng_indices]
                            order_en = np.argsort(scores_en)[::-1][:TOPK]
                            cand_en = [(eng_indices[k], float(scores_en[k])) for k in order_en]
                            if cand_en and cand_en[0][1] >= threshold_fallback:
                                chosen = ("en", cand_en)

                    if chosen is not None:
                        pool_name, cand_list = chosen
                        # anti-collasso: se target scelto è già troppo usato, prova il 2°
                        primary_local_j, primary_score = cand_list[0]
                        primary_df_j = new_map[primary_local_j]
                        primary_url = df_new.loc[primary_df_j, "url"]

                        alt_url = None
                        alt_score = None
                        if len(cand_list) > 1:
                            alt_local_j, alt_score = cand_list[1]
                            alt_df_j = new_map[alt_local_j]
                            alt_url = df_new.loc[alt_df_j, "url"]

                        use_count = target_use.get(primary_url, 0)
                        if use_count >= MAX_REUSE_SAME_TARGET and alt_url is not None:
                            # se il secondo è vicino, preferisci alternare
                            if (primary_score - alt_score) <= REUSE_SCORE_GAP_ALLOW:
                                primary_url = alt_url
                                primary_score = alt_score

                        best_url = primary_url
                        best_score = primary_score

                        if len(cand_list) > 1:
                            delta = cand_list[0][1] - cand_list[1][1]

                        method = "AI match" if pool_name == "same" else "AI fallback EN"
                        if best_score < LOW_CONF_SCORE or delta < LOW_CONF_DELTA:
                            method += " (low confidence)"

            if best_url:
                target_use[best_url] = target_use.get(best_url, 0) + 1

            results.append({
                "Old URL": old_url,
                "Old Lang": old_lang,
                "New URL": best_url,
                "Confidence %": round(best_score * 100, 1) if best_score else 0.0,
                "Delta": round(delta, 4),
                "Method": method
            })

        final_df = pd.DataFrame(results)
        status.update(label="Fatto!", state="complete", expanded=False)

        total = len(final_df)
        matched = int((final_df["New URL"] != "").sum())
        unmapped = total - matched

        c1, c2, c3 = st.columns(3)
        c1.metric("Totale URL", total)
        c2.metric("Redirect trovati", matched)
        c3.metric("Senza match", unmapped)

        st.subheader("Anteprima")
        st.dataframe(final_df.head(50), use_container_width=True)

        export_df = final_df[final_df["New URL"] != ""][["Old URL", "New URL"]].copy()
        st.download_button(
            "📥 Scarica Excel (redirect map)",
            data=to_excel_bytes(export_df),
            file_name="redirect_map.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
