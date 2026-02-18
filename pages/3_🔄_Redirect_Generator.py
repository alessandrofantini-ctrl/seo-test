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

TOPK = 3                 # esportiamo sempre top 3 candidati (qualità + controllo)
LOW_CONF_SCORE = 0.80    # sotto questo: match debole
LOW_CONF_DELTA = 0.05    # se score1-score2 è basso: match instabile

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
Se l’URL non contiene una cartella lingua (/it/, /en/, /es/), non ha TLD dedicato (.it, .es, ecc.)
e la colonna “Language” non aiuta, il sistema userà questa lingua come riferimento.

- sito principalmente italiano in root → **it**
- sito principalmente inglese in root → **en**
- scenario misto / non vuoi forzare → **base**
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
Livello minimo di similarità per creare un redirect nella stessa lingua.

- più alta = più sicuro ma più “nessun match”
- più bassa = più match ma più rischio errori

Valore tipico: **0.80 – 0.85**
"""
    )
    threshold_primary = st.slider("Soglia stessa lingua", 0.0, 1.0, 0.82)

    st.markdown("---")
    st.subheader("Soglia fallback EN")
    st.markdown(
        """
Se non si trova un match nella lingua originale, prova a cercare in inglese.

Usalo se stai consolidando su EN o se alcune lingue sono state rimosse.
Di solito è un po’ più basso della soglia principale (es. **0.75 – 0.78**).
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
# Helpers (URL + lingua)
# =========================
def norm_url(u: str) -> str:
    u = ("" if u is None else str(u)).strip()
    if not u:
        return ""
    p = urlparse(u)
    scheme = p.scheme or "https"
    netloc = (p.netloc or "").lower()
    path = p.path or "/"
    # normalizzazione minima: niente query/fragment
    return urlunparse((scheme, netloc, path, "", "", ""))

def detect_language(url: str, fallback="base", lang_hint=None) -> str:
    # 1) hint dalla colonna "Language" (es. it-IT, es-ES, en)
    if isinstance(lang_hint, str) and lang_hint.strip():
        m = re.match(r"^\s*([a-z]{2})", lang_hint.strip().lower())
        if m:
            code = m.group(1)
            if code in {"it", "en", "es", "fr", "de"}:
                return code

    p = urlparse(url if isinstance(url, str) else str(url))
    domain = (p.netloc or "").lower()
    path = (p.path or "").lower()

    # 2) cartella lingua: /it/ /es-mx/ /es-419/ ecc.
    m = re.search(r"/([a-z]{2})(?:[-_][a-z0-9]{2,4})?(?:/|$)", path)
    if m:
        code = m.group(1)
        if code in {"it", "en", "es", "fr", "de"}:
            return code

    # 3) TLD
    if domain.endswith(".it"): return "it"
    if domain.endswith(".es"): return "es"
    if domain.endswith(".fr"): return "fr"
    if domain.endswith(".de"): return "de"
    if domain.endswith(".co.uk") or domain.endswith(".uk"): return "en"

    # 4) fallback
    return fallback

# =========================
# Helpers (embedding)
# =========================
def make_embedding_text(row: pd.Series) -> str:
    # robusto: converte NaN/float in stringa vuota
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
    # ritorna sempre una lista lunga quanto texts
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
# Helpers (parsing Screaming Frog)
# =========================
REQUIRED_COLS = ["Address", "Content Type", "Status Code"]

def ensure_cols(df: pd.DataFrame, label: str) -> bool:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"{label}: mancano colonne obbligatorie: {', '.join(missing)}")
        return False
    return True

def filter_html_200(df: pd.DataFrame) -> pd.DataFrame:
    # solo HTML e status 200
    out = df.copy()
    out["Content Type"] = out["Content Type"].astype(str)
    out = out[out["Content Type"].str.contains("html", case=False, na=False)]
    # status code può essere int o string
    sc = out["Status Code"]
    out = out[sc.astype(str).str.strip() == "200"]
    return out

def build_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # usa i campi che hai indicato; se non esistono, li crea vuoti
    def col(name):
        return df[name] if name in df.columns else pd.Series([""] * len(df))

    clean = pd.DataFrame({
        "url": col("Address").map(norm_url),
        "title": col("Title 1"),
        "h1": col("H1-1"),
        "meta": col("Meta Description 1"),
        "lang_hint": col("Language"),
    })

    # pulizia base per evitare NaN
    for c in ["url", "title", "h1", "meta", "lang_hint"]:
        clean[c] = clean[c].fillna("").astype(str)

    clean = clean[clean["url"] != ""].drop_duplicates(subset=["url"]).reset_index(drop=True)
    return clean

# =========================
# Matching
# =========================
def topk_candidates(scores: np.ndarray, idx_list: list[int], k: int):
    if len(idx_list) == 0:
        return []
    order = np.argsort(scores)[::-1][:k]
    return [(idx_list[i], float(scores[i])) for i in order]

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

    # filtri fissi: HTML + 200
    df_old_raw = filter_html_200(df_old_raw)
    df_new_raw = filter_html_200(df_new_raw)

    df_old = build_clean_df(df_old_raw)
    df_new = build_clean_df(df_new_raw)

    if df_old.empty or df_new.empty:
        st.warning("Dopo i filtri (HTML + 200) non ci sono righe sufficienti per fare matching.")
        st.stop()

    # lingue
    df_old["lang"] = df_old.apply(lambda r: detect_language(r["url"], default_lang_fallback, r["lang_hint"]), axis=1)
    df_new["lang"] = df_new.apply(lambda r: detect_language(r["url"], default_lang_fallback, r["lang_hint"]), axis=1)

    st.info(f"📚 Analisi: {len(df_old)} URL sorgente vs {len(df_new)} URL destinazione")

    if st.button("🚀 Avvia matching"):
        if not openai_api_key:
            st.error("Inserisci OpenAI key.")
            st.stop()

        client = OpenAI(api_key=openai_api_key)

        status = st.status("Esecuzione…", expanded=True)

        # testi per embedding (robusti)
        df_old["text"] = df_old.apply(make_embedding_text, axis=1)
        df_new["text"] = df_new.apply(make_embedding_text, axis=1)

        status.write("🧠 Generazione embeddings…")
        prog = status.progress(0.0)

        # embeddings new
        emb_new = []
        for i in range(0, len(df_new), BATCH_SIZE):
            emb_new.extend(get_embeddings(df_new["text"].iloc[i:i+BATCH_SIZE].tolist(), client))
            prog.progress(min(0.45, (i + BATCH_SIZE) / max(1, len(df_new)) * 0.45))

        # embeddings old
        emb_old = []
        for i in range(0, len(df_old), BATCH_SIZE):
            emb_old.extend(get_embeddings(df_old["text"].iloc[i:i+BATCH_SIZE].tolist(), client))
            prog.progress(0.45 + min(0.45, (i + BATCH_SIZE) / max(1, len(df_old)) * 0.45))

        # rimuovi righe con embedding None (rare, ma meglio sicuri)
        old_ok = [i for i, e in enumerate(emb_old) if e is not None]
        new_ok = [j for j, e in enumerate(emb_new) if e is not None]

        if not old_ok or not new_ok:
            status.update(label="Errore", state="error", expanded=True)
            st.error("Impossibile generare embeddings (tutti None). Controlla la key e riprova.")
            st.stop()

        mat_old = np.array([emb_old[i] for i in old_ok])
        mat_new = np.array([emb_new[j] for j in new_ok])

        # mappa indici locali -> indici df
        old_map = {local_i: df_old.index[i] for local_i, i in enumerate(old_ok)}
        new_map = {local_j: df_new.index[j] for local_j, j in enumerate(new_ok)}

        status.write("🔍 Similarità e matching…")
        sims = cosine_similarity(mat_old, mat_new)
        prog.progress(0.95)

        # pre-calcola indici new per lingua
        new_by_lang = {}
        for local_j, df_j in new_map.items():
            lang = df_new.loc[df_j, "lang"]
            new_by_lang.setdefault(lang, []).append(local_j)

        eng_indices = new_by_lang.get("en", [])

        results = []
        for local_i in range(sims.shape[0]):
            df_i = old_map[local_i]
            old_url = df_old.loc[df_i, "url"]
            old_lang = df_old.loc[df_i, "lang"]

            best_url = ""
            best_score = 0.0
            method = "Nessuno (404)"

            # topk stessa lingua
            idx_list = new_by_lang.get(old_lang, [])
            candidates_same = []
            if idx_list:
                scores = sims[local_i, idx_list]
                candidates_same = topk_candidates(scores, idx_list, TOPK)

            # fallback EN solo se:
            # - non c'è match sopra soglia nella stessa lingua
            # - e abbiamo pagine EN
            candidates_en = []
            if eng_indices and old_lang != "en":
                scores_en = sims[local_i, eng_indices]
                candidates_en = topk_candidates(scores_en, eng_indices, TOPK)

            # valuta miglior match: prima stessa lingua, poi fallback EN
            chosen_pool = "same"
            chosen = candidates_same[0] if candidates_same else None
            if chosen:
                chosen_score = chosen[1]
                if chosen_score < threshold_primary:
                    chosen = None

            if chosen is None and candidates_en:
                chosen_pool = "en"
                chosen = candidates_en[0]
                if chosen[1] < threshold_fallback:
                    chosen = None

            # compila scelta
            delta = 0.0
            if chosen is not None:
                local_j = chosen[0]
                df_j = new_map[local_j]
                best_url = df_new.loc[df_j, "url"]
                best_score = chosen[1]

                # delta calcolato sulla pool scelta
                if chosen_pool == "same" and len(candidates_same) > 1:
                    delta = candidates_same[0][1] - candidates_same[1][1]
                elif chosen_pool == "en" and len(candidates_en) > 1:
                    delta = candidates_en[0][1] - candidates_en[1][1]
                else:
                    delta = 0.0

                # etichetta metodo
                if chosen_pool == "same":
                    method = f"AI match ({old_lang})"
                else:
                    method = "AI fallback EN"

                if best_score < LOW_CONF_SCORE or delta < LOW_CONF_DELTA:
                    method += " (low confidence)"

            # salva anche top3 URL/score stessa lingua per revisione
            top_same_urls = []
            top_same_scores = []
            for cand in (candidates_same or [])[:TOPK]:
                df_j = new_map[cand[0]]
                top_same_urls.append(df_new.loc[df_j, "url"])
                top_same_scores.append(round(cand[1], 4))

            results.append({
                "Old URL": old_url,
                "Old Lang": old_lang,
                "New URL": best_url,
                "Confidence %": round(best_score * 100, 1),
                "Delta": round(delta, 4),
                "Method": method,
                "Top1 same-lang URL": top_same_urls[0] if len(top_same_urls) > 0 else "",
                "Top1 same-lang score": top_same_scores[0] if len(top_same_scores) > 0 else "",
                "Top2 same-lang URL": top_same_urls[1] if len(top_same_urls) > 1 else "",
                "Top2 same-lang score": top_same_scores[1] if len(top_same_scores) > 1 else "",
                "Top3 same-lang URL": top_same_urls[2] if len(top_same_urls) > 2 else "",
                "Top3 same-lang score": top_same_scores[2] if len(top_same_scores) > 2 else "",
            })

        final_df = pd.DataFrame(results)
        status.update(label="Fatto!", state="complete", expanded=False)

        # metriche
        total = len(final_df)
        matched = int((final_df["New URL"] != "").sum())
        unmapped = total - matched

        c1, c2, c3 = st.columns(3)
        c1.metric("Totale URL", total)
        c2.metric("Redirect trovati", matched)
        c3.metric("Senza match", unmapped)

        st.subheader("Anteprima")
        st.dataframe(final_df.head(50), use_container_width=True)

        # export: solo 2 colonne per implementazione redirect
        export_df = final_df[final_df["New URL"] != ""][["Old URL", "New URL"]].copy()

        st.download_button(
            "📥 Scarica Excel (redirect map)",
            data=to_excel_bytes(export_df),
            file_name="redirect_map.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
