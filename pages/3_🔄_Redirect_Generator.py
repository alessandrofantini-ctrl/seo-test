import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from io import BytesIO
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

st.set_page_config(page_title="AI Redirect Mapper - SEO Pro", layout="wide")
st.title("🚀 AI Redirect Mapper — Screaming Frog Edition")

with st.sidebar:
    st.header("⚙️ Configurazione")
    openai_api_key = st.text_input("OpenAI API Key", type="password")

    st.markdown("---")
    st.subheader("🌐 Mappatura Domini → Lingua")
    domain_map_input = st.text_area(
        "dominio:lingua (uno per riga)",
        value="bossong.co.uk:en\nbossong.es:es\nbossong.it:it\nbossong-befestigungssysteme.de:de",
        height=150,
    )

    st.markdown("---")
    st.subheader("🎚️ Soglie di Qualità")
    threshold_good = st.slider("✅ Match Confermato (verde)", 0.0, 1.0, 0.65)
    threshold_low  = st.slider("⚠️ Match Incerto (giallo)",  0.0, 1.0, 0.40)
    st.caption("Sotto la soglia incerta → Fallback Home (rosso)")

    st.markdown("---")
    st.subheader("🔧 Opzioni Avanzate")
    use_path_match   = st.checkbox("Usa Path-Match prima degli embeddings", value=True)
    exclude_mapped   = st.checkbox("Evita URL destinazione duplicati", value=True)
    extra_cols       = st.multiselect(
        "Colonne aggiuntive da includere nell'output",
        ["Status Code", "Inlinks", "Word Count", "Meta Description 1"],
        default=["Status Code", "Inlinks"],
    )

# =========================
# COLONNE SCREAMING FROG
# =========================
# Mapping flessibile: nome interno → possibili nomi SF
SF_COLUMNS = {
    "Address":             ["Address", "URL"],
    "Content Type":        ["Content Type", "Content-Type"],
    "Status Code":         ["Status Code", "Status"],
    "Title 1":             ["Title 1", "Title"],
    "H1-1":                ["H1-1", "H1 1", "H1"],
    "Meta Description 1":  ["Meta Description 1", "Meta Description"],
    "Word Count":          ["Word Count"],
    "Inlinks":             ["Inlinks"],
}

def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rinomina le colonne SF nei nomi interni standard."""
    rename_map = {}
    for internal, candidates in SF_COLUMNS.items():
        for c in candidates:
            if c in df.columns and internal not in df.columns:
                rename_map[c] = internal
                break
    return df.rename(columns=rename_map)

# =========================
# FUNZIONI CORE
# =========================

def get_domain_map(text: str) -> dict:
    d = {}
    for line in text.strip().split("\n"):
        if ":" in line:
            parts = line.split(":", 1)
            d[parts[0].strip().lower().replace("www.", "")] = parts[1].strip().lower()
    return d

def detect_language(url: str, domain_mapping: dict) -> str:
    p = urlparse(url)
    domain = p.netloc.lower().replace("www.", "")
    if domain in domain_mapping:
        return domain_mapping[domain]
    path = p.path.lower()
    m = re.search(r"/([a-z]{2})(?:/|$)", path)
    if m:
        return m.group(1)
    if domain.endswith(".it"):   return "it"
    if domain.endswith(".es"):   return "es"
    if domain.endswith(".de"):   return "de"
    if domain.endswith(".fr"):   return "fr"
    return "en"

def get_seo_content(row: pd.Series) -> str:
    """
    Fingerprint SEO ottimizzato per Screaming Frog.
    Pesa di più path e titolo; aggiunge meta description e word count come segnali secondari.
    """
    url   = str(row.get("Address", ""))
    path  = (urlparse(url).path
             .replace("/", " ")
             .replace("-", " ")
             .replace("_", " ")
             .replace(".html", "")
             .strip())
    title = str(row.get("Title 1", ""))
    h1    = str(row.get("H1-1", ""))
    meta  = str(row.get("Meta Description 1", ""))
    wc    = str(row.get("Word Count", ""))

    # Triplico path e titolo per dargli più peso nel vettore semantico
    return (
        f"PATH: {path} PATH: {path} PATH: {path} | "
        f"TITLE: {title} TITLE: {title} | "
        f"H1: {h1} | "
        f"META: {meta[:200]} | "
        f"WORDS: {wc}"
    )[:8000]

def normalize_path_segments(url: str) -> set:
    """Estrae segmenti di path puliti per il match basato su struttura URL."""
    path = urlparse(url).path.lower()
    parts = {s.replace("-", " ").replace("_", " ")
             for s in path.split("/") if len(s) > 2}
    return parts

def path_similarity_score(old_url: str, new_url: str) -> float:
    """Jaccard similarity sui segmenti di path — gratuita, molto efficace per e-commerce."""
    old_parts = normalize_path_segments(old_url)
    new_parts = normalize_path_segments(new_url)
    if not old_parts and not new_parts:
        return 1.0
    if not old_parts or not new_parts:
        return 0.0
    intersection = old_parts & new_parts
    union        = old_parts | new_parts
    return len(intersection) / len(union)

def get_embeddings_batched(text_list: list, client: OpenAI) -> list:
    """Batch embeddings — minimizza le chiamate API."""
    all_embeddings = []
    for i in range(0, len(text_list), BATCH_SIZE):
        batch    = [t.replace("\n", " ") for t in text_list[i : i + BATCH_SIZE]]
        response = client.embeddings.create(input=batch, model=EMBED_MODEL)
        all_embeddings.extend([d.embedding for d in response.data])
    return all_embeddings

def flag(score: float, thr_good: float, thr_low: float) -> str:
    if score >= thr_good: return "✅ Confermato"
    if score >= thr_low:  return "⚠️ Da verificare"
    return "🔴 Fallback"

def color_row(row):
    """Colorazione condizionale per Excel tramite flag."""
    f = row.get("Flag", "")
    if "✅" in f:   return ["background-color: #d4edda"] * len(row)
    if "⚠️" in f:  return ["background-color: #fff3cd"] * len(row)
    return ["background-color: #f8d7da"] * len(row)

# =========================
# UI — CARICAMENTO FILE
# =========================
col1, col2 = st.columns(2)
with col1:
    old_files = st.file_uploader(
        "📂 Export Screaming Frog — Vecchio Sito",
        accept_multiple_files=True,
        type=["csv", "xlsx"],
    )
with col2:
    new_files = st.file_uploader(
        "📂 Export Screaming Frog — Nuovo Sito",
        accept_multiple_files=True,
        type=["csv", "xlsx"],
    )

def load_sf_export(files) -> pd.DataFrame:
    """Carica e normalizza un export Screaming Frog (CSV o XLSX)."""
    dfs = []
    for f in files:
        if f.name.endswith(".csv"):
            # SF a volte usa encoding UTF-8-BOM
            try:
                df = pd.read_csv(f, encoding="utf-8-sig")
            except Exception:
                df = pd.read_csv(f)
        else:
            df = pd.read_excel(f)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return resolve_columns(combined)

# =========================
# MOTORE PRINCIPALE
# =========================
if old_files and new_files:
    d_mapping = get_domain_map(domain_map_input)

    df_old_raw = load_sf_export(old_files)
    df_new_raw = load_sf_export(new_files)

    # Controllo colonne obbligatorie
    for col in ["Address", "Content Type"]:
        if col not in df_old_raw.columns:
            st.error(f"❌ Colonna '{col}' non trovata nel vecchio sito. Controlla l'export SF.")
            st.stop()
        if col not in df_new_raw.columns:
            st.error(f"❌ Colonna '{col}' non trovata nel nuovo sito. Controlla l'export SF.")
            st.stop()

    # Filtro HTML + solo 200 per il nuovo sito
    df_old = (df_old_raw[df_old_raw["Content Type"]
              .str.contains("html", case=False, na=False)]
              .copy().reset_index(drop=True))

    df_new = (df_new_raw[
                df_new_raw["Content Type"].str.contains("html", case=False, na=False) &
                (df_new_raw.get("Status Code", pd.Series(["200"] * len(df_new_raw)))
                 .astype(str).str.strip() == "200")
              ].copy().reset_index(drop=True))

    # Rilevamento lingua con domain mapping su entrambi i siti
    df_old["lang"] = df_old["Address"].apply(lambda x: detect_language(x, d_mapping))
    df_new["lang"] = df_new["Address"].apply(lambda x: detect_language(x, d_mapping))

    # Anteprima dati
    with st.expander(f"👁️ Anteprima dati ({len(df_old)} vecchie URL · {len(df_new)} nuove URL)"):
        tab1, tab2 = st.tabs(["Vecchio Sito", "Nuovo Sito"])
        with tab1: st.dataframe(df_old.head(20), use_container_width=True)
        with tab2: st.dataframe(df_new.head(20), use_container_width=True)

    st.markdown(f"**Vecchio sito:** {len(df_old)} pagine HTML | **Nuovo sito:** {len(df_new)} pagine 200 HTML")

    if st.button("🚀 GENERA REDIRECT MAP", type="primary"):
        if not openai_api_key:
            st.error("❌ Inserisci la OpenAI API Key nella sidebar.")
            st.stop()

        client = OpenAI(api_key=openai_api_key)
        status = st.status("Elaborazione in corso...", expanded=True)

        # ── STEP 1: Embeddings ──────────────────────────────────────────────
        status.write("🧠 [1/3] Calcolo embeddings (batch mode)...")
        old_texts = df_old.apply(get_seo_content, axis=1).tolist()
        new_texts = df_new.apply(get_seo_content, axis=1).tolist()

        emb_old = np.array(get_embeddings_batched(old_texts, client))
        emb_new = np.array(get_embeddings_batched(new_texts, client))

        # Matrice di similarità completa (indici posizionali — corretti)
        sims = cosine_similarity(emb_old, emb_new)  # shape: (n_old, n_new)

        # ── STEP 2: Home pages per lingua ──────────────────────────────────
        status.write("🌐 [2/3] Identificazione home pages per lingua...")
        home_pages = {}
        for lang in df_new["lang"].unique():
            subset = df_new[df_new["lang"] == lang]
            home_pages[lang] = subset.loc[subset["Address"].str.len().idxmin(), "Address"]

        # ── STEP 3: Matching con logica gerarchica ──────────────────────────
        status.write("🔍 [3/3] Applicazione regole di matching...")

        results          = []
        used_new_urls    = set()  # per evitare duplicati di destinazione

        for i in range(len(df_old)):
            old_row  = df_old.iloc[i]
            old_url  = old_row["Address"]
            old_lang = old_row["lang"]

            # Pool di destinazione: stessa lingua, fallback → inglese
            pool_mask = df_new["lang"] == old_lang
            if not pool_mask.any():
                pool_mask = df_new["lang"] == "en"
            pool_pos_idxs = np.where(pool_mask.values)[0].tolist()  # indici POSIZIONALI

            # Valori di default
            best_url  = home_pages.get(old_lang, home_pages.get("en", df_new.iloc[0]["Address"]))
            best_score = 0.0
            method     = "Fallback: Home di Lingua"

            # ── Regola 0: Home del vecchio sito → Home del nuovo ──────────
            if urlparse(old_url).path.strip("/") == "":
                best_url   = home_pages.get(old_lang, best_url)
                best_score = 1.0
                method     = "Home → Home"

            elif pool_pos_idxs:

                # ── Regola 1: Path-Match (Jaccard sui segmenti URL) ────────
                best_path_score = 0.0
                best_path_idx   = None

                if use_path_match:
                    for pi in pool_pos_idxs:
                        ps = path_similarity_score(old_url, df_new.iloc[pi]["Address"])
                        if ps > best_path_score:
                            best_path_score = ps
                            best_path_idx   = pi

                # Se path-match forte (≥0.5) lo usiamo direttamente
                if use_path_match and best_path_score >= 0.5:
                    best_url   = df_new.iloc[best_path_idx]["Address"]
                    best_score = best_path_score
                    method     = "Path Match (struttura URL)"

                else:
                    # ── Regola 2: Semantic Match (embeddings) ──────────────
                    pool_sims     = sims[i, pool_pos_idxs]
                    best_local    = int(np.argmax(pool_sims))
                    best_sem_score = float(pool_sims[best_local])
                    best_sem_url   = df_new.iloc[pool_pos_idxs[best_local]]["Address"]

                    if best_sem_score >= threshold_low:
                        # Combina path + semantico se entrambi disponibili
                        if use_path_match and best_path_idx is not None and best_path_score > 0:
                            # Media pesata: 40% path, 60% semantico
                            combined_scores = []
                            for pi in pool_pos_idxs:
                                ps  = path_similarity_score(old_url, df_new.iloc[pi]["Address"])
                                sem = float(sims[i, pi])
                                combined_scores.append(0.4 * ps + 0.6 * sem)
                            best_combo = int(np.argmax(combined_scores))
                            best_url   = df_new.iloc[pool_pos_idxs[best_combo]]["Address"]
                            best_score = combined_scores[best_combo]
                            method     = "Match Combinato (Path 40% + Semantico 60%)"
                        else:
                            best_url   = best_sem_url
                            best_score = best_sem_score
                            method     = "Match Semantico"
                    else:
                        # Sotto soglia minima → fallback home
                        best_url   = home_pages.get(old_lang, best_url)
                        best_score = best_sem_score
                        method     = "Fallback: Home di Lingua (score troppo basso)"

            # ── Gestione duplicati di destinazione ────────────────────────
            if exclude_mapped and best_url in used_new_urls and method not in ("Home → Home",):
                # Cerca il secondo miglior match non ancora usato
                if pool_pos_idxs:
                    sorted_idxs = np.argsort(sims[i, pool_pos_idxs])[::-1]
                    for si in sorted_idxs:
                        candidate = df_new.iloc[pool_pos_idxs[si]]["Address"]
                        if candidate not in used_new_urls:
                            best_url   = candidate
                            best_score = float(sims[i, pool_pos_idxs[si]])
                            method    += " (dedup)"
                            break

            used_new_urls.add(best_url)

            # ── Costruzione riga risultato ────────────────────────────────
            row_result = {
                "Old URL":   old_url,
                "New URL":   best_url,
                "Score":     round(best_score * 100, 1),   # numero, non stringa → ordinabile
                "Flag":      flag(best_score, threshold_good, threshold_low),
                "Method":    method,
                "Lingua":    old_lang,
                "Old Title": str(old_row.get("Title 1", "")),
                "Old H1":    str(old_row.get("H1-1", "")),
            }
            # Colonne extra richieste dall'utente
            for ec in extra_cols:
                row_result[f"Old {ec}"] = str(old_row.get(ec, ""))

            results.append(row_result)

        status.update(label="✅ Mappatura completata!", state="complete", expanded=False)

        final_df = pd.DataFrame(results)

        # ── Statistiche riassuntive ────────────────────────────────────────
        st.markdown("### 📊 Riepilogo")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Totale URL", len(final_df))
        c2.metric("✅ Confermati",    (final_df["Flag"] == "✅ Confermato").sum())
        c3.metric("⚠️ Da verificare", (final_df["Flag"] == "⚠️ Da verificare").sum())
        c4.metric("🔴 Fallback",      (final_df["Flag"] == "🔴 Fallback").sum())

        # ── Preview con colori ────────────────────────────────────────────
        st.markdown("### 🗂️ Mappa Redirect")
        styled = final_df.style.apply(color_row, axis=1)
        st.dataframe(styled, use_container_width=True)

        # ── Export Excel con formattazione ────────────────────────────────
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            final_df.to_excel(writer, index=False, sheet_name="Redirect Map")

            wb  = writer.book
            ws  = writer.sheets["Redirect Map"]

            # Formati colore
            fmt_green  = wb.add_format({"bg_color": "#d4edda", "border": 1})
            fmt_yellow = wb.add_format({"bg_color": "#fff3cd", "border": 1})
            fmt_red    = wb.add_format({"bg_color": "#f8d7da", "border": 1})
            fmt_header = wb.add_format({"bold": True, "bg_color": "#343a40", "font_color": "white", "border": 1})

            # Header
            for ci, col_name in enumerate(final_df.columns):
                ws.write(0, ci, col_name, fmt_header)

            # Righe colorate
            flag_col = list(final_df.columns).index("Flag")
            for ri, row_vals in enumerate(final_df.itertuples(index=False), start=1):
                f = row_vals[flag_col]
                fmt = fmt_green if "✅" in f else (fmt_yellow if "⚠️" in f else fmt_red)
                for ci, val in enumerate(row_vals):
                    ws.write(ri, ci, val, fmt)

            # Larghezze colonne
            ws.set_column(0, 1, 60)   # Old/New URL
            ws.set_column(2, 2, 10)   # Score
            ws.set_column(3, 5, 20)   # Flag, Method, Lingua
            ws.set_column(6, 6, 40)   # Old Title
            ws.freeze_panes(1, 0)

            # Foglio riepilogo
            summary_data = {
                "Categoria":    ["Totale URL", "✅ Confermati", "⚠️ Da verificare", "🔴 Fallback"],
                "Conteggio":    [
                    len(final_df),
                    (final_df["Flag"] == "✅ Confermato").sum(),
                    (final_df["Flag"] == "⚠️ Da verificare").sum(),
                    (final_df["Flag"] == "🔴 Fallback").sum(),
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name="Riepilogo")

        st.download_button(
            label="📥 Scarica Excel",
            data=output.getvalue(),
            file_name="redirect_map.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
