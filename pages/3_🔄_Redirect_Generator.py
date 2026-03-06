import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
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
    st.subheader("🌍 Lingua Root Nuovo Sito")
    st.caption("Se il nuovo sito non ha sottocartella lingua alla root (es: /it/ assente ma il contenuto è italiano), specifica qui la lingua della root.")
    root_lang = st.selectbox(
        "Lingua della root del nuovo sito",
        options=["it", "en", "es", "de", "fr", "pt"],
        index=0,
    )

    st.markdown("---")
    st.subheader("🔀 Forza Domini → Lingua Destinazione")
    st.caption("I domini qui elencati verranno sempre matchati nel pool della lingua indicata, ignorando la loro lingua originale.")
    forced_domain_input = st.text_area(
        "dominio:lingua_forzata (uno per riga)",
        value="bossong-befestigungssysteme.de:en",
        height=120,
    )

    st.markdown("---")
    st.subheader("🎯 Vincola Dominio Vecchio → Dominio Nuovo")
    st.caption(
        "Specifica su quale nuovo dominio deve puntare un vecchio dominio (o una sua lingua specifica). "
        "Formato: `vecchio_dominio:nuovo_dominio` oppure `vecchio_dominio/lingua:nuovo_dominio` per un vincolo per-lingua. "
        "Esempio: `ghidini-gb.it:ghidinigben.coridemo.com` oppure `ghidini-gb.it/fr:ghidinigbfr.coridemo.com`"
    )
    domain_target_input = st.text_area(
        "vecchio_dominio[:lingua]:nuovo_dominio (uno per riga)",
        value="",
        placeholder="ghidini-gb.it:ghidinigben.coridemo.com\nghidini-gb.it/fr:ghidinigbfr.coridemo.com",
        height=120,
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
    validate_urls    = st.checkbox("Valida URL di destinazione (anomalie)", value=False)
    extra_cols       = st.multiselect(
        "Colonne aggiuntive da includere nell'output",
        ["Status Code", "Inlinks", "Word Count", "Meta Description 1"],
        default=["Status Code", "Inlinks"],
    )

# =========================
# COLONNE SCREAMING FROG
# =========================
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

VALID_LANGS = {"it", "es", "de", "fr", "en", "pt", "nl", "pl", "ru", "zh", "ja", "ar"}

def detect_language(url: str, domain_mapping: dict, root_lang: str = "en") -> str:
    p = urlparse(url)
    domain = p.netloc.lower().replace("www.", "")
    parts = [s for s in p.path.lower().split("/") if s]

    # 1a. Chiave specifica dominio/prima-lingua-nel-path (priorità massima)
    if parts and len(parts[0]) == 2 and parts[0] in VALID_LANGS:
        domain_lang_key = f"{domain}/{parts[0]}"
        if domain_lang_key in domain_mapping:
            return domain_mapping[domain_lang_key]

    # 1b. Domain mapping generico
    if domain in domain_mapping:
        return domain_mapping[domain]

    # 2. Sottocartella lingua nel path
    if parts and len(parts[0]) == 2 and parts[0] in VALID_LANGS:
        return parts[0]

    # 3. TLD multi-parte
    if domain.endswith(".co.uk"):  return "en"
    if domain.endswith(".co.nz"):  return "en"
    if domain.endswith(".co.au"):  return "en"

    # 4. TLD semplice
    if domain.endswith(".it"):  return "it"
    if domain.endswith(".es"):  return "es"
    if domain.endswith(".de"):  return "de"
    if domain.endswith(".fr"):  return "fr"
    if domain.endswith(".pt"):  return "pt"
    if domain.endswith(".nl"):  return "nl"

    return root_lang

def get_seo_content(row: pd.Series) -> str:
    url   = str(row.get("Address", ""))
    path  = (urlparse(url).path
             .replace("/", " ").replace("-", " ")
             .replace("_", " ").replace(".html", "").strip())
    title = str(row.get("Title 1", ""))
    h1    = str(row.get("H1-1", ""))
    meta  = str(row.get("Meta Description 1", ""))
    wc    = str(row.get("Word Count", ""))
    return (
        f"PATH: {path} PATH: {path} PATH: {path} | "
        f"TITLE: {title} TITLE: {title} | "
        f"H1: {h1} | META: {meta[:200]} | WORDS: {wc}"
    )[:8000]

def _tok_similar(a: str, b: str) -> bool:
    """True se due token slug sono varianti della stessa parola (plurali, forme flesse).
    Es: 'contattore' ≈ 'contattori', 'prodotto' ≈ 'prodotti'.
    Algoritmo: prefisso comune ≥ 80% del token più corto."""
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    plen = next((i for i, (x, y) in enumerate(zip(short, long)) if x != y), len(short))
    return plen >= len(short) * 0.80

def path_depth(url: str) -> int:
    """Numero di segmenti non vuoti nel path. Indica categoria (bassa) vs prodotto (alta)."""
    return len([s for s in urlparse(url).path.split("/") if s])

def path_similarity_score(old_url: str, new_url: str) -> float:
    """SEO-grade path similarity con logica gerarchica:
    1. Slug finale identico → 1.0
    2. Slug corto contenuto nel più lungo (o viceversa) → 0.90
    3. Soft token overlap slug (copre plurali/forme flesse) → fino a 0.95
    4. Jaccard sull'intero path come segnale debole (max 0.70)
    """
    p_old = urlparse(old_url).path
    p_new = urlparse(new_url).path

    old_slug = Path(p_old).name.lower().replace("-", " ").replace("_", " ")
    new_slug = Path(p_new).name.lower().replace("-", " ").replace("_", " ")

    # Step 1: slug identico → match diretto
    if old_slug and new_slug and old_slug == new_slug:
        return 1.0

    # Step 2: containment (es: "ambria" ⊂ "ambria 17")
    if old_slug and new_slug and (old_slug in new_slug or new_slug in old_slug):
        return 0.90

    # Step 3: soft token overlap slug (es: contattore ≈ contattori, prodotto ≈ prodotti)
    old_toks = [t for t in old_slug.split() if len(t) > 2]
    new_toks = [t for t in new_slug.split() if len(t) > 2]
    if old_toks and new_toks:
        matched = sum(1 for ot in old_toks if any(_tok_similar(ot, nt) for nt in new_toks))
        # Jaccard soft: token simili contano come 1 nel denominatore invece di 2
        denom = len(old_toks) + len(new_toks) - matched
        soft_jac = matched / denom if denom else 0.0
        if soft_jac > 0.6:
            return soft_jac * 0.95

    # Step 4: Jaccard sull'intero path (segnale debole, max 0.70)
    old_path_toks = {t for seg in p_old.split("/")
                     for t in re.findall(r"[a-z0-9]{3,}", seg.lower().replace("-", " "))}
    new_path_toks = {t for seg in p_new.split("/")
                     for t in re.findall(r"[a-z0-9]{3,}", seg.lower().replace("-", " "))}
    if not old_path_toks or not new_path_toks:
        return 0.0
    return len(old_path_toks & new_path_toks) / len(old_path_toks | new_path_toks) * 0.70

def get_forced_lang(url: str, forced_mapping: dict) -> str | None:
    domain = urlparse(url).netloc.lower().replace("www.", "")
    return forced_mapping.get(domain, None)

def get_target_domain(url: str, lang: str, domain_target_mapping: dict) -> str | None:
    domain = urlparse(url).netloc.lower().replace("www.", "")
    return domain_target_mapping.get(f"{domain}/{lang}") or domain_target_mapping.get(domain, None)

def get_embeddings_batched(text_list: list, client: OpenAI) -> list:
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
    f = row.get("Flag", "")
    if "✅" in f:  return ["background-color: #d4edda"] * len(row)
    if "⚠️" in f: return ["background-color: #fff3cd"] * len(row)
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
    dfs = []
    for f in files:
        if f.name.endswith(".csv"):
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
    d_mapping             = get_domain_map(domain_map_input)
    forced_mapping        = get_domain_map(forced_domain_input)
    domain_target_mapping = get_domain_map(domain_target_input)

    df_old_raw = load_sf_export(old_files)
    df_new_raw = load_sf_export(new_files)

    for col in ["Address", "Content Type"]:
        if col not in df_old_raw.columns:
            st.error(f"❌ Colonna '{col}' non trovata nel vecchio sito.")
            st.stop()
        if col not in df_new_raw.columns:
            st.error(f"❌ Colonna '{col}' non trovata nel nuovo sito.")
            st.stop()

    df_old = (df_old_raw[df_old_raw["Content Type"]
              .str.contains("html", case=False, na=False)]
              .copy().reset_index(drop=True))

    df_new = (df_new_raw[
                df_new_raw["Content Type"].str.contains("html", case=False, na=False) &
                (df_new_raw.get("Status Code", pd.Series(["200"] * len(df_new_raw)))
                 .astype(str).str.strip() == "200")
              ].copy().reset_index(drop=True))

    df_old["lang"]  = df_old["Address"].apply(lambda x: detect_language(x, d_mapping))
    df_new["lang"]  = df_new["Address"].apply(lambda x: detect_language(x, d_mapping, root_lang=root_lang))
    df_old["depth"] = df_old["Address"].apply(path_depth)
    df_new["depth"] = df_new["Address"].apply(path_depth)

    with st.expander("🔍 Debug: Lingue rilevate"):
        tab_old, tab_new, tab_map = st.tabs(["Vecchio Sito", "Nuovo Sito", "Domain Map"])
        with tab_old:
            st.dataframe(df_old[["Address", "lang", "depth"]].head(50), use_container_width=True)
        with tab_new:
            st.dataframe(df_new[["Address", "lang", "depth"]].head(50), use_container_width=True)
        with tab_map:
            st.dataframe(
                pd.DataFrame(list(d_mapping.items()), columns=["Dominio", "Lingua"]),
                use_container_width=True
            )

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

        # ── STEP 1: Embeddings ──────────────────────────────────────────
        status.write("🧠 [1/3] Calcolo embeddings (batch mode)...")
        old_texts = df_old.apply(get_seo_content, axis=1).tolist()
        new_texts = df_new.apply(get_seo_content, axis=1).tolist()
        emb_old = np.array(get_embeddings_batched(old_texts, client))
        emb_new = np.array(get_embeddings_batched(new_texts, client))
        sims = cosine_similarity(emb_old, emb_new)  # (n_old, n_new)

        # ── STEP 2: Home pages per lingua ───────────────────────────────
        status.write("🌐 [2/3] Identificazione home pages per lingua...")
        home_pages = {}
        for lang in df_new["lang"].unique():
            subset = df_new[df_new["lang"] == lang]
            home_pages[lang] = subset.loc[subset["Address"].str.len().idxmin(), "Address"]

        # ── STEP 3: Matching SEO-grade ──────────────────────────────────
        status.write("🔍 [3/3] Applicazione regole di matching...")
        results       = []
        used_new_urls = set()

        for i in range(len(df_old)):
            old_row   = df_old.iloc[i]
            old_url   = old_row["Address"]
            old_lang  = old_row["lang"]
            old_depth = int(old_row["depth"])

            forced_lang  = get_forced_lang(old_url, forced_mapping)
            match_lang   = forced_lang if forced_lang else old_lang
            target_domain = get_target_domain(old_url, match_lang, domain_target_mapping)

            # ── Build pool ──────────────────────────────────────────────
            pool_mask = df_new["lang"] == match_lang
            if not pool_mask.any():
                pool_mask = df_new["lang"] == "en"
            if not pool_mask.any():
                pool_mask = pd.Series([True] * len(df_new))

            if target_domain:
                domain_mask = df_new["Address"].apply(
                    lambda u: urlparse(u).netloc.lower().replace("www.", "") == target_domain
                )
                constrained = pool_mask & domain_mask
                pool_mask = constrained if constrained.any() else (domain_mask if domain_mask.any() else pool_mask)

            pool_pos_idxs = np.where(pool_mask.values)[0].tolist()

            # Fallback home per questo URL
            if target_domain and pool_pos_idxs:
                fallback_home = df_new.iloc[pool_pos_idxs[0]]["Address"]
            else:
                fallback_home = home_pages.get(match_lang, home_pages.get("en", df_new.iloc[0]["Address"]))

            best_url    = fallback_home
            best_score  = 0.0
            score_path  = 0.0
            score_sem   = 0.0
            method      = f"Fallback: nessun candidato nella lingua {match_lang}"
            is_duplicate = False

            # ── Regola 0: Homepage → Homepage ──────────────────────────
            if urlparse(old_url).path.strip("/") == "":
                best_url   = fallback_home
                best_score = 1.0
                score_path = 1.0
                score_sem  = 1.0
                method     = "Home → Home"

            elif pool_pos_idxs:
                # ── Calcolo path scores per tutti i candidati ──────────
                path_scores  = []
                pool_sims_arr = []

                for pi in pool_pos_idxs:
                    new_url_cand = df_new.iloc[pi]["Address"]
                    ps = path_similarity_score(old_url, new_url_cand) if use_path_match else 0.0

                    # Penalità profondità: categoria vs prodotto
                    # Un SEO non manda mai un prodotto su una categoria e viceversa
                    new_d = int(df_new.iloc[pi]["depth"])
                    if (old_depth >= 3 and new_d <= 2) or (old_depth <= 2 and new_d >= 3):
                        ps *= 0.70

                    path_scores.append(ps)
                    pool_sims_arr.append(float(sims[i, pi]))

                best_path_score = max(path_scores) if path_scores else 0.0

                # ── Pesi adattativi in base alla forza del path match ──
                # Path molto forte (slug identico / containment) → path domina
                # Path medio → bilancia con semantico
                # Path debole → lascia guidare il semantico
                if best_path_score >= 0.80:
                    w_p, w_s = 1.00, 0.00   # path sufficiente da solo
                elif best_path_score >= 0.50:
                    w_p, w_s = 0.70, 0.30
                else:
                    w_p, w_s = 0.30, 0.70

                combined_scores = [
                    w_p * path_scores[j] + w_s * pool_sims_arr[j]
                    for j in range(len(pool_pos_idxs))
                ]

                best_j     = int(np.argmax(combined_scores))
                best_combo = combined_scores[best_j]
                best_pi    = pool_pos_idxs[best_j]

                score_path = path_scores[best_j]
                score_sem  = pool_sims_arr[best_j]

                if best_combo >= threshold_low:
                    best_url   = df_new.iloc[best_pi]["Address"]
                    best_score = best_combo

                    if score_path >= 1.0:
                        method = "Slug Identico"
                    elif score_path >= 0.90:
                        method = "Slug Match (containment)"
                    elif score_path >= 0.80:
                        method = "Slug Match (token overlap)"
                    elif best_path_score >= 0.50:
                        method = "Match Combinato (Path 70% + Semantico 30%)"
                    elif best_path_score > 0.0:
                        method = "Match Combinato (Path 30% + Semantico 70%)"
                    else:
                        method = "Match Semantico"
                else:
                    best_url   = fallback_home
                    best_score = best_combo
                    method     = f"Fallback: score troppo basso ({best_combo*100:.0f}%)"

                # ── Gestione duplicati ─────────────────────────────────
                # Un redirect sbagliato è peggio di un redirect alla home
                if exclude_mapped and best_url in used_new_urls and "Home" not in method and "Fallback" not in method:
                    sorted_j = np.argsort(combined_scores)[::-1]
                    for sj in sorted_j:
                        candidate = df_new.iloc[pool_pos_idxs[sj]]["Address"]
                        if candidate not in used_new_urls:
                            alt_score = combined_scores[sj]
                            if alt_score >= threshold_low * 0.8:
                                # Alternativa accettabile
                                best_url   = candidate
                                best_score = alt_score
                                score_path = path_scores[sj]
                                score_sem  = pool_sims_arr[sj]
                                method    += " (dedup)"
                                is_duplicate = True
                            else:
                                # Alternativa troppo scadente → meglio la home
                                best_url   = fallback_home
                                best_score = alt_score
                                method     = f"Fallback: duplicato, alternativa insufficiente ({alt_score*100:.0f}%)"
                                is_duplicate = True
                            break
                    else:
                        best_url     = fallback_home
                        method       = "Fallback: duplicato, nessun candidato libero"
                        is_duplicate = True

            used_new_urls.add(best_url)

            # ── Costruzione riga risultato ─────────────────────────────
            old_slug_val = Path(urlparse(old_url).path).name or "/"
            new_slug_val = Path(urlparse(best_url).path).name or "/"

            row_result = {
                "Old URL":      old_url,
                "New URL":      best_url,
                "Score":        round(best_score * 100, 1),
                "Score %":      f"{best_score*100:.0f}%",
                "Flag":         flag(best_score, threshold_good, threshold_low),
                "Method":       method,
                "Score Path":   f"{score_path*100:.0f}%",
                "Score Sem":    f"{score_sem*100:.0f}%",
                "Slug Old":     old_slug_val,
                "Slug New":     new_slug_val,
                "Lingua":       f"{old_lang} → {match_lang}" if forced_lang else old_lang,
                "Depth Old":    old_depth,
                "Depth New":    path_depth(best_url),
                "Duplicato":    is_duplicate,
                "Old Title":    str(old_row.get("Title 1", "")),
                "Old H1":       str(old_row.get("H1-1", "")),
            }
            for ec in extra_cols:
                row_result[f"Old {ec}"] = str(old_row.get(ec, ""))

            results.append(row_result)

        status.update(label="✅ Mappatura completata!", state="complete", expanded=False)
        final_df = pd.DataFrame(results)

        # ── Statistiche riassuntive ────────────────────────────────────
        st.markdown("### 📊 Riepilogo")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Totale URL",        len(final_df))
        c2.metric("✅ Confermati",     (final_df["Flag"] == "✅ Confermato").sum())
        c3.metric("⚠️ Da verificare",  (final_df["Flag"] == "⚠️ Da verificare").sum())
        c4.metric("🔴 Fallback",       (final_df["Flag"] == "🔴 Fallback").sum())
        c5.metric("🔁 Dedup forzati",  final_df["Duplicato"].sum())

        # Breakdown per metodo
        n_slug    = final_df["Method"].str.startswith("Slug").sum()
        n_home    = (final_df["Method"] == "Home → Home").sum()
        n_path    = final_df["Method"].str.contains("Combinato|Path 70").sum()
        n_sem     = final_df["Method"].str.contains("Semantico|Path 30").sum()
        n_fallback = final_df["Method"].str.startswith("Fallback").sum()
        total     = len(final_df)

        st.markdown("#### Distribuzione metodo di matching")
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("🎯 Slug esatto/sim",  f"{n_slug} ({n_slug/total*100:.0f}%)")
        d2.metric("🏠 Home → Home",      f"{n_home} ({n_home/total*100:.0f}%)")
        d3.metric("🔗 Path+Sem",         f"{n_path} ({n_path/total*100:.0f}%)")
        d4.metric("🧠 Semantico",        f"{n_sem} ({n_sem/total*100:.0f}%)")
        d5.metric("🔴 Fallback",         f"{n_fallback} ({n_fallback/total*100:.0f}%)")

        # Istogramma distribuzione score
        with st.expander("📈 Distribuzione Score"):
            score_counts = pd.cut(
                final_df["Score"],
                bins=[0, 40, 55, 65, 80, 90, 101],
                labels=["0–40%", "40–55%", "55–65%", "65–80%", "80–90%", "90–100%"],
                right=False
            ).value_counts().sort_index()
            st.bar_chart(score_counts)

        # Validazione anomalie
        if validate_urls:
            with st.expander("🔎 Validazione URL di Destinazione"):
                dest_counts = final_df["New URL"].value_counts()
                high_freq   = dest_counts[dest_counts > 5]
                if not high_freq.empty:
                    st.warning(f"⚠️ {len(high_freq)} URL di destinazione appaiono più di 5 volte:")
                    st.dataframe(high_freq.reset_index().rename(columns={"New URL": "URL", "count": "Occorrenze"}),
                                 use_container_width=True)
                else:
                    st.success("✅ Nessuna destinazione con frequenza anomala.")

                # Score molto diversi per stessa destinazione
                anomalies = (
                    final_df.groupby("New URL")["Score"]
                    .agg(["min", "max", "count"])
                    .query("count > 1 and (max - min) > 30")
                    .reset_index()
                )
                if not anomalies.empty:
                    st.warning(f"⚠️ {len(anomalies)} destinazioni hanno score molto variabili tra vecchie URL:")
                    st.dataframe(anomalies, use_container_width=True)

        # ── Mappa redirect completa ────────────────────────────────────
        st.markdown("### 🗂️ Mappa Redirect")
        display_cols = ["Old URL", "New URL", "Score %", "Flag", "Method",
                        "Score Path", "Score Sem", "Slug Old", "Slug New", "Lingua"]
        styled = final_df[display_cols].style.apply(color_row, axis=1)
        st.dataframe(styled, use_container_width=True)

        # ── Review manuale dei match incerti ───────────────────────────
        review_mask = final_df["Flag"] != "✅ Confermato"
        if review_mask.any():
            with st.expander(f"✏️ Review manuale — {review_mask.sum()} URL da verificare"):
                st.caption("Modifica la colonna **New URL** per correggere i match prima del download Excel.")
                editable_cols = ["Old URL", "New URL", "Score %", "Flag", "Method",
                                 "Slug Old", "Slug New"]
                edited = st.data_editor(
                    final_df.loc[review_mask, editable_cols].reset_index(drop=True),
                    column_config={"New URL": st.column_config.TextColumn("New URL", width="large")},
                    use_container_width=True,
                    key="review_editor",
                )
                # Applica le correzioni manuali al dataframe finale
                review_idxs = final_df.index[review_mask].tolist()
                for j, orig_idx in enumerate(review_idxs):
                    if j < len(edited):
                        edited_url = edited.iloc[j]["New URL"]
                        if edited_url != final_df.at[orig_idx, "New URL"]:
                            final_df.at[orig_idx, "New URL"]   = edited_url
                            final_df.at[orig_idx, "Slug New"]  = Path(urlparse(edited_url).path).name or "/"
                            final_df.at[orig_idx, "Method"]   += " (corretto manualmente)"

        # ── Export Excel ───────────────────────────────────────────────
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            # Esporta tutte le colonne eccetto "Score %" (c'è già "Score" numerico)
            export_df = final_df.drop(columns=["Score %"])
            export_df.to_excel(writer, index=False, sheet_name="Redirect Map")

            wb  = writer.book
            ws  = writer.sheets["Redirect Map"]

            fmt_green  = wb.add_format({"bg_color": "#d4edda", "border": 1})
            fmt_yellow = wb.add_format({"bg_color": "#fff3cd", "border": 1})
            fmt_red    = wb.add_format({"bg_color": "#f8d7da", "border": 1})
            fmt_header = wb.add_format({"bold": True, "bg_color": "#343a40",
                                        "font_color": "white", "border": 1})

            for ci, col_name in enumerate(export_df.columns):
                ws.write(0, ci, col_name, fmt_header)

            flag_col = list(export_df.columns).index("Flag")
            for ri, row_vals in enumerate(export_df.itertuples(index=False), start=1):
                f   = row_vals[flag_col]
                fmt = fmt_green if "✅" in f else (fmt_yellow if "⚠️" in f else fmt_red)
                for ci, val in enumerate(row_vals):
                    ws.write(ri, ci, val, fmt)

            ws.set_column(0, 1, 60)   # Old/New URL
            ws.set_column(2, 2, 10)   # Score
            ws.set_column(3, 7, 20)   # Flag, Method, Score Path/Sem, Slug
            ws.set_column(8, 9, 15)   # Slug Old/New
            ws.freeze_panes(1, 0)

            # Foglio riepilogo
            summary_data = {
                "Categoria": ["Totale URL", "✅ Confermati", "⚠️ Da verificare",
                              "🔴 Fallback", "🎯 Slug match", "🧠 Semantico",
                              "🔁 Dedup forzati"],
                "Conteggio": [
                    len(final_df),
                    (final_df["Flag"] == "✅ Confermato").sum(),
                    (final_df["Flag"] == "⚠️ Da verificare").sum(),
                    (final_df["Flag"] == "🔴 Fallback").sum(),
                    n_slug, n_sem,
                    final_df["Duplicato"].sum(),
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name="Riepilogo")

        st.download_button(
            label="📥 Scarica Excel",
            data=output.getvalue(),
            file_name="redirect_map.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
