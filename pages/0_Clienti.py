import streamlit as st
import json
import re
import requests
from datetime import datetime
from openai import OpenAI
from utils.scraper import scrape_client_deep
from utils.helpers import safe_text

# =========================
# CONFIG PAGINA
# =========================
st.set_page_config(page_title="Gestione Clienti | SEO Brief", layout="wide", page_icon="👤")

GITHUB_RAW_URL = "https://raw.githubusercontent.com/alessandrofantini-ctrl/seo-test/main/profiles/clients.json"

TONE_OPTIONS = ["Autorevole & tecnico", "Empatico & problem solving", "Diretto & commerciale"]

# =========================
# LOAD / SAVE (sessione + GitHub)
# =========================
@st.cache_data(show_spinner=False, ttl=30)
def load_profiles_github() -> dict:
    """Legge il JSON da GitHub (sola lettura, cached 30s)."""
    try:
        r = requests.get(GITHUB_RAW_URL, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

def get_profiles() -> dict:
    """
    Fonte unica di verità: usa la versione in sessione se esiste,
    altrimenti carica da GitHub.
    """
    if "profiles_session" not in st.session_state:
        st.session_state["profiles_session"] = load_profiles_github()
    return st.session_state["profiles_session"]

def set_profiles(profiles: dict):
    """Salva i profili in sessione (non su GitHub)."""
    st.session_state["profiles_session"] = profiles

def build_client_context(profile: dict) -> str:
    products_list = [
        line.strip()
        for line in profile.get("products_services", "").splitlines()
        if line.strip()
    ]
    return "\n".join([
        f"Cliente: {profile.get('name', '')}",
        f"Settore: {profile.get('sector', '')}",
        f"Zona geografica: {profile.get('geo', '')}",
        f"Target audience: {profile.get('target_audience', '')}",
        f"Prodotti/servizi offerti: {products_list}",
        f"USP e punti di forza: {profile.get('usp', '')}",
        f"Note strategiche: {profile.get('notes', '')}",
        f"Keyword già usate: {profile.get('keyword_history', [])}",
    ])

def empty_profile() -> dict:
    return {
        "name": "", "url": "", "sector": "", "brand_name": "",
        "tone_of_voice": "Autorevole & tecnico", "usp": "",
        "products_services": "", "target_audience": "", "geo": "",
        "notes": "", "keyword_history": [],
        "created_at": "", "updated_at": "",
    }

# =========================
# AUTO-GENERAZIONE PROFILO DA URL
# =========================
def generate_profile_from_url(base_url: str, client: OpenAI, model: str) -> dict:
    with st.spinner("Analizzo il sito... (15-30 secondi)"):
        pages_data = scrape_client_deep(base_url, keyword="", max_pages=6)

    if not pages_data:
        st.warning("Non ho trovato contenuto sufficiente. Compila i campi manualmente.")
        return {}

    all_text_parts = []
    for label, page in pages_data:
        h2_str = str(page.get("h2s", []))
        snippet = f"[{label}]\nTitle: {page.get('title','')}\nH1: {page.get('h1','')}\nH2: {h2_str}\n{page.get('text','')[:1200]}"
        all_text_parts.append(snippet)

    combined_text = "\n\n---\n\n".join(all_text_parts)
    system = "Sei un esperto SEO e analista di business. Estrai informazioni strutturate da testi di siti web aziendali."
    prompt = f"""
Analizza il contenuto di questo sito web e restituisci SOLO un oggetto JSON valido con questa struttura esatta:

{{
  "name": "nome azienda/brand",
  "sector": "settore di attività",
  "brand_name": "nome brand commerciale",
  "products_services": "lista prodotti/servizi separati da newline, uno per riga",
  "usp": "punti di forza e differenziatori principali in 2-3 frasi",
  "target_audience": "descrizione del cliente tipo",
  "geo": "zona geografica di operatività",
  "tone_of_voice": "uno tra: Autorevole & tecnico | Empatico & problem solving | Diretto & commerciale",
  "notes": "eventuali note strategiche SEO utili (max 2 frasi)"
}}

Contenuto sito:
{combined_text[:6000]}

Rispondi SOLO con il JSON, senza markdown, senza spiegazioni.
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.2
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        st.error(f"Errore GPT: {e}")
        return {}

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.subheader("🔑 API Keys")
    openai_api_key = st.text_input("OpenAI key", type="password", key="oai_key")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    model_name = st.selectbox("Modello", ["gpt-4o", "gpt-4o-mini"], index=0)

    st.markdown("---")

    # === DOWNLOAD JSON (da caricare su GitHub) ===
    profiles_now = get_profiles()
    st.subheader("💾 Salva su GitHub")
    st.caption("Dopo aver modificato i profili, scarica il JSON e caricalo su GitHub.")
    st.download_button(
        label="📥 Scarica clients.json",
        data=json.dumps(profiles_now, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="clients.json",
        mime="application/json",
        use_container_width=True,
        type="primary",
    )
    st.caption("→ Caricalo in `profiles/clients.json` sul repo GitHub")

    st.markdown("---")

    # === RICARICA DA GITHUB ===
    if st.button("🔄 Ricarica da GitHub", use_container_width=True):
        load_profiles_github.clear()
        if "profiles_session" in st.session_state:
            del st.session_state["profiles_session"]
        st.success("Profili ricaricati da GitHub.")
        st.rerun()

    st.markdown("---")

    # === IMPORT JSON ===
    st.subheader("📥 Importa JSON")
    uploaded = st.file_uploader("Carica un clients.json", type="json")
    if uploaded:
        try:
            imported = json.loads(uploaded.read())
            existing = get_profiles()
            existing.update(imported)
            set_profiles(existing)
            st.success(f"✅ Importati {len(imported)} profili in sessione.")
            st.rerun()
        except Exception as e:
            st.error(f"Errore: {e}")

# =========================
# TITOLO
# =========================
st.title("👤 Gestione profili cliente")

# Banner workflow
st.info(
    "**Workflow:** Modifica i profili qui sotto → clicca **Scarica clients.json** nel sidebar "
    "→ carica il file su GitHub in `profiles/clients.json` → l'app lo rileggerà automaticamente.",
    icon="ℹ️"
)

# =========================
# LAYOUT PRINCIPALE
# =========================
profiles = get_profiles()
col_list, col_detail = st.columns([1, 2], gap="large")

# --- LISTA CLIENTI ---
with col_list:
    st.subheader(f"📋 Clienti ({len(profiles)})")

    if not profiles:
        st.info("Nessun profilo. Crea il primo cliente →")
    else:
        search = st.text_input("🔍 Cerca", placeholder="Nome o settore...")
        filtered = {
            k: v for k, v in profiles.items()
            if not search or search.lower() in k.lower() or search.lower() in v.get("sector", "").lower()
        }
        if not filtered:
            st.warning("Nessun cliente trovato.")
        else:
            for name, profile in filtered.items():
                kw_count = len(profile.get("keyword_history", []))
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**{name}**")
                        st.caption(f"{profile.get('sector','—')} · {kw_count} keyword")
                    with c2:
                        if st.button("Apri", key=f"open_{name}", use_container_width=True):
                            st.session_state["selected_client"] = name
                            st.session_state["client_mode"] = "edit"
                            st.rerun()

    st.markdown("---")
    if st.button("➕ Nuovo cliente", use_container_width=True, type="primary"):
        st.session_state["selected_client"] = None
        st.session_state["client_mode"] = "new"
        st.session_state["prefilled_profile"] = empty_profile()
        st.rerun()

# --- DETTAGLIO / FORM ---
with col_detail:
    mode = st.session_state.get("client_mode", None)
    selected_client = st.session_state.get("selected_client", None)

    # ===== NESSUNA SELEZIONE =====
    if mode is None:
        st.markdown("### Seleziona un cliente o creane uno nuovo")
        st.markdown("""
        **Come funziona:**
        1. Crea o modifica un profilo cliente
        2. Scarica il `clients.json` dal sidebar
        3. Caricalo su GitHub in `profiles/clients.json`
        4. L'app lo rilegge automaticamente in tutti gli strumenti
        """)

    # ===== NUOVO CLIENTE =====
    elif mode == "new":
        st.subheader("➕ Nuovo profilo cliente")

        with st.expander("🤖 Genera automaticamente da URL", expanded=True):
            col_u, col_b = st.columns([3, 1])
            with col_u:
                auto_url = st.text_input("URL sito cliente", placeholder="https://www.cliente.it", key="auto_url_new")
            with col_b:
                st.markdown("<br>", unsafe_allow_html=True)
                run_auto = st.button("🔍 Analizza", use_container_width=True)

            if run_auto:
                if not auto_url:
                    st.error("Inserisci un URL.")
                elif not openai_api_key:
                    st.error("Inserisci la OpenAI key nel sidebar.")
                else:
                    oai_client = OpenAI(api_key=openai_api_key)
                    extracted = generate_profile_from_url(auto_url, oai_client, model_name)
                    if extracted:
                        pf = {**empty_profile(), **extracted, "url": auto_url}
                        st.session_state["prefilled_profile"] = pf
                        st.success("✅ Profilo pre-compilato! Rivedi i campi e salva.")
                        st.rerun()

        pf = st.session_state.get("prefilled_profile", empty_profile())
        st.markdown("---")

        detected_tone = pf.get("tone_of_voice", TONE_OPTIONS[0])
        tone_idx = TONE_OPTIONS.index(detected_tone) if detected_tone in TONE_OPTIONS else 0

        with st.form("form_new_client"):
            new_name = st.text_input("Nome identificativo *", value=pf.get("name", ""), placeholder="Es. Rossi Impianti Srl")
            new_url = st.text_input("URL sito", value=pf.get("url", ""))
            col_a, col_b = st.columns(2)
            with col_a:
                new_sector = st.text_input("Settore", value=pf.get("sector", ""))
                new_brand = st.text_input("Brand name", value=pf.get("brand_name", ""))
                new_geo = st.text_input("Zona geografica", value=pf.get("geo", ""))
            with col_b:
                new_tone = st.selectbox("Tono di voce", TONE_OPTIONS, index=tone_idx)
                new_audience = st.text_input("Target audience", value=pf.get("target_audience", ""))

            new_products = st.text_area(
                "Prodotti / Servizi *", value=pf.get("products_services", ""), height=150,
                placeholder="Es:\nImpianti fotovoltaici industriali\nQuadri elettrici BT/MT"
            )
            new_usp = st.text_area("USP / Punti di forza", value=pf.get("usp", ""), height=90)
            new_notes = st.text_area("Note strategiche SEO", value=pf.get("notes", ""), height=80)

            col_save, col_cancel = st.columns(2)
            with col_save:
                submitted = st.form_submit_button("✅ Salva in sessione", type="primary", use_container_width=True)
            with col_cancel:
                cancelled = st.form_submit_button("❌ Annulla", use_container_width=True)

        if submitted:
            if not new_name:
                st.error("Il nome identificativo è obbligatorio.")
            elif new_name in profiles:
                st.error(f"Esiste già un profilo con nome '{new_name}'.")
            else:
                profiles[new_name] = {
                    "name": new_name, "url": new_url, "sector": new_sector,
                    "brand_name": new_brand, "tone_of_voice": new_tone,
                    "usp": new_usp, "products_services": new_products,
                    "target_audience": new_audience, "geo": new_geo,
                    "notes": new_notes, "keyword_history": [],
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                set_profiles(profiles)
                st.session_state["prefilled_profile"] = empty_profile()
                st.session_state["selected_client"] = new_name
                st.session_state["client_mode"] = "edit"
                st.success(f"✅ '{new_name}' salvato in sessione. Ricorda di scaricare e caricare il JSON su GitHub!")
                st.rerun()

        if cancelled:
            st.session_state["client_mode"] = None
            st.rerun()

    # ===== MODIFICA CLIENTE =====
    elif mode == "edit" and selected_client and selected_client in profiles:
        profile = profiles[selected_client]

        col_title, col_actions = st.columns([3, 1])
        with col_title:
            st.subheader(f"✏️ {selected_client}")
            st.caption(f"Creato: {profile.get('created_at','—')} · Aggiornato: {profile.get('updated_at','—')}")
        with col_actions:
            if st.button("🗑️ Elimina", use_container_width=True):
                st.session_state["confirm_delete"] = True

        if st.session_state.get("confirm_delete"):
            st.warning(f"Eliminare **{selected_client}**?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Sì, elimina", type="primary", use_container_width=True):
                    del profiles[selected_client]
                    set_profiles(profiles)
                    st.session_state["selected_client"] = None
                    st.session_state["client_mode"] = None
                    st.session_state["confirm_delete"] = False
                    st.success("Eliminato dalla sessione. Scarica e carica il JSON su GitHub.")
                    st.rerun()
            with c2:
                if st.button("❌ Annulla", use_container_width=True):
                    st.session_state["confirm_delete"] = False
                    st.rerun()

        tab_edit, tab_history, tab_preview = st.tabs(["📝 Modifica", "📚 Storico keyword", "👁️ Anteprima prompt"])

        with tab_edit:
            current_tone = profile.get("tone_of_voice", TONE_OPTIONS[0])
            tone_idx = TONE_OPTIONS.index(current_tone) if current_tone in TONE_OPTIONS else 0

            with st.expander("🔄 Rigenera da URL"):
                col_u2, col_b2 = st.columns([3, 1])
                with col_u2:
                    regen_url = st.text_input("URL", value=profile.get("url", ""), key="regen_url")
                with col_b2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    run_regen = st.button("🔍 Analizza", key="btn_regen", use_container_width=True)

                if run_regen:
                    if not openai_api_key:
                        st.error("Inserisci la OpenAI key.")
                    else:
                        oai_client = OpenAI(api_key=openai_api_key)
                        extracted = generate_profile_from_url(regen_url or profile.get("url", ""), oai_client, model_name)
                        if extracted:
                            for k, v in extracted.items():
                                if k in profile:
                                    profile[k] = v
                            profile["url"] = regen_url or profile.get("url", "")
                            st.session_state["regen_data"] = profile
                            st.success("✅ Dati aggiornati! Salva le modifiche.")
                            st.rerun()

            regen = st.session_state.get("regen_data", {})
            if regen:
                profile = {**profile, **regen}

            with st.form("form_edit_client"):
                e_url = st.text_input("URL sito", value=profile.get("url", ""))
                col_a, col_b = st.columns(2)
                with col_a:
                    e_sector = st.text_input("Settore", value=profile.get("sector", ""))
                    e_brand = st.text_input("Brand name", value=profile.get("brand_name", ""))
                    e_geo = st.text_input("Zona geografica", value=profile.get("geo", ""))
                with col_b:
                    e_tone = st.selectbox("Tono di voce", TONE_OPTIONS, index=tone_idx)
                    e_audience = st.text_input("Target audience", value=profile.get("target_audience", ""))

                e_products = st.text_area("Prodotti / Servizi", value=profile.get("products_services", ""), height=150)
                e_usp = st.text_area("USP / Punti di forza", value=profile.get("usp", ""), height=90)
                e_notes = st.text_area("Note strategiche SEO", value=profile.get("notes", ""), height=80)

                save_edit = st.form_submit_button("💾 Salva in sessione", type="primary", use_container_width=True)

            if save_edit:
                profiles[selected_client].update({
                    "url": e_url, "sector": e_sector, "brand_name": e_brand,
                    "tone_of_voice": e_tone, "usp": e_usp,
                    "products_services": e_products, "target_audience": e_audience,
                    "geo": e_geo, "notes": e_notes,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                })
                set_profiles(profiles)
                st.session_state.pop("regen_data", None)
                st.success("✅ Salvato in sessione. Ricorda di scaricare e caricare il JSON su GitHub!")
                st.rerun()

        with tab_history:
            history = profile.get("keyword_history", [])
            st.markdown(f"**{len(history)} keyword usate**")
            if not history:
                st.info("Nessuna keyword ancora.")
            else:
                for i, kw in enumerate(reversed(history)):
                    col_kw, col_rm = st.columns([4, 1])
                    with col_kw:
                        st.markdown(f"`{kw}`")
                    with col_rm:
                        if st.button("✕", key=f"rm_kw_{i}"):
                            profiles[selected_client]["keyword_history"].remove(kw)
                            set_profiles(profiles)
                            st.rerun()
                st.markdown("---")
                if st.button("🗑️ Svuota storico", use_container_width=True):
                    profiles[selected_client]["keyword_history"] = []
                    set_profiles(profiles)
                    st.rerun()

        with tab_preview:
            st.markdown("**Contesto cliente inviato al prompt GPT:**")
            st.code(build_client_context(profile), language=None)
