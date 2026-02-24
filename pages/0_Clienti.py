import streamlit as st
import json
import re
from datetime import datetime
from openai import OpenAI

from utils.profiles import load_profiles, save_profiles, empty_profile, build_client_context
from utils.scraper import scrape_client_deep
from utils.helpers import safe_text, truncate_chars

# =========================
# CONFIG PAGINA
# =========================
st.set_page_config(page_title="Gestione Clienti | SEO Brief", layout="wide", page_icon="👤")

st.title("👤 Gestione profili cliente")
st.markdown("Crea, modifica ed elimina i profili dei tuoi clienti. Ogni profilo viene caricato automaticamente nel tool di generazione brief.")

# =========================
# API KEY (necessaria per auto-generazione)
# =========================
with st.sidebar:
    st.subheader("🔑 API Keys")
    openai_api_key = st.text_input("OpenAI key", type="password", key="oai_key")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]

    model_name = st.selectbox("Modello", ["gpt-4o", "gpt-4o-mini"], index=0)

    st.markdown("---")
    st.caption("Le API key sono necessarie solo per la generazione automatica del profilo da URL.")

    st.markdown("---")
    st.subheader("📦 Import / Export")
    profiles_for_export = load_profiles()
    if profiles_for_export:
        st.download_button(
            "📤 Esporta tutti i profili",
            data=json.dumps(profiles_for_export, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="clienti_backup.json",
            mime="application/json",
        )
    uploaded = st.file_uploader("📥 Importa profili (.json)", type="json")
    if uploaded:
        try:
            imported = json.loads(uploaded.read())
            existing = load_profiles()
            existing.update(imported)
            save_profiles(existing)
            st.success(f"✅ Importati {len(imported)} profili.")
            st.rerun()
        except Exception as e:
            st.error(f"Errore importazione: {e}")

# =========================
# FUNZIONE AUTO-GENERAZIONE PROFILO
# =========================
def generate_profile_from_url(base_url: str, client: OpenAI, model: str) -> dict:
    """Crawla il sito e usa GPT per estrarre un profilo strutturato."""
    with st.spinner("Analizzo il sito... (15-30 secondi)"):
        pages_data = scrape_client_deep(base_url, keyword="", max_pages=6)

    if not pages_data:
        st.warning("Non ho trovato contenuto sufficiente. Compila i campi manualmente.")
        return {}

    # Costruisce testo aggregato da tutte le pagine
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
  "sector": "settore di attività (es. Impianti elettrici industriali, E-commerce moda, etc.)",
  "brand_name": "nome brand commerciale",
  "products_services": "lista prodotti/servizi separati da newline, uno per riga",
  "usp": "punti di forza e differenziatori principali in 2-3 frasi",
  "target_audience": "descrizione del cliente tipo (es. PMI manifatturiere, privati, professionisti)",
  "geo": "zona geografica di operatività (es. Italia, Nord Italia, Milano e provincia)",
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
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
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
# LAYOUT PRINCIPALE
# =========================
profiles = load_profiles()

# Colonna sinistra: lista clienti | Colonna destra: dettaglio/form
col_list, col_detail = st.columns([1, 2], gap="large")

# --- LISTA CLIENTI ---
with col_list:
    st.subheader(f"📋 Clienti ({len(profiles)})")

    if not profiles:
        st.info("Nessun profilo ancora. Crea il primo cliente →")
    else:
        # Ricerca
        search = st.text_input("🔍 Cerca cliente", placeholder="Nome o settore...")
        filtered = {
            k: v for k, v in profiles.items()
            if not search or search.lower() in k.lower() or search.lower() in v.get("sector", "").lower()
        }

        if not filtered:
            st.warning("Nessun cliente trovato.")
        else:
            for name, profile in filtered.items():
                kw_count = len(profile.get("keyword_history", []))
                sector = profile.get("sector", "—")
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**{name}**")
                        st.caption(f"{sector} · {kw_count} keyword usate")
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

    # ===== STATO: NESSUNA SELEZIONE =====
    if mode is None:
        st.markdown("### Seleziona un cliente dalla lista o creane uno nuovo")
        st.markdown("""
        Con i profili cliente puoi:
        - 🚀 Avviare il brief senza dover ricompilare ogni volta i dati del cliente
        - 📚 Tenere traccia delle keyword già usate per ogni cliente
        - 🎯 Assicurarti che il brief rispecchi prodotti e servizi reali del cliente
        - 📤 Esportare e importare i profili per backup o condivisione con il team
        """)

    # ===== STATO: NUOVO CLIENTE =====
    elif mode == "new":
        st.subheader("➕ Nuovo profilo cliente")

        # Auto-generazione da URL
        with st.expander("🤖 Genera automaticamente da URL del sito", expanded=True):
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
                        st.success("✅ Profilo pre-compilato! Rivedi i campi qui sotto e salva.")
                        st.rerun()

        pf = st.session_state.get("prefilled_profile", empty_profile())

        st.markdown("---")
        st.markdown("**✏️ Compila o modifica i campi**")

        tone_options = ["Autorevole & tecnico", "Empatico & problem solving", "Diretto & commerciale"]
        detected_tone = pf.get("tone_of_voice", "Autorevole & tecnico")
        tone_idx = tone_options.index(detected_tone) if detected_tone in tone_options else 0

        with st.form("form_new_client"):
            new_name = st.text_input("Nome identificativo *", value=pf.get("name", ""), placeholder="Es. Rossi Impianti Srl")
            new_url = st.text_input("URL sito", value=pf.get("url", ""))

            col_a, col_b = st.columns(2)
            with col_a:
                new_sector = st.text_input("Settore", value=pf.get("sector", ""))
                new_brand = st.text_input("Brand name", value=pf.get("brand_name", ""))
                new_geo = st.text_input("Zona geografica", value=pf.get("geo", ""))
            with col_b:
                new_tone = st.selectbox("Tono di voce", tone_options, index=tone_idx)
                new_audience = st.text_input("Target audience", value=pf.get("target_audience", ""))

            new_products = st.text_area(
                "Prodotti / Servizi offerti *",
                value=pf.get("products_services", ""),
                height=150,
                help="Uno per riga. Più è dettagliato, più il brief sarà preciso.",
                placeholder="Es:\nImpianti fotovoltaici industriali\nQuadri elettrici BT/MT\nManutenzione impianti"
            )
            new_usp = st.text_area("USP / Punti di forza", value=pf.get("usp", ""), height=90)
            new_notes = st.text_area("Note strategiche SEO", value=pf.get("notes", ""), height=80,
                                     placeholder="Es. Puntare su long-tail locali, evitare keyword già presidiate da X...")

            col_save, col_cancel = st.columns([1, 1])
            with col_save:
                submitted = st.form_submit_button("✅ Salva profilo", type="primary", use_container_width=True)
            with col_cancel:
                cancelled = st.form_submit_button("❌ Annulla", use_container_width=True)

        if submitted:
            if not new_name:
                st.error("Il nome identificativo è obbligatorio.")
            elif new_name in profiles:
                st.error(f"Esiste già un profilo con nome '{new_name}'. Scegli un nome diverso.")
            else:
                profiles[new_name] = {
                    "name": new_name,
                    "url": new_url,
                    "sector": new_sector,
                    "brand_name": new_brand,
                    "tone_of_voice": new_tone,
                    "usp": new_usp,
                    "products_services": new_products,
                    "target_audience": new_audience,
                    "geo": new_geo,
                    "notes": new_notes,
                    "keyword_history": [],
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                save_profiles(profiles)
                st.session_state["prefilled_profile"] = empty_profile()
                st.session_state["selected_client"] = new_name
                st.session_state["client_mode"] = "edit"
                st.success(f"✅ Profilo '{new_name}' creato!")
                st.rerun()

        if cancelled:
            st.session_state["client_mode"] = None
            st.rerun()

    # ===== STATO: MODIFICA CLIENTE =====
    elif mode == "edit" and selected_client and selected_client in profiles:
        profile = profiles[selected_client]

        # Header con nome e azioni rapide
        col_title, col_actions = st.columns([3, 1])
        with col_title:
            st.subheader(f"✏️ {selected_client}")
            st.caption(f"Creato: {profile.get('created_at','—')} · Aggiornato: {profile.get('updated_at','—')}")
        with col_actions:
            if st.button("🗑️ Elimina", use_container_width=True):
                st.session_state["confirm_delete"] = True

        # Conferma eliminazione
        if st.session_state.get("confirm_delete"):
            st.warning(f"Sei sicuro di voler eliminare **{selected_client}**? L'azione è irreversibile.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Sì, elimina", type="primary", use_container_width=True):
                    del profiles[selected_client]
                    save_profiles(profiles)
                    st.session_state["selected_client"] = None
                    st.session_state["client_mode"] = None
                    st.session_state["confirm_delete"] = False
                    st.success("Profilo eliminato.")
                    st.rerun()
            with c2:
                if st.button("❌ Annulla", use_container_width=True):
                    st.session_state["confirm_delete"] = False
                    st.rerun()

        # Tab: Modifica / Storico keyword / Anteprima prompt
        tab_edit, tab_history, tab_preview = st.tabs(["📝 Modifica", "📚 Storico keyword", "👁️ Anteprima prompt"])

        with tab_edit:
            tone_options = ["Autorevole & tecnico", "Empatico & problem solving", "Diretto & commerciale"]
            current_tone = profile.get("tone_of_voice", "Autorevole & tecnico")
            tone_idx = tone_options.index(current_tone) if current_tone in tone_options else 0

            # Rigenera da URL
            with st.expander("🔄 Rigenera automaticamente da URL"):
                col_u2, col_b2 = st.columns([3, 1])
                with col_u2:
                    regen_url = st.text_input("URL", value=profile.get("url", ""), key="regen_url")
                with col_b2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    run_regen = st.button("🔍 Analizza", key="btn_regen", use_container_width=True)

                if run_regen:
                    if not openai_api_key:
                        st.error("Inserisci la OpenAI key nel sidebar.")
                    else:
                        oai_client = OpenAI(api_key=openai_api_key)
                        extracted = generate_profile_from_url(regen_url or profile.get("url",""), oai_client, model_name)
                        if extracted:
                            # Aggiorna solo i campi estratti, mantieni storico keyword
                            for k, v in extracted.items():
                                if k in profile:
                                    profile[k] = v
                            profile["url"] = regen_url or profile.get("url","")
                            st.session_state["regen_data"] = profile
                            st.success("✅ Dati aggiornati! Controlla i campi e salva.")
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
                    e_tone = st.selectbox("Tono di voce", tone_options, index=tone_idx)
                    e_audience = st.text_input("Target audience", value=profile.get("target_audience", ""))

                e_products = st.text_area(
                    "Prodotti / Servizi offerti",
                    value=profile.get("products_services", ""),
                    height=150
                )
                e_usp = st.text_area("USP / Punti di forza", value=profile.get("usp", ""), height=90)
                e_notes = st.text_area("Note strategiche SEO", value=profile.get("notes", ""), height=80)

                save_edit = st.form_submit_button("💾 Salva modifiche", type="primary", use_container_width=True)

            if save_edit:
                profiles[selected_client].update({
                    "url": e_url,
                    "sector": e_sector,
                    "brand_name": e_brand,
                    "tone_of_voice": e_tone,
                    "usp": e_usp,
                    "products_services": e_products,
                    "target_audience": e_audience,
                    "geo": e_geo,
                    "notes": e_notes,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                })
                save_profiles(profiles)
                st.session_state.pop("regen_data", None)
                st.success("✅ Profilo aggiornato!")
                st.rerun()

        with tab_history:
            history = profile.get("keyword_history", [])
            st.markdown(f"**{len(history)} keyword usate** per questo cliente")

            if not history:
                st.info("Nessuna keyword ancora. Verranno aggiunte automaticamente ogni volta che generi un brief.")
            else:
                # Mostra in ordine inverso (più recenti prima)
                for i, kw in enumerate(reversed(history)):
                    col_kw, col_rm = st.columns([4, 1])
                    with col_kw:
                        st.markdown(f"`{kw}`")
                    with col_rm:
                        if st.button("✕", key=f"rm_kw_{i}", help="Rimuovi"):
                            profiles[selected_client]["keyword_history"].remove(kw)
                            save_profiles(profiles)
                            st.rerun()

                st.markdown("---")
                if st.button("🗑️ Svuota storico", use_container_width=True):
                    profiles[selected_client]["keyword_history"] = []
                    save_profiles(profiles)
                    st.rerun()

        with tab_preview:
            st.markdown("**Anteprima del contesto cliente che viene inviato al prompt GPT:**")
            st.markdown("*Questo è esattamente il testo che il modello riceve per personalizzare il brief.*")
            ctx = build_client_context(profile)
            st.code(ctx, language=None)
