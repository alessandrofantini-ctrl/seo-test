import streamlit as st
from openai import OpenAI
from docx import Document
from io import BytesIO
import re
import json

st.set_page_config(page_title="Redattore AI - SEO Expert Mode", layout="wide")

# =========================
# SIDEBAR (API KEY INVARIATA)
# =========================
with st.sidebar:
    st.title("⚙️ Configurazione")

    openai_api_key = st.text_input("OpenAI Key", type="password")
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]

    st.markdown("---")

    lunghezza_label = st.select_slider(
        "Lunghezza Articolo",
        options=["Standard", "Long Form", "Authority Guide (Pillar)"],
        value="Long Form"
    )

    creativita = st.slider("Livello Creatività", 0.0, 1.0, 0.5)

# =========================
# IMPORT AUTOMATICO BRIEF DA ANALISI SEO
# =========================
brief_default = ""
json_data = None

if "ultimo_brief" in st.session_state:
    st.success("✅ Brief importato automaticamente dal tool Analisi SEO.")
    brief_default = st.session_state["ultimo_brief"]

    # Estrazione JSON dal brief
    match = re.search(r"```json\s*(\{.*?\})\s*```", brief_default, re.S)
    if match:
        try:
            json_data = json.loads(match.group(1))
        except:
            json_data = None

st.title("✍️ SEO Copywriter – Connected Mode")

brief_input = st.text_area(
    "Brief SEO (auto-importato, modificabile)",
    value=brief_default,
    height=450
)

# =========================
# LUNGHEZZA TARGET
# =========================
def get_word_target(label):
    if label == "Standard":
        return "1200-1600"
    elif label == "Long Form":
        return "1800-2500"
    else:
        return "2500-3500"

# =========================
# DOCX EXPORT
# =========================
def create_docx(content):
    doc = Document()
    doc.add_heading("SEO Article", 0)
    for line in content.split("\n"):
        if line.startswith("# "):
            doc.add_heading(line.replace("# ", ""), level=1)
        elif line.startswith("## "):
            doc.add_heading(line.replace("## ", ""), level=2)
        elif line.startswith("### "):
            doc.add_heading(line.replace("### ", ""), level=3)
        else:
            doc.add_paragraph(line)
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

# =========================
# GENERAZIONE ARTICOLO
# =========================
if st.button("🚀 Genera Articolo SEO"):
    if not openai_api_key or not brief_input:
        st.error("Inserisci API Key e Brief.")
    else:
        try:
            client = OpenAI(api_key=openai_api_key)
            word_target = get_word_target(lunghezza_label)

            # Se JSON presente, estrai dati strategici
            primary_kw = ""
            language = "English"

            if json_data:
                primary_kw = json_data.get("primary_keyword", "")
                language = json_data.get("language", "English")

            system_prompt = f"""
You are a Senior SEO Copywriter.

You write high-authority, in-depth content that ranks and is optimized for LLM retrieval.

Rules:
- Write in {language}
- Avoid generic phrasing
- Include technical depth
- Use structured Markdown
- Include tables where relevant
- Provide clear definitions
- Maintain expert tone
"""

            user_prompt = f"""
Use this SEO brief:

{brief_input}

Write the complete article.

Mandatory requirements:

1. Write a powerful introduction (150-200 words) that:
   - Includes the primary keyword "{primary_kw}"
   - Establishes authority
   - Explains what the reader will master

2. Follow the provided H1/H2/H3 structure exactly.

3. Integrate secondary keywords naturally.

4. Expand on the gap analysis:
   - Add deeper explanations
   - Include technical parameters
   - Add comparisons
   - Add troubleshooting frameworks

5. Minimum length: {word_target} words.
   Do NOT write a short article.

6. End with:
   - Optimized FAQ section
   - Strategic CTA

Write like a real expert, not like an AI.
"""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=creativita,
                max_tokens=7000
            )

            article = response.choices[0].message.content

            st.markdown("## 📄 Articolo Generato")
            st.markdown(article)

            docx = create_docx(article)
            st.download_button(
                "📥 Scarica Articolo (.docx)",
                data=docx,
                file_name="seo_article.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

        except Exception as e:
            st.error(f"Errore: {e}")
