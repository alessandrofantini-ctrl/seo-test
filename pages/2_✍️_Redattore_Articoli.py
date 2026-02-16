import streamlit as st
from openai import OpenAI
from docx import Document
from io import BytesIO
import re

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

st.title("✍️ SEO Copywriter – Expert Mode")
st.markdown("Genera contenuti SEO scritti come un **esperto reale**, non come un template AI.")

# =========================
# INPUT BRIEF
# =========================
brief_input = st.text_area("Incolla il Brief SEO", height=500)

# =========================
# PARAMETRI LUNGHEZZA
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
# GENERAZIONE CONTENUTO
# =========================
if st.button("🚀 Genera Articolo SEO"):
    if not openai_api_key or not brief_input:
        st.error("Inserisci API Key e Brief.")
    else:
        try:
            client = OpenAI(api_key=openai_api_key)
            word_target = get_word_target(lunghezza_label)

            system_prompt = """
You are a Senior SEO Copywriter and Home Barista Expert.

You write authoritative, detailed, data-informed content that ranks on Google and is optimized for LLM retrieval.

You NEVER:
- write generic introductions
- use filler phrases
- repeat obvious information
- create placeholder sections (no “video tutorial” text)
- summarize when depth is required

You ALWAYS:
- open with a strong introduction (hook + promise + context + keyword)
- write dense, information-rich paragraphs
- include numbers, parameters, examples
- structure clearly with Markdown headings
- maintain consistent language (no mixing languages)
- write naturally as a human expert
"""

            user_prompt = f"""
Use this SEO Brief as your only strategic input:

{brief_input}

Write the complete article.

STRICT REQUIREMENTS:

1. Start with a strong introduction (150-200 words) that:
   - naturally includes the primary keyword
   - explains why making espresso at home matters
   - sets expectations (what the reader will master)
   - positions the writer as an expert

2. Follow the exact H1/H2/H3 structure provided.

3. Respect:
   - Primary keyword
   - Secondary keywords (integrate naturally)
   - Mandatory entities
   - Gap analysis improvements
   - Intent alignment

4. Expand depth:
   - Include technical parameters (grams, ratios, timing, temperature)
   - Include comparison tables where useful
   - Add practical decision criteria
   - Add troubleshooting logic trees
   - Add maintenance best practices

5. Make the article LLM-friendly:
   - Define key terms explicitly
   - Avoid vague statements
   - Use structured explanations
   - Add summary bullets where useful

6. Length target: {word_target} words minimum.
   DO NOT write a short article.

7. Finish with:
   - FAQ section (optimized for schema)
   - Strategic CTA aligned with provided notes

Write in fluent American English.
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
