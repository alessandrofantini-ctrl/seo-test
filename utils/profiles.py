import json
from pathlib import Path
from datetime import datetime

PROFILES_PATH = Path("profiles/clients.json")


def load_profiles() -> dict:
    if PROFILES_PATH.exists():
        try:
            return json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_profiles(profiles: dict):
    PROFILES_PATH.parent.mkdir(exist_ok=True)
    PROFILES_PATH.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def empty_profile() -> dict:
    return {
        "name": "",
        "url": "",
        "sector": "",
        "brand_name": "",
        "tone_of_voice": "Autorevole & tecnico",
        "usp": "",
        "products_services": "",
        "target_audience": "",
        "geo": "",
        "notes": "",
        "keyword_history": [],
        "created_at": "",
        "updated_at": "",
    }


def add_keyword_to_history(profile_name: str, keyword: str):
    """Aggiunge una keyword allo storico del profilo cliente."""
    profiles = load_profiles()
    if profile_name not in profiles:
        return
    history = profiles[profile_name].get("keyword_history", [])
    if keyword not in history:
        history.append(keyword)
        profiles[profile_name]["keyword_history"] = history[-50:]
        profiles[profile_name]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        save_profiles(profiles)


def build_client_context(profile: dict) -> str:
    """
    Costruisce il blocco di testo contesto-cliente da iniettare nel prompt GPT.
    """
    products_list = [
        line.strip()
        for line in profile.get("products_services", "").splitlines()
        if line.strip()
    ]
    lines = [
        f"Cliente: {profile.get('name', '')}",
        f"Settore: {profile.get('sector', '')}",
        f"Zona geografica: {profile.get('geo', '')}",
        f"Target audience: {profile.get('target_audience', '')}",
        f"Prodotti/servizi offerti: {products_list}",
        f"USP: {profile.get('usp', '')}",
        f"Note strategiche: {profile.get('notes', '')}",
        f"Keyword già usate (evita duplicati o sfruttale per topic cluster): "
        f"{profile.get('keyword_history', [])}",
    ]
    return "\n".join(lines)
