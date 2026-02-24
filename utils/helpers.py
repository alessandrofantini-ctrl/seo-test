import re
import requests
from collections import Counter
from requests.adapters import HTTPAdapter, Retry

# =========================
# HTTP SESSION + RETRY
# =========================
def build_session():
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

HTTP = build_session()

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# =========================
# STOPWORDS
# =========================
STOPWORDS = set("""
a al allo alla alle agli all' and are as at be by con che da dal dalla dalle degli dei del della delle di do
e ed en est et for from il in is it la le lo los las les more nel nei nell' of on or per por pour que qui
su the to un una uno und une with y zu""".split())

# =========================
# TEXT UTILS
# =========================
def safe_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

def truncate_chars(s: str, n: int) -> str:
    s = safe_text(s)
    return s[:n].rstrip()

def normalize_sentence_case(text: str) -> str:
    t = safe_text(text)
    if not t:
        return ""
    if t.isupper() and len(t) > 6:
        t = t.lower()
    return t[0].upper() + t[1:] if len(t) > 1 else t.upper()

def tokenize(text: str):
    text = (text or "").lower()
    tokens = re.findall(r"[a-zàèéìòùäöüßñç0-9]{3,}", text, flags=re.I)
    return [t for t in tokens if t not in STOPWORDS]

def domain_of(url: str) -> str:
    from urllib.parse import urlparse
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""
