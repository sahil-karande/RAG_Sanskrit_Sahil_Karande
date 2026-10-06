"""
transliterate.py - Sanskrit Transliteration & Text Normalization Module
Handles script detection, Roman-to-Devanagari transliteration (IAST, ITRANS, HK),
and Unicode normalization for Sanskrit queries and corpus text.
"""

import re
import unicodedata
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

DEVANAGARI_RANGE = (0x0900, 0x097F)

def is_devanagari(text: str) -> bool:
    """
    Checks if the string contains predominantly Devanagari characters.
    """
    if not text:
        return False
    devanagari_count = sum(1 for c in text if DEVANAGARI_RANGE[0] <= ord(c) <= DEVANAGARI_RANGE[1])
    total_letters = sum(1 for c in text if c.isalpha())
    if total_letters == 0:
        return False
    return (devanagari_count / total_letters) > 0.5

def detect_transliteration_scheme(text: str) -> str:
    """
    Heuristically detects Roman transliteration scheme (IAST, ITRANS, or Harvard-Kyoto).
    """
    # IAST diacritic markers: ā, ī, ū, ṛ, ṝ, ḷ, ṃ, ḥ, ñ, ṅ, ṭ, ḍ, ṇ, ś, ṣ
    iast_diacritics = re.findall(r'[āīūṛṝḷṃḥñṅṭḍṇśṣĀĪŪṚṜḶṂḤÑṄṬḌṆŚṢ]', text)
    if iast_diacritics:
        return sanscript.IAST
    
    # ITRANS commonly uses: aa, ii, uu, RRi, .m, .h, shh, ch, chh
    itrans_patterns = re.findall(r'(aa|ii|uu|RRi|\.m|\.h|shh|chh|kh|gh|jh|th|dh|ph|bh)', text, re.IGNORECASE)
    if itrans_patterns:
        return sanscript.ITRANS
        
    # Default to Harvard-Kyoto for standard romanized text
    return sanscript.HK

def to_devanagari(text: str, scheme: str = None) -> str:
    """
    Converts romanized Sanskrit text into Devanagari script.
    If text is already in Devanagari, it returns the normalized text.
    """
    if not text or not text.strip():
        return ""
    
    text = unicodedata.normalize('NFKC', text.strip())
    
    if is_devanagari(text):
        return text
    
    if scheme is None:
        scheme = detect_transliteration_scheme(text)
        
    try:
        devanagari_text = transliterate(text, scheme, sanscript.DEVANAGARI)
        return devanagari_text
    except Exception:
        # Fallback to ITRANS
        return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)

def normalize_sanskrit(text: str) -> str:
    """
    Cleans Sanskrit text: normalizes spaces, preserves virama (्) and danda (।, ॥),
    and standardizes quotes.
    """
    if not text:
        return ""
    # NFKC unicode normalization
    text = unicodedata.normalize('NFKC', text)
    # Standardize dandas
    text = text.replace('||', '॥').replace('|', '।')
    # Collapse multiple whitespaces
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

if __name__ == "__main__":
    test_queries = [
        "dharmah",
        "kaalidaasaH",
        "vṛddhāyāḥ cāturyam",
        "kasmai suvarNam dattavaaN",
        "मूर्खभृत्यः कः ?"
    ]
    print("Testing Transliteration Module:")
    for q in test_queries:
        converted = to_devanagari(q)
        print(f"  Input: {q:25} -> Scheme: {detect_transliteration_scheme(q):8} -> Devanagari: {converted}")
