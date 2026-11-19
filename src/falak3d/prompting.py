"""Préparation de la description envoyée à Shap-E.

Shap-E a été entraîné sur des légendes en anglais : une description en français
donne un objet sans rapport, ou une forme molle. Les utilisateurs, eux, écrivent
en français. On traduit donc en amont, idéalement avec le LLM local déjà présent
sur la machine, et on garde un petit lexique pour fonctionner sans lui.
"""

import re

import httpx

GLOSSARY = {
    "chaise": "chair", "table": "table", "lampe": "lamp", "tasse": "mug", "bouteille": "bottle",
    "boîte": "box", "boite": "box", "carton": "cardboard box", "canapé": "sofa",
    "étagère": "shelf", "fauteuil": "armchair", "vase": "vase", "voiture": "car",
    "arbre": "tree", "maison": "house", "épée": "sword", "coffre": "chest",
    "tonneau": "barrel", "bureau": "desk", "rouge": "red", "bleu": "blue",
    "vert": "green", "noir": "black", "blanc": "white", "jaune": "yellow",
    "bois": "wooden", "métal": "metal", "metal": "metal", "verre": "glass",
    "petit": "small", "petite": "small", "grand": "large", "grande": "large",
    "ronde": "round", "rond": "round", "carré": "square", "avec": "with",
}

# Articles et prépositions : les gabarits de preset posent déjà leur article.
DROP = {"un", "une", "le", "la", "les", "en", "de", "du", "des", "d'"}
# En français l'adjectif suit le nom, en anglais il le précède.
ADJECTIVES = {"red", "blue", "green", "black", "white", "yellow", "wooden", "metal",
              "glass", "small", "large", "round", "square"}

TRANSLATE_PROMPT = (
    "Translate this description of a physical object into short English, "
    "as a 3D model caption. Answer with the caption only, no quotes.\n\n{text}"
)

MAX_CHARS = 200


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CHARS]


def looks_english(text: str) -> bool:
    if re.search(r"[éèêàâçùûôî]", text.lower()):
        return False
    return not any(w in GLOSSARY and GLOSSARY[w] != w for w in text.lower().split())


def glossary_translate(text: str) -> str:
    words = [GLOSSARY.get(w, w) for w in text.lower().split() if w not in DROP]
    adjectives = [w for w in words if w in ADJECTIVES]
    return " ".join(adjectives + [w for w in words if w not in ADJECTIVES])


def to_english(text: str, ollama_url: str | None = None, model: str = "",
               timeout: float = 60.0) -> str:
    text = clean(text)
    if looks_english(text):
        return text
    if ollama_url:
        try:
            r = httpx.post(f"{ollama_url}/api/generate", timeout=timeout, json={
                "model": model, "prompt": TRANSLATE_PROMPT.format(text=text),
                "stream": False, "options": {"temperature": 0, "num_predict": 60},
            })
            r.raise_for_status()
            answer = clean(r.json()["response"]).strip("\"'. ")
            if answer:
                return answer
        except (httpx.HTTPError, KeyError):
            pass
    return glossary_translate(text)
