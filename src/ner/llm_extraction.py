# Extracts, via a call to the Mistral model API, the entities belonging to the different categories

import sys, os
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
import json
import re
import os
from mistralai import Mistral
from dotenv import load_dotenv
from src.ner.ner_prompt_template import NER_PROMPT


# Load the API key
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

# Select the model
MISTRAL_MODEL_NER = "mistral-small-latest"

# Dictionary of abbreviations
ABBREVIATIONS_DICT = {
    "RACR": "Retour à la conduite des réseaux",
    "RDCR": "Retrait de la conduite des réseaux",
    "TIR": "Thermographie infrarouge",
    "PO": "Poste opérateur",
    "RSD": "Régime surveillance dispatching",
    "SUAV": "Sous tension à vide",
    "MNV": "Manoeuvre",
    "PF": "Point figé",
    "CSS": "Central sous station (SNCF)",
    "GEH": "Groupe exploitation hydraulique",
    "PDM": "Personnel de manoeuvre",
    "SMACC": "Système avancé pour la commande de la compensation",
    "HO": "Heures ouvrées",
    "BR": "Bâtiment de relayage",
    "GT": "Groupe de traction",
    "TST": "Travaux sous tensions",
    "CCO": "Chargé de conduite",
    "FDE": "Filet d'essai",
    "DIFFB": "Protection différentielle de barre",
    "RADA": "Retrait agile suite à détection d'anomalie",
    "TR": "Transformateur",
    "RA": "Réenclencheur automatique",
    "CTS": "Cycle triphasé supplémentaire",
    "CEX": "Chargé d'exploitation",
    "COSE": "Centre opérationnel du système électrique",
    "COSE-P": "Centre opérationnel du système électrique de Paris"
}


# Translate abbreviations before passing to the LLM

def translate_abbreviations(text: str, abbr_dict: dict[str, str] = ABBREVIATIONS_DICT) -> str:
    """
    Translates all uppercase abbreviations found in the text.
    Returns the translated text.
    """
    # Robust pattern for uppercase abbreviations with hyphens or slashes
    pattern = r'\b[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸ]+(?:[-/][A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸ]+)*\b'

    matches = list(re.finditer(pattern, text))

    # Robust pattern for uppercase abbreviations with hyphens or slashes
    matches = sorted(matches, key=lambda m: m.start(), reverse=True)

    translated_text = text

    for m in matches:
        abbr = m.group(0)
        start, end = m.start(), m.end()

        if abbr in abbr_dict:
            translation = abbr_dict[abbr]

            translated_text = translated_text[:start] + \
                translation + translated_text[end:]

    return translated_text


# Entity extraction

def format_abbreviations_for_prompt(abbrev_dict: dict[str, str]) -> str:
    """
    Converts the dictionary into a readable list for the prompt.
    """
    return "\n".join([f"- {k} --> {v}" for k, v in abbrev_dict.items()])


def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Extracts entities from a French text using the Mistral API.
    Returns a dict {label: [values]}.
    """
    labels = [
        "GEO", "ACTOR", "DATETIME", "EVENT",
        "INFRASTRUCTURE", "OPERATING_CONTEXT",
        "PHONE_NUMBER", "ELECTRICAL_VALUE", 
        "ABBREVIATION_UNKNOWN"
    ]

    if not text.strip():
        return {label: [] for label in labels}

    translated_text = translate_abbreviations(text)
    abbrev_str = format_abbreviations_for_prompt(ABBREVIATIONS_DICT)

    prompt = NER_PROMPT.format(translated_text=translated_text, abbrev_str=abbrev_str)

    # Mistral API call
    response = client.chat.complete(
        model=MISTRAL_MODEL_NER,
        messages=[{"role": "user", "content": prompt}]
    )

    output = response.choices[0].message.content

    # Robust extraction of the generated JSON
    start, end = output.find("{"), output.rfind("}") + 1
    json_str = output[start:end] if start != -1 and end != -1 else "{}"

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        data = {}

    # Data normalization
    normalized = {}
    for label in labels:
        val = data.get(label, [])
        if isinstance(val, str):
            normalized[label] = [val.strip()]
        elif isinstance(val, list):
            normalized[label] = [str(v).strip() for v in val if str(v).strip()]
        else:
            normalized[label] = []

    return normalized