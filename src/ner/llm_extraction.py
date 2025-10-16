import json
import re
import os
from typing import Dict, List
from mistralai import Mistral
from dotenv import load_dotenv

# Charger clé API
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

MISTRAL_MODEL = "mistral-small-latest"


# Dictionnaire des abréviations
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


# Traduction des abréviations avant le LLM


def detect_abbreviations(text):
    """
    Détecte toutes les suites de lettres majuscules (abréviations potentielles).
    Retourne une liste de tuples (abréviation, position_début, position_fin).
    """
    # Pattern : au moins 2 lettres majuscules consécutives (et tiret possible)
    pattern = r'\b[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸ]+(?:-[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸ]+)*\b'

    found_abbreviations = []
    for match in re.finditer(pattern, text):
        found_abbreviations.append(
            (match.group(0), match.start(), match.end()))
    return found_abbreviations


def translate_abbreviations(text, abbr_dict=ABBREVIATIONS_DICT):
    """
    Traduit toutes les abréviations MAJUSCULES trouvées dans le texte.
    Retourne le texte traduit.
    """
    # Pattern robuste pour abréviations majuscules avec tirets ou slash
    pattern = r'\b[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸ]+(?:[-/][A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸ]+)*\b'

    matches = list(re.finditer(pattern, text))

    # Tri décroissant pour éviter les problèmes de déplacement d'index
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


# Extraction des entités
def format_abbreviations_for_prompt(abbrev_dict: Dict[str, str]) -> str:
    """Transforme le dictionnaire en liste lisible pour le prompt."""
    return "\n".join([f"- {k} → {v}" for k, v in abbrev_dict.items()])


def extract_entities(text: str) -> Dict[str, List[str]]:
    """ 
    Extrait les entités d'un texte en français en utilisant l'API Mistral. 
    Retourne un dict {label: [valeurs]}.
    """
    labels = [
        "GEO", "ACTOR", "DATETIME", "EVENT",
        "INFRASTRUCTURE", "OPERATING_CONTEXT",
        "PHONE_NUMBER", "ELECTRICAL_VALUE"
    ]

    if not text.strip():
        return {label: [] for label in labels}

    translated_text = translate_abbreviations(text)
    abbrev_str = format_abbreviations_for_prompt(ABBREVIATIONS_DICT)

    prompt = f"""
    Tu es un extracteur d'entités strict en français.

    **Règles absolues** :
    - Tu ne dois **jamais inventer d'entités**.
    - N'extrais que des sous-chaînes **exactement présentes** dans le texte ci-dessous.
    - Les entités doivent être **formées de mots consécutifs** dans le texte d'origine, dans le même ordre.
    - N'ajoute aucune interprétation, reformulation ou élément typique du domaine s'il n'est pas écrit noir sur blanc.
    
     Les abréviations suivantes ont été traduites dans le texte en leur forme développée complète.  
    Elles doivent être considérées comme des **blocs indivisibles** : si une de ces expressions apparaît dans le texte, tu dois **toujours extraire l'expression entière**, jamais une sous-partie.

    {abbrev_str}
    
    Texte :
    \"\"\"{translated_text}\"\"\"

    Catégories :
    - GEO : noms de lieux, villes, régions, axes géographiques. Inclure les noms composés ou reliés par des tirets (ex : "Ivry-sur-Seine", "Trappes-Deauville", "Paris-Lyon"). Lorsque deux villes sont reliées par un tiret, considère l'ensemble comme **une seule entité GEO**.
    - ACTOR : noms d'acteurs (personnes, équipes, entités organisationnelles) 
    - DATETIME : dates ou horaires (y compris formats naturels) 
    - EVENT : événements spécifiques (ex : incident, manœuvre, etc.) 
    - INFRASTRUCTURE : noms ou identifiants d'ouvrages électriques, lignes, postes 
    - OPERATING_CONTEXT : contexte opérationnel (ordres, statuts, conditions particulières) 
    - PHONE_NUMBER : numéros de téléphone 
    - ELECTRICAL_VALUE : valeurs électriques comme "225kV", "63 kV", "400 MW", etc.

    Règles :
    - Ne produis que du JSON strict valide.
    - Chaque valeur doit être une sous-chaîne copiée exactement telle qu'elle apparaît dans le texte.
    - Si aucune entité n'est trouvée pour une catégorie, mets une liste vide.
    - N'inclus aucune explication, aucun commentaire, aucun texte hors du JSON.

    Format attendu :
    {{
    "GEO": [],
    "ACTOR": [],
    "DATETIME": [],
    "EVENT": [],
    "INFRASTRUCTURE": [],
    "OPERATING_CONTEXT": [],
    "PHONE_NUMBER": [],
    "ELECTRICAL_VALUE": []
    }}
    
    Quelques exemples de mots par catégories : 
    {{ "GEO": ["Caen", "Chartres", "Paris"], 
    "ACTOR": ["SNCF, Enedis", "Chargé d'exploitation"], 
    "DATETIME": ["15 juillet 2023", "ce matin", "demain", "14h", "18:00", "12/03/2020"], 
    "EVENT": ["Manoeuvre", "Incident", "Retrait", "Bascule"], 
    "INFRASTRUCTURE": ["Ligne", "N-1", "Transformateur"], 
    "OPERATING_CONTEXT": ["invalide en HO", "Régime surveillance dispatching"], 
    "PHONE_NUMBER": ["06 12 34 56 78", "+33 9 38 39 29 30"], 
    "ELECTRICAL_VALUE": ["225kV", "38 kV", "400 GW"] }} 
    """

    # Appel API Mistral
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    output = response.choices[0].message.content

    # Extraction robuste du JSON généré
    start, end = output.find("{"), output.rfind("}") + 1
    json_str = output[start:end] if start != -1 and end != -1 else "{}"

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        data = {}

    # Normalisation des données
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


# Test
if __name__ == "__main__":
    print(extract_entities("Etude RDCR Caen Cherbourg 225kV au +33 3 93 93 93 93"))
