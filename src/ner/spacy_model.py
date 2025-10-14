import spacy
from spacy.pipeline import EntityRuler
import re


# Dictionnaire des abréviations
ABBREVIATIONS_DICT = {
    "RACR": ("Retour à la conduite des réseaux", "operation"),
    "RDCR": ("Retrait de la conduite des réseaux", "operation"),
    "TIR": ("Thermographie infrarouge", "TOOL ACTION"),
    "PO": ("Poste opérateur", "actor"),
    "RSD": ("Régime surveillance dispatching", "OPERATING MODE"),
    "SUAV": ("Sous tension à vide", "OPERATING CONDITIONS"),
    "MNV": ("Manoeuvre", "operation"),
    "PF": ("Point figé", "TOPOLOGY"),
    "CSS": ("Central sous station", "substation"),
    "GEH": ("Groupe exploitation hydraulique", "actor"),
    "PDM": ("Personnel de manoeuvre", "actor"),
    "SMACC": ("Système avancé pour la commande de la compensation", "OPERATING MODE"),
    "HO": ("Heures ouvrées", "datetime"),
    "BR": ("Bâtiment de relayage", "infrastructure"),
    "GT": ("Groupe de traction", "actor"),
    "TST": ("Travaux sous tensions", "network_event"),
    "CCO": ("Chargé de conduite", "actor"),
    "FDE": ("Filet d'essai", "OPERATING MODE"),
    "DIFFB": ("Protection différentielle de barre", "infrastructure"),
    "RADA": ("Retrait agile suite à détection d'anomalie", "network_event"),
    "TR": ("Transformateur", "infrastructure"),
    "RA": ("Réenclencheur automatique", "infrastructure"),
    "CTS": ("Cycle triphasé supplémentaire", "OPERATING CONDITIONS"),
    "CEX": ("Chargé d'exploitation", "actor"),
    "COSE": ("Centre opérationnel du système électrique", "center"),
    "COSE-P": ("Centre opérationnel du système électrique de Paris", "center"),
}

LABEL_GROUPS = {
    # Acteurs humains ou organisations
    "PER": "ACTOR",
    "ORG": "ACTOR",
    "actor": "ACTOR",
    "third-party_actor": "ACTOR",

    # Localisation
    "LOC": "GEO",
    "center": "GEO",

    # Temps
    "DATE": "DATETIME",
    "TIME": "DATETIME",
    "datetime": "DATETIME",

    # Événements
    "event": "EVENT",
    "external_event": "EVENT",
    "network_event": "EVENT",
    "MISC": "EVENT",

    # Infrastructures réseau
    "infrastructure": "INFRASTRUCTURE",
    "substation": "INFRASTRUCTURE",
    "LINE": "INFRASTRUCTURE",

    # Contexte opérationnel
    "operation": "OPERATING_CONTEXT",
    "TOPOLOGY": "OPERATING_CONTEXT",
    "OPERATING CONDITIONS": "OPERATING_CONTEXT",
    "OPERATING MODE": "OPERATING_CONTEXT",
    "STUDY": "OPERATING_CONTEXT",
    "TRANSIT": "OPERATING_CONTEXT",

    # Téléphone (inutile ici mais pour information)
    "PHONE_NUMBER": "PHONE_NUMBER",

    # Voltage et puissance (inutile ici mais pour information)
    "voltage_level": "ELECTRICAL_VALUE",
    "power": "ELECTRICAL_VALUE",
}


def group_labels(label):
    return LABEL_GROUPS.get(label, label)

# Traduction des abréviations


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
            translation, _ = abbr_dict[abbr]

            translated_text = translated_text[:start] + \
                translation + translated_text[end:]

    return translated_text


# Model
nlp = spacy.load("fr_core_news_lg")

# EntityRuler
ruler = nlp.add_pipe("entity_ruler", after="ner",
                     config={"overwrite_ents": True})

patterns = []

for (abbr_translation, original_labels) in ABBREVIATIONS_DICT.values():
    grouped_label = group_labels(original_labels)
    patterns.append({"label": grouped_label, "pattern": abbr_translation})


# Ajout de Regex dans l'EntityRuler
# Téléphone
patterns += [
    # +33 X XX XX XX XX
    {"label": "PHONE_NUMBER", "pattern": [
        {"TEXT": {"REGEX": r"^\+33$"}},
        {"TEXT": {"REGEX": r"^[1-9]$"}},
        {"TEXT": {"REGEX": r"^\d{2}$"}},
        {"TEXT": {"REGEX": r"^\d{2}$"}},
        {"TEXT": {"REGEX": r"^\d{2}$"}},
        {"TEXT": {"REGEX": r"^\d{2}$"}}
    ]},

    # 0XXXXXXXXX ou 0X.XX.XX.XX.XX ou 0X-XX-XX-XX-XX — mono-token
    {"label": "PHONE_NUMBER", "pattern": [
        {"TEXT": {"REGEX": r"^0[1-9](?:[ \.\-]?\d{2}){4}$"}}
    ]},


    # 0X XX XX XX XX
    {"label": "PHONE_NUMBER", "pattern": [
        {"TEXT": {"REGEX": r"^0[1-9]$"}},
        {"TEXT": {"REGEX": r"^\d{2}$"}},
        {"TEXT": {"REGEX": r"^\d{2}$"}},
        {"TEXT": {"REGEX": r"^\d{2}$"}},
        {"TEXT": {"REGEX": r"^\d{2}$"}}
    ]}
]

# Heures
patterns += [
    {"label": "DATETIME", "pattern": [
        {"TEXT": {
            "REGEX": r"^(?:[01]?\d|2[0-3])(?:\s?[hH](?:\s?[0-5]\d)?|:[0-5]\d|\s?heures?(?:\s[0-5]\d)?)$"}}
    ]}
]

# Dates
patterns += [
    # Dates numériques (JJ/MM/AAAA, AAAA-MM-JJ, JJ-MM-AA…)
    {"label": "DATETIME", "pattern": [
        {"TEXT": {
            "REGEX": r"(?i)^(?:(?:[0-3]?\d[\/\-.][0-1]?\d(?:[\/\-.]\d{2,4})?)|(?:\d{4}[\/\-.][0-1]?\d[\/\-.][0-3]?\d))$"}}
    ]},

    # Jour + mois + année optionnelle (ex. 15 juillet, 15 Juillet 2020)
    {"label": "DATETIME", "pattern": [
        {"TEXT": {"REGEX": r"(?i)^(?:[0-3]?\d(?:er)?)$"}},
        {"TEXT": {
            "REGEX": r"(?i)^(?:janv(?:\.|ier)?|févr(?:\.|ier)?|fevr(?:\.|ier)?|mars|avr(?:\.|il)?|mai|juin|juil(?:\.|let)?|ao[uû]t|sept(?:\.|embre)?|oct(?:\.|obre)?|nov(?:\.|embre)?|d[ée]c(?:\.|embre)?)$"}},
        {"TEXT": {"REGEX": r"(?i)^\d{2,4}$"}, "OP": "?"}
    ]},

    # Mois seul + année optionnelle (ex. Juillet 2020)
    {"label": "DATETIME", "pattern": [
        {"TEXT": {
            "REGEX": r"(?i)^(?:janv(?:\.|ier)?|févr(?:\.|ier)?|fevr(?:\.|ier)?|mars|avr(?:\.|il)?|mai|juin|juil(?:\.|let)?|ao[uû]t|sept(?:\.|embre)?|oct(?:\.|obre)?|nov(?:\.|embre)?|d[ée]c(?:\.|embre)?)$"}},
        {"TEXT": {"REGEX": r"(?i)^\d{2,4}$"}, "OP": "?"}
    ]},

    # Dates relatives et expressions temporelles
    {"label": "DATETIME", "pattern": [
        {"TEXT": {"REGEX": r"(?i)^(aujourd['’]?hui|demain|après[-\s]?demain|apres[-\s]?demain|hier|avant[-\s]?hier|(ce|cet)\s?(matin|soir|après[-\s]?midi|apres[-\s]?midi|week[-\s]?end)|(?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\s?(prochain|dernier)?|\d{1,2}\s?(AM|PM))$"}}
    ]},
]


# Voltage / Puissance
patterns += [
    {"label": "ELECTRICAL_VALUE", "pattern": [
        {"TEXT": {"REGEX": r"^\d+(?:[.,]\d+)?$"}},
        {"TEXT": {"REGEX": r"^(?:[kKmMgG]?[VvWw])$"}}
    ]}
]

# Lignes Ville–Ville
city = r"[A-ZÉÈÎÏÀÂÙÛÜŸ][a-zàâçéèêëîïôûùüÿ\-]+"
patterns += [
    {
        "label": "GEO",
        "pattern": [
            {"TEXT": {"REGEX": f"^{city}$"}},
            {"TEXT": "-"},
            {"TEXT": {"REGEX": f"^{city}$"}}
        ]
    }
]

ruler.add_patterns(patterns)

# Extraction d'entités


def extract_entities(text, abbr_dict=ABBREVIATIONS_DICT):
    """
    Pipeline complet : traduction des abréviations + extraction d'entités avec spaCy.
    """
    # Extraction des entités nommées par spaCy
    translated_text = translate_abbreviations(text, abbr_dict)
    doc = nlp(translated_text)

    results = {
        "original_text": text,
        "translated_text": translated_text,
        "entities": {}
    }

    for ent in doc.ents:
        grouped_label = group_labels(ent.label_)
        results["entities"].setdefault(grouped_label, []).append(ent.text)

    # Dédoublonner
    for key in results["entities"]:
        results["entities"][key] = list(set(results["entities"][key]))

    return results


if __name__ == "__main__":
    text = "Appeler Enedis au 07 28 39 83 23 pour RACR: incident au niveau du N-1 de Charles - Trappes."
    results = extract_entities(text)
    print(results)
