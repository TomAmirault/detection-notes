# Format of the prompt sent to the Mistral model for entity detection

NER_PROMPT = """
    Tu es un extracteur d'entités strict en français.

    Règles absolues :
    - Tu ne dois **jamais inventer d'entités**.
    - N'extrais que des sous-chaînes **exactement présentes** dans le texte ci-dessous.
    - Les entités doivent être **formées de mots consécutifs** dans le texte d'origine, dans le même ordre.
    - N'ajoute aucune interprétation, reformulation ou élément typique du domaine s'il n'est pas écrit noir sur blanc.
    
    Les abréviations suivantes ont été traduites dans le texte en leur forme développée complète.  
    Elles doivent être considérées comme des **blocs indivisibles** : si une de ces expressions apparaît dans le texte, tu dois **toujours extraire l'expression entière**, jamais une sous-partie.

    {abbrev_str}
    
    Typiquement, les centres opérationnels sont des ACTOR (COSE suivis juste après d'une ville est un ACTOR entier). 
    
    Règle supplémentaire :
    - Si une **suite d’au moins 3 lettres majuscules consécutives** (ex. "NIPCCO", "ACR") apparaît dans le texte et **ne correspond à aucune autre catégorie identifiable** (ni ACTOR, ni INFRASTRUCTURE, ni autre), alors **et seulement dans ce cas**, tu dois l’extraire dans la catégorie **`ABBREVIATION_UNKNOWN`**.
    - Autrement dit, **`ABBREVIATION_UNKNOWN`** est une **catégorie de dernier recours**, utilisée uniquement pour les abréviations qui **n'ont aucun sens clair ou identifiable** dans le contexte.
    - Si une abréviation peut correspondre à une autre catégorie (ex. "SNCF", "EDF" ou "RTE" = "ACTOR"), **tu dois toujours la classer dans la catégorie la plus pertinente**, et **jamais dans `ABBREVIATION_UNKNOWN`**.

    Texte :
    \"{translated_text}\"

    Catégories :
    - GEO :
        Noms de lieux, villes, régions ou axes géographiques.
        Inclure les lieux composés ou reliés par des tirets, par exemple : "Ivry-sur-Seine", "Trappes-Deauville", "Paris-Lyon".
        Règle spéciale pour les tirets : lorsqu’un nom contient plusieurs villes reliées par un tiret (ex. "Ivry-sur-Seine-Saint-Etienne"), considère chaque ville séparément comme une entité GEO distincte.
        Exemple : "Ivry-sur-Seine-Saint-Etienne" --> deux entités GEO : "Ivry-sur-Seine" et "Saint-Etienne".
    - ACTOR :
        Noms d'acteurs (personnes, équipes, entités organisationnelles).
    - DATETIME :
        Dates ou horaires, y compris formats naturels (ex. "10h30", "20 octobre 2025").
    - EVENT :
        Événements spécifiques, par exemple incidents, manœuvres, interventions.
    - INFRASTRUCTURE :
        Noms ou identifiants d'ouvrages électriques, lignes, postes.
    - OPERATING_CONTEXT :
        Contexte opérationnel, incluant ordres, statuts ou conditions particulières.
    - PHONE_NUMBER :
        Numéros de téléphone.
    - ELECTRICAL_VALUE :
        Valeurs électriques, comme "225kV", "63 kV", "400 MW", etc.
    - ABBREVIATION_UNKNOWN :  
        Abréviations en lettres majuscules (au moins 3 caractères) **qui ne correspondent à aucune autre catégorie identifiable** et dont le sens **n’est pas interprétable dans le contexte** (si leur signification n’est pas reconnue).

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
    "ELECTRICAL_VALUE": [],
    "ABBREVIATION_UNKNOWN": []
    }}
    
    Quelques exemples de mots par catégories : 
    {{ "GEO": ["Caen", "Chartres", "Paris"], 
    "ACTOR": ["SNCF, Enedis", "Chargé d'exploitation"], 
    "DATETIME": ["15 juillet 2023", "ce matin", "demain", "14h", "18:00", "12/03/2020"], 
    "EVENT": ["Manoeuvre", "Incident", "Retrait", "Bascule"], 
    "INFRASTRUCTURE": ["Ligne", "N-1", "Transformateur"], 
    "OPERATING_CONTEXT": ["invalide en HO", "Régime surveillance dispatching"], 
    "PHONE_NUMBER": ["06 12 34 56 78", "+33 9 38 39 29 30"], 
    "ELECTRICAL_VALUE": ["225kV", "38 kV", "400 GW"],
    "ABBREVIATION_UNKNOWN": ["NIPCCO", "ACR"]}} 
    """