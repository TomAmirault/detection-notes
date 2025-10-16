import os
from mistralai import Mistral
from dotenv import load_dotenv

# Charger la clé API Mistral
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

def nettoyer_transcription_audio(texte: str) -> str:
    """
    Nettoie un texte brut issu d'une transcription audio.
    - Corrige les fautes évidentes
    - Restaure une ponctuation minimale
    - Ne change pas le sens ni l'ordre
    - Ne reformule pas
    """
    prompt = f"""Tu es un assistant de normalisation de texte issu d’une transcription audio de chez RTE (Réseau Transport Electricité).

    Objectif :
    Nettoyer la transcription tout en conservant fidèlement le contenu et l’ordre des mots.

    RÈGLES STRICTES :
    1) Ne change pas l’ordre du texte ni les phrases.
    2) Corrige uniquement les erreurs évidentes :
       - Fautes d’orthographe simples.
       - Espaces en trop ou manquants.
       - Mots collés ou séparés à tort.
       - Ponctuation minimale (virgules, points, majuscules au début de phrase).
       - Apostrophes ou accents oubliés.
    3) Ne reformule pas. Ne paraphrase pas.
    4) N’ajoute rien, ne commente pas, ne mets pas de balises ou de markdown.
    5) Si le texte ne contient aucune information exploitable, renvoie une chaîne vide.
    6) Abréviations officielles (ne pas développer ; corrige variantes proches vers la forme officielle) :
     SNCF, ABC, RSD, TIR, PF, GEH, SMACC, COSE, TRX, VPL, MNV, N-1, COSE-P

    7) Noms de villes françaises :
    Corrige les noms de villes françaises mal transcrits vers leur forme correcte officielle.
    Exemple : 
    "parie" → "Paris", 
    "lion" → "Lyon", 
    "gre noble" → "Grenoble",
    "nant" → "Nantes",
    "cean" → "Caen",
    "cher bour" → "Cherbourg",
    "vanne" → "Vannes"
   
    8) Formats déterministes :
    - Heures : “16 h” ou “16h” → “16h”
    - Numéros de téléphone : supprimer espaces
    - Tensions : normaliser en kV
    - cost → COSE

    Exemple :
    Entrée :
    "prévenir euuh monsieur martin ancien zéro sept six six trente sept zéro deux quatre sept nouveau zéro sept soixante six trente sept huit deux quatre sept appel privé avec joan maintenace vérifier planning travau confirmation travaux demain huit heure t quatre t deux t trois envoyer c r à c m à seize heure
"

    Sortie :
    "Prévenir M. Martin ancien: 0766370247
    Nouveau: 0766378247
    Appel privé avec Jean Maintenance vérifier planning travaux
    Confirmation travaux demain 8h
    T4 T2 T3
    Envoyer CR à CM à 16h"

    Texte à nettoyer :
    <<<
    {texte}
    >>>
    """

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    clean_text = response.choices[0].message.content.strip()
    return clean_text


# Exemple d'utilisation
if __name__ == "__main__":
    transcription = """
    alors on a fait la maintenance du poste t quatre ce matin
    on a aussi verifié le relais principal et on a rien detecté d'anormal
    """
    print(nettoyer_transcription_audio(transcription))
