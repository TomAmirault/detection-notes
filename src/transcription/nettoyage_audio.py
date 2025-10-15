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
    prompt = f"""Tu es un assistant de normalisation de texte issu d’une transcription audio.

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

    Exemple :
    Entrée :
    "bonjour euh aujourd'hui on a verifié le poste t quatre et on a detecté un bruit anormal sur le transfo deux"

    Sortie :
    "Bonjour, aujourd'hui on a vérifié le poste T4 et on a détecté un bruit anormal sur le transformateur 2."

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
