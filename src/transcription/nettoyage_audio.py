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
    prompt = f"""Tu es un assistant chargé de corriger légèrement un texte issu d’une transcription audio.

RÈGLES DE CORRECTION :

1) Ne modifie **ni le sens**, **ni l’ordre des mots**.  
2) Corrige uniquement les **erreurs évidentes** :
   - Orthographe simple.  
   - Espaces manquants ou en trop.  
   - Ponctuation minimale (points, virgules, majuscules).  
   - Apostrophes et accents oubliés.  

3) Améliore la **lisibilité** :
   - Une idée ou une phrase = une nouvelle ligne.  
   - Ne colle pas tout sur une seule ligne.  
   - N’utilise **jamais** les balises de code (```, ```python, etc.).  

4) Ne reformule pas et **n’ajoute aucun mot**.  
5) Si le texte est vide ou incompréhensible, renvoie **une chaîne vide**.  
6) Si le texte est dans une autre langue, **traduis-le en français simple** sans changer le sens.  

7) **Abréviations officielles** :  
   - Ne pas développer les abréviations.  
   - Corrige les variantes proches vers la forme officielle.  
   - Liste des abréviations à respecter :  
     "RACR, RDCR, TIR, PO, RSD, SUAV, MNV, PF, CSS, GEH, PDM, SMACC, HO, BR, GT, TST, CCO, FDE, DIFFB, RADA, TR, RA, CTS, CEX, COSE, COSE-P, SNCF, ABC, TRX, VPL, N-1"


8) **Noms de villes françaises** :  
   - Corrige les noms de villes mal transcrits vers leur forme correcte officielle.  
   Exemples :  
     "parie" → "Paris"  
     "lion" → "Lyon"  
     "gre noble" → "Grenoble"  
     "nant" → "Nantes"  
     "cean" → "Caen"  
     "cher bour" → "Cherbourg"  
     "vanne" → "Vannes"  

9) **Formats déterministes** :
   - Heures : “16 h” ou “16h” → “16h”.  
   - Numéros de téléphone : **supprime les espaces**.  
   - Tensions : **normalise en kV**.  
   - “cost” → **COSE**.  

Texte à corriger :
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






