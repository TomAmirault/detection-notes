import os
from mistralai import Mistral
from dotenv import load_dotenv

# Charger la clé API Mistral
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

def synthèse(texte: str) -> str:
    """
    Nettoie un texte brut issu d'une transcription audio.
    - Corrige les fautes évidentes
    - Restaure une ponctuation minimale
    - Ne change pas le sens ni l'ordre
    - Ne reformule pas
    """
    prompt = f"""Tu es un assistant spécialisé dans la synthèse de documents professionnels pour l'entreprise RTE. 
Tu disposes de notes manuscrites et de transcriptions audio. Ces notes contiennent des informations opérationnelles, des appels à prévoir, des actions à réaliser et des incidents à signaler.

Ta tâche est de produire une **synthèse très courte, directe et complète**, en conservant toutes les informations importantes. La synthèse doit : 
- Indiquer les actions à effectuer et leur priorité.
- Mentionner les appels ou contacts à prévoir avec les horaires.
- Signaler tout incident ou événement particulier.
- Résumer les changements importants (ex : changement de propriétaire, modifications de liaison).

Voici les notes : 
    <<<
    {texte}
    >>>
Rédige un texte **très concis**, sous forme de **phrases complètes et explicatives**, en éliminant tout détail superflu, pour que l’équipe opérationnelle puisse immédiatement comprendre les actions à prendre et les informations essentielles. **Priorise la clarté et l’efficacité du message.**
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
~ Ligne 1. Prévoir retrait de la liaison Caen-Cherbourg
~ Ligne 3. Bien sûr Thierry
+ Ligne 2. Appel COSE-P à prévoir
+ Ligne 4. Appeler le COSE Nantes à 19h27
+ Ligne 5. Changement de propriétaire
+ Ligne 6. Chacal détecté mai
    """
    print(synthèse(transcription))
