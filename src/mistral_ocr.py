import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY_TOM")
client = Mistral(api_key=api_key)

# ------ exemple simple avec une image en ligne ------

# ocr_response = client.ocr.process(
#     model="mistral-ocr-latest",
#     document={
#         "type": "image_url",
#         "image_url": "https://drive.google.com/drive/folders/1Y8SehDMb62f4IpMRZXGpDtFVkuv3goKG"
#     },
#     include_image_base64=True
# )

# print(ocr_response.pages[0].markdown)

# ------ on décrypte le texte d'une image locale ------

import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

# image_path = "images/IMG_3232.JPG"

# base64_image = encode_image(image_path)

# ocr_response = client.ocr.process(
#     model="mistral-ocr-latest",
#     document={
#         "type": "image_url",
#         "image_url": f"data:image/jpeg;base64,{base64_image}" 
#     },
#     include_image_base64=True
# )

# extracted_text = ocr_response.pages[0].markdown

# ------ pipeline pour analyser le texte d'images en local ------

image_folder = "images"
image_extensions = [".png", ".jpg", ".jpeg"]

extracted_text = ""

for filename in os.listdir(image_folder):
    if not any(filename.lower().endswith(ext) for ext in image_extensions):
        continue

    image_path = os.path.join(image_folder, filename)
    base64_image = encode_image(image_path)
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_image}"
        },
        include_image_base64=True
    )
    extracted_text += ocr_response.pages[0].markdown + "\n\n"

# ------ on demande au modèle Mistral de corriger le texte extrait en lui donnant du contexte ------

ocr_text = extracted_text

prompt = f"Contexte (désambiguïsation) :Réunion de kick-off du projet RTE (exploitation/pilotage du réseau). Participants : Bruno (R&D RTE), Philippe (interface projet côté RTE), Nael (Head of Data Illuin), équipe DTY. Sujet : prise de notes manuscrites des dispatcheurs, transmission entre quarts (6–8h), charge mentale, solutions « tapis/plaque » et « scanner de secours », feedback immédiat (LED), feuille A3 (schéma de quart), traçabilité, robustesse. Vocabulaire métier : dispatcheur, relève, incident, transfo, ligne, poste, SCADA, consignes, rappels, horodatage.Ta tâche :Nettoyer et corriger le texte OCR ci-dessous pour le rendre lisible et cohérent, **sans changer le sens ni ajouter/supprimer d’information**.Règles de correction (strictes) :- Corrige uniquement les artefacts d’OCR : accents, césures, espaces manquants/excessifs, lettres confondues (I/l/1, O/0), ponctuation, guillemets/apostrophes, mots collés ou éclatés, casse.- Préserve l’ordre des phrases, les retours à la ligne, la structure (titres, listes, puces, timestamps).- Conserve à l’identique les nombres, unités, acronymes et termes métier (ex. RTE, R&D, SCADA, A3, 225 kV, « quart », « relève », « dispatcheur », « transfo »).- Ne reformule pas, ne résume pas, n’invente rien.- En cas de doute sur un mot, **ne le modifie pas** (pas de devinette).Format de sortie :Rends **uniquement le texte corrigé**, en français, sans préambule ni commentaire. Attention, ce n'est pas starylines mais Storylines. Texte OCR à corriger :\"\"\"{ocr_text}\"\"\""""

response = client.chat.complete(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": prompt}]
)

corrected_text = response.choices[0].message.content

# ------ on demande au modèle Mistral de synthétiser le texte extrait en lui donnant du contexte ------

ocr_text = corrected_text

prompt = prompt = f"""
Contexte : Ce texte provient de notes corrigées issues d'une réunion de kick-off avec RTE. 
Participants : Bruno (responsable R&D de RTE), Philippe (interface projet côté RTE), Nael (head of data chez Illuin), et notre équipe DTY. 
Sujet de la réunion : lancement du projet RTE sur la prise de notes et la transmission d’informations pour les dispatcheurs réseau.

Tâche :
À partir du texte corrigé ci-dessous, rédige un compte rendu clair, exhaustif et bien structuré de la réunion. 

Instructions :
- Organise le texte en sections logiques : Contexte, Objectifs, Attentes exprimées, Contraintes, Points soulevés, Prochaines étapes. 
- Clarifie et homogénéise la formulation pour rendre le document fluide et professionnel. 
- N’ajoute aucune information non présente dans le texte, mais complète les phrases si nécessaire pour assurer la lisibilité.
- Style attendu : un compte rendu professionnel, neutre, concis mais exhaustif.

Texte corrigé :
\"\"\"{ocr_text}\"\"\"
"""

response = client.chat.complete(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": prompt}]
)

synthetised_text = response.choices[0].message.content

print(synthetised_text)


# ------ exporter le compte-rendu en PDF ------

# import markdown2
# from weasyprint import HTML

# def markdown_to_pdf(md_text, output_path="compte_rendu_reunion.pdf"):
#     # Convertir le markdown en HTML
#     html = markdown2.markdown(md_text, extras=["tables", "fenced-code-blocks"])
#     # Convertir le HTML en PDF
#     HTML(string=html).write_pdf(output_path)
#     print(f"PDF généré : {output_path}")

# markdown_to_pdf(synthetised_text)