import json
import uuid
import sys
import os
import re

# Ajout du dossier src au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# fmt: off
# isort: skip
from src.processing.mistral_ocr_llm import image_transcription
from src.backend.db import (
    DB_PATH,
    insert_note_meta,
    get_last_text_for_notes,
    find_similar_note 
)
from src.processing.mistral_ocr_llm import image_transcription
from ner.spacy_model import extract_entities
# fmt on  


from src.utils.text_utils import has_meaningful_line, has_meaningful_text, compute_diff, is_htr_buggy, clean_added_text_for_ner

from src.utils.image_utils import encode_image


def add_data2db(image_path: str, db_path: str = DB_PATH):
    """
    Workflow :
    1) OCR + normalisation stable (LLM)
    2) Cherche une note similaire (même feuille)
    3) Si similaire :
         - calcule les vraies nouveautés (lignes ajoutées / changées)
         - si rien de nouveau => ignore
         - sinon => insère avec le même note_id
       Sinon :
         - crée un nouveau note_id et insère tout le texte (comme ajout initial)
    """
    # 0) Encodage de l'image en base64 (pour Mistral OCR)
    encoded_image = encode_image(image_path)

    # 1) OCR + normalisation
    ocr_text, cleaned_text = image_transcription(encoded_image)

    # >>> Pare-feu avant toute logique de DB
    buggy, reason = is_htr_buggy(ocr_text, cleaned_text)
    if buggy:
        print(f"[SKIP][HTR-BUG] {reason} pour {image_path}")
        return None

    if not cleaned_text or not cleaned_text.strip():
        print(
            f"[SKIP] Aucun texte exploitable après normalisation pour {image_path}")
        return None

    # 2) Cherche une note existante similaire

    similar_note_id = find_similar_note(
        cleaned_text, db_path=db_path, threshold=0.7)

    diff_human = ""
    diff_json = []

    if similar_note_id:
        # même feuille → calcul des vraies nouveautés
        last_texts = get_last_text_for_notes(db_path)
        old_text = last_texts.get(similar_note_id, "")
        diff_human, diff_json = compute_diff(old_text, cleaned_text, minor_change_threshold=0.90)
        print("=== DIFF HUMAIN ===")
        print(diff_human)
        print("=== DIFF JSON ===")
        print(diff_json)

        if not diff_human.strip():
            print(
                f"Aucune vraie nouveauté pour la note {similar_note_id}. Ignorée.")
            return None

        if not has_meaningful_line(diff_human):
            print(f"[SKIP] Diff sans contenu utile pour note {similar_note_id}")
            return None

        if not has_meaningful_text(cleaned_text):
            print(f"[SKIP] Aucun texte exploitable (anti-bruit) pour {image_path}")
            return None

        note_id = similar_note_id
        print(f"Nouvelle version pour la note existante {note_id}")

    else:
        # nouvelle feuille
        note_id = str(uuid.uuid4())
        lines = [l for l in cleaned_text.splitlines() if l.strip()]
        diff_human = "\n".join(
            f"+ Ligne {i+1}. {l}" for i, l in enumerate(lines))
        diff_json = [{"type": "insert", "line": i+1, "content": l}
                     for i, l in enumerate(lines)]
        print(f"Nouvelle note créée avec id {note_id}")

    # 3) Insertion en DB
    raw = {
        "source": "mistral-ocr-latest + mistral-large-latest",
        "image_path": image_path,
        "diff": diff_json,
    }

    # Extraction d'entités
    if diff_human.strip():
        cleaned_diff_human = clean_added_text_for_ner(diff_human)
        entities = extract_entities(cleaned_diff_human)
    else:
        entities = {}

    extracted_data = {
        "note_id": note_id,
        "transcription_brute": ocr_text,        # <— OCR brut
        "transcription_clean": cleaned_text,     # <— texte normalisé stable
        "texte_ajoute": diff_human,
        "img_path_proc": image_path,
        "images": [],
        "raw_json": json.dumps(raw, ensure_ascii=False),
        "entite_GEO": json.dumps(entities.get("GEO", []), ensure_ascii=False),
        "entite_ACTOR": json.dumps(entities.get("ACTOR", []), ensure_ascii=False),
        "entite_DATETIME": json.dumps(entities.get("DATETIME", []), ensure_ascii=False),
        "entite_EVENT": json.dumps(entities.get("EVENT", []), ensure_ascii=False),
        "entite_INFRASTRUCTURE": json.dumps(entities.get("INFRASTRUCTURE", []), ensure_ascii=False),
        "entite_OPERATING_CONTEXT": json.dumps(entities.get("OPERATING_CONTEXT", []), ensure_ascii=False),
        "entite_PHONE_NUMBER": json.dumps(entities.get("PHONE_NUMBER", []), ensure_ascii=False),
        "entite_ELECTRICAL_VALUE": json.dumps(entities.get("ELECTRICAL_VALUE", []), ensure_ascii=False),
    }

    meta_id = insert_note_meta(
        extracted_data, img_path_proc=image_path, db_path=db_path)
    print(f"Note insérée (note_id {note_id}, meta_id {meta_id})")
    return meta_id


import os
folder = "/Users/tomamirault/Documents/projects/p1-dty-rte/vertical-attention-network-for-handwritten-text-recognition/data/raw"
for filename in os.listdir(folder):
    if filename.lower().endswith(('jpg')):
        image_path = os.path.join(folder, filename)
        add_data2db(image_path)

