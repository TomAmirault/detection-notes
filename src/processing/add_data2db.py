# 

import json
import sys
from difflib import SequenceMatcher

# Ajoute la racine du projet au sys.path pour permettre les imports internes
import sys
import os
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from src.processing.mistral_ocr_llm import image_transcription
# from src.processing.teklia_ocr_llm import image_transcription

from src.backend.db import (
    DB_PATH,
    insert_note_meta,
    get_last_text_for_notes,
    find_similar_note ,
    find_similar_image
)
from src.ner.llm_extraction import extract_entities
from src.utils.text_utils import (
    has_meaningful_line,
    has_meaningful_text,
    compute_diff,
    is_htr_buggy,
    clean_added_text,
    score_and_categorize_texts,
    reflow_sentences,
)
from src.utils.image_utils import encode_image

AUD_COUNTER = 0
TEXT_COUNTER = 0

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

    # Précheck : comparaison visuelle des images
    if find_similar_image(image_path, db_path) is not None:
        print("L'image captée a déjà été enregistrée en BBD.")
        return None

    # 1) OCR + normalisation
    ocr_text, cleaned_text, confidence_score = image_transcription(image_path)

    # >>> Pare-feu avant toute logique de DB
    buggy, reason = is_htr_buggy(ocr_text, cleaned_text)
    if buggy:
        print(f"[SKIP][HTR-BUG] {reason} pour {image_path}")
        return None

    if not cleaned_text or not cleaned_text.strip():
        print(
            f"[SKIP] Aucun texte exploitable après normalisation pour {image_path}")
        return None

    # consider literal empty-quoted results as empty (e.g. '""' or "''")
    if cleaned_text.strip() in ('""', "''"):
        print(f"[SKIP] Transcription nettoyée vide (literal quotes) pour {image_path}")
        return None

    # 2) Cherche une note existante similaire

    similar_note_id = find_similar_note(
        cleaned_text, db_path=db_path, threshold=0.7)

    try:
        last_texts = get_last_text_for_notes(db_path)
    except Exception:
        last_texts = {}


    # 1) Cas des notes courtes 



    for nid, prev_text in (last_texts or {}).items():
        s_prev = reflow_sentences(prev_text or "", width=80)
        s_new = reflow_sentences(cleaned_text or "", width=80)
        score_info = score_and_categorize_texts(s_prev, s_new)

        # print(f"""score_F1_Jacquard={score_info['score']} \n
        #       score_seqm={SequenceMatcher(None, s_prev, s_new).ratio()} \n
        #             Note de la BDD \n
        #             {len(s_prev)}]{s_prev} \n
        #             est très similaire à la nouvelle note \n
        #             {len(s_new)}{s_new}""")

        if len(s_prev) - len(s_new) < 35 and len(s_new) - len(s_prev) < 35 and SequenceMatcher(None, s_prev, s_new).ratio() > 0.5 :
            print(f"""Brigade anti-répétition : note similaire trouvée en BDD avec score : {SequenceMatcher(None, s_prev, s_new).ratio()} \n
                  Note de la BDD \n
             {len(s_prev)}]{s_prev} \n
             est très similaire à la nouvelle note \n
             {len(s_new)}{s_new}""")
            None
    
    

    diff_human = ""
    diff_json = []

    if similar_note_id:
        # même feuille → calcul des vraies nouveautés
        last_texts = get_last_text_for_notes(db_path)
        old_text = last_texts.get(similar_note_id, "")
        diff_human, diff_json = compute_diff(old_text, cleaned_text, minor_change_threshold=0.90)
        # print("=== DIFF HUMAIN ===")
        # print(diff_human)
        # print("=== DIFF JSON ===")
        # print(diff_json)

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
        global TEXT_COUNTER
        TEXT_COUNTER += 1
        note_id = f"TEXT-{TEXT_COUNTER}"
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
        cleaned_diff_human = clean_added_text(diff_human)
        entities = extract_entities(cleaned_diff_human)
    else:
        entities = {}

    extracted_data = {
        "note_id": note_id,
        "transcription_brute": ocr_text,        # <— OCR brut
        "transcription_clean": cleaned_text,     # <— texte normalisé stable
        "texte_ajoute": diff_human,
        "confidence_score": confidence_score,
        "img_path_proc": image_path,
        "raw_json": json.dumps(raw, ensure_ascii=False),
        "entite_GEO": json.dumps(entities.get("GEO", []), ensure_ascii=False),
        "entite_ACTOR": json.dumps(entities.get("ACTOR", []), ensure_ascii=False),
        "entite_DATETIME": json.dumps(entities.get("DATETIME", []), ensure_ascii=False),
        "entite_EVENT": json.dumps(entities.get("EVENT", []), ensure_ascii=False),
        "entite_INFRASTRUCTURE": json.dumps(entities.get("INFRASTRUCTURE", []), ensure_ascii=False),
        "entite_OPERATING_CONTEXT": json.dumps(entities.get("OPERATING_CONTEXT", []), ensure_ascii=False),
        "entite_PHONE_NUMBER": json.dumps(entities.get("PHONE_NUMBER", []), ensure_ascii=False),
        "entite_ELECTRICAL_VALUE": json.dumps(entities.get("ELECTRICAL_VALUE", []), ensure_ascii=False),
        "entite_ABBREVIATION_UNKNOWN": json.dumps(entities.get("ABBREVIATION_UNKNOWN", []), ensure_ascii=False),
    }

    meta_id = insert_note_meta(
        extracted_data, img_path_proc=image_path, db_path=db_path)
    print(f"Note insérée (note_id {note_id}, meta_id {meta_id})")
    return meta_id


def add_audio2db(audio_path: str, transcription_brute: str, transcription_clean: str, db_path: str = DB_PATH):
    """
    Insert an audio transcription as a notes_meta entry.

    Policy: each audio segment is treated as an "ajout" (new line). We create a new note_id
    per audio segment so the front displays each audio as its own entry; evenement grouping
    is still handled by the DB via entity matching.
    """
    # Ensure we have some text
    # Normalize transcription_clean: remove surrounding quotes if present
    def strip_surrounding_quotes_local(s: str) -> str:
        if s is None:
            return s
        s = s.strip()
        while len(s) >= 2 and ((s[0] == s[-1] and s[0] in ('"', "'"))):
            s = s[1:-1].strip()
        return s

    if not transcription_clean or not transcription_clean.strip() or transcription_clean.strip() in ('""', "''"):
        print(f"[SKIP] audio {audio_path} has no cleaned transcription (empty or literal quotes)")
        return None
    transcription_clean = strip_surrounding_quotes_local(transcription_clean)
    
    # Pare-feu HTR / anti-bruit pour audio
    buggy, reason = is_htr_buggy(transcription_brute or "", transcription_clean or "")
    if buggy:
        print(f"[SKIP][HTR-BUG] audio {audio_path}: {reason}")
        return None

    if not has_meaningful_text(transcription_clean):
        print(f"[SKIP] audio {audio_path} : transcription non-significative (anti-bruit)")
        return None


    # Prepare diff/texte_ajoute for audio: each audio is a single added line
    diff_human = f"+ Ligne 1. {transcription_clean.strip()}"
    diff_json = [{"type": "insert", "line": 1, "content": transcription_clean.strip()}]

    # Extract entities from cleaned transcription
    cleaned_for_ner = clean_added_text(diff_human)
    entities = extract_entities(cleaned_for_ner) if cleaned_for_ner else {}

    global AUD_COUNTER
    AUD_COUNTER += 1
    note_id = f"AUD-{AUD_COUNTER}"

    raw = {
        "source": "audio-wav2vec2",
        "audio_path": audio_path,
        "diff": diff_json,
    }

    extracted_data = {
        "note_id": note_id,
        "transcription_brute": transcription_brute,
        "transcription_clean": transcription_clean,
        "texte_ajoute": diff_human,
        "confidence_score": 0.5,
        "img_path_proc": None,
        "raw_json": json.dumps(raw, ensure_ascii=False),
        "entite_GEO": json.dumps(entities.get("GEO", []), ensure_ascii=False),
        "entite_ACTOR": json.dumps(entities.get("ACTOR", []), ensure_ascii=False),
        "entite_DATETIME": json.dumps(entities.get("DATETIME", []), ensure_ascii=False),
        "entite_EVENT": json.dumps(entities.get("EVENT", []), ensure_ascii=False),
        "entite_INFRASTRUCTURE": json.dumps(entities.get("INFRASTRUCTURE", []), ensure_ascii=False),
        "entite_OPERATING_CONTEXT": json.dumps(entities.get("OPERATING_CONTEXT", []), ensure_ascii=False),
        "entite_PHONE_NUMBER": json.dumps(entities.get("PHONE_NUMBER", []), ensure_ascii=False),
        "entite_ELECTRICAL_VALUE": json.dumps(entities.get("ELECTRICAL_VALUE", []), ensure_ascii=False),
        "entite_ABBREVIATION_UNKNOWN": json.dumps(entities.get("ABBREVIATION_UNKNOWN", []), ensure_ascii=False),
    }

    meta_id = insert_note_meta(extracted_data, img_path_proc=None, db_path=db_path)
    print(f"Audio inséré (note_id {note_id}, meta_id {meta_id})")
    return meta_id