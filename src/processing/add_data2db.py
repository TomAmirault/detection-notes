"""
Database ingestion module for OCR/HTR and ASR transcriptions.

This module handles the insertion of processed notes (from images or audio) into the SQLite database,
including deduplication, diff computation, entity extraction, and metadata management.
"""

import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

# Add repository root to sys.path for internal imports
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from logger_config import setup_logger
from src.backend.db import (
    DB_PATH,
    find_similar_image,
    find_similar_note,
    get_last_text_for_notes,
    insert_note_meta,
)
from src.ner.llm_extraction import extract_entities
from src.processing.mistral_ocr_llm import image_transcription
from src.utils.image_utils import encode_image
from src.utils.text_utils import (
    clean_added_text,
    compute_diff,
    has_meaningful_line,
    has_meaningful_text,
    is_htr_buggy,
    reflow_sentences,
    score_and_categorize_texts,
)

logger = setup_logger(__name__)

# Global counters for generating unique note IDs
AUD_COUNTER: int = 0
TEXT_COUNTER: int = 0


def add_data2db(image_path: str | Path, db_path: str | Path = DB_PATH) -> Optional[int]:
    """
    Process an image with OCR/HTR, compute diffs against existing notes, and insert into database.

    This function performs the following workflow:
    1. Check if image already exists in database (visual similarity)
    2. Run OCR + LLM normalization on the image
    3. Apply quality filters (HTR bugs, empty text, noise detection)
    4. Find similar existing notes (same physical sheet)
    5. Compute text diffs and filter out non-meaningful changes
    6. Extract named entities from new content
    7. Insert note metadata into database

    Args:
        image_path: Path to the input image file.
        db_path: Path to the SQLite database file.

    Returns:
        The meta_id of the inserted record, or None if the image was skipped.

    Raises:
        Exception: Database insertion errors are propagated from insert_note_meta.
    """
    global TEXT_COUNTER

    image_path_str = str(image_path)
    db_path_str = str(db_path)

    # Visual pre-check: avoid re-processing identical images
    if find_similar_image(image_path_str, db_path_str) is not None:
        logger.info(f"Image already recorded in database: {image_path_str}")
        return None

    # Run OCR and LLM-based text normalization
    ocr_text, cleaned_text, confidence_score = image_transcription(image_path_str)

    # Quality firewall: detect and reject buggy HTR output
    buggy, reason = is_htr_buggy(ocr_text, cleaned_text)
    if buggy:
        logger.warning(f"[SKIP][HTR-BUG] {reason} for {image_path_str}")
        return None

    # Reject empty or trivially empty transcriptions
    if not cleaned_text or not cleaned_text.strip():
        logger.info(f"[SKIP] No exploitable text after normalization for {image_path_str}")
        return None

    if cleaned_text.strip() in ('""', "''"):
        logger.info(f"[SKIP] Cleaned transcription is empty (literal quotes) for {image_path_str}")
        return None

    # Search for an existing similar note (same physical sheet)
    similar_note_id = find_similar_note(cleaned_text, db_path=db_path_str, threshold=0.7)

    # Retrieve last known text for all notes
    try:
        last_texts = get_last_text_for_notes(db_path_str)
    except Exception as e:
        logger.warning(f"Could not retrieve last texts from database: {e}")
        last_texts = {}

    # Anti-repetition brigade: detect near-duplicate short notes across all existing notes
    for nid, prev_text in (last_texts or {}).items():
        s_prev = reflow_sentences(prev_text or "", width=80)
        s_new = reflow_sentences(cleaned_text or "", width=80)
        score_info = score_and_categorize_texts(s_prev, s_new)

        length_diff = abs(len(s_prev) - len(s_new))
        similarity_ratio = SequenceMatcher(None, s_prev, s_new).ratio()

        if length_diff < 35 and similarity_ratio > 0.5:
            logger.info(
                f"Anti-repetition: similar note found in database (score={similarity_ratio:.2f})\n"
                f"DB note [{len(s_prev)} chars]: {s_prev}\n"
                f"New note [{len(s_new)} chars]: {s_new}"
            )
            return None

    diff_human = ""
    diff_json: list[dict[str, str | int]] = []

    # Determine if this is a new note or an update to an existing one
    if similar_note_id:
        # Same sheet: compute actual new content
        old_text = last_texts.get(similar_note_id, "")
        diff_human, diff_json = compute_diff(
            old_text, cleaned_text, minor_change_threshold=0.90
        )

        if not diff_human.strip():
            logger.info(
                f"No meaningful novelty for note {similar_note_id}. Skipping insertion."
            )
            return None

        if not has_meaningful_line(diff_human):
            logger.info(f"[SKIP] Diff has no meaningful content for note {similar_note_id}")
            return None

        if not has_meaningful_text(cleaned_text):
            logger.info(f"[SKIP] No exploitable text (anti-noise) for {image_path_str}")
            return None

        note_id = similar_note_id
        logger.info(f"New version for existing note {note_id}")

    else:
        # New sheet: create new note_id and treat all lines as additions
        TEXT_COUNTER += 1
        note_id = f"TEXT-{TEXT_COUNTER}"
        lines = [line for line in cleaned_text.splitlines() if line.strip()]
        diff_human = "\n".join(f"+ Ligne {i+1}. {line}" for i, line in enumerate(lines))
        diff_json = [
            {"type": "insert", "line": i + 1, "content": line}
            for i, line in enumerate(lines)
        ]
        logger.info(f"New note created with id {note_id}")

    # Prepare raw metadata for database storage
    raw = {
        "source": "mistral-ocr-latest + mistral-large-latest",
        "image_path": image_path_str,
        "diff": diff_json,
    }

    # Extract named entities from new content
    if diff_human.strip():
        cleaned_diff_human = clean_added_text(diff_human)
        entities = extract_entities(cleaned_diff_human)
    else:
        entities = {}

    # Assemble structured data for database insertion
    extracted_data = {
        "note_id": note_id,
        "transcription_brute": ocr_text,
        "transcription_clean": cleaned_text,
        "texte_ajoute": diff_human,
        "confidence_score": confidence_score,
        "img_path_proc": image_path_str,
        "raw_json": json.dumps(raw, ensure_ascii=False),
        "entite_GEO": json.dumps(entities.get("GEO", []), ensure_ascii=False),
        "entite_ACTOR": json.dumps(entities.get("ACTOR", []), ensure_ascii=False),
        "entite_DATETIME": json.dumps(entities.get("DATETIME", []), ensure_ascii=False),
        "entite_EVENT": json.dumps(entities.get("EVENT", []), ensure_ascii=False),
        "entite_INFRASTRUCTURE": json.dumps(
            entities.get("INFRASTRUCTURE", []), ensure_ascii=False
        ),
        "entite_OPERATING_CONTEXT": json.dumps(
            entities.get("OPERATING_CONTEXT", []), ensure_ascii=False
        ),
        "entite_PHONE_NUMBER": json.dumps(
            entities.get("PHONE_NUMBER", []), ensure_ascii=False
        ),
        "entite_ELECTRICAL_VALUE": json.dumps(
            entities.get("ELECTRICAL_VALUE", []), ensure_ascii=False
        ),
        "entite_ABBREVIATION_UNKNOWN": json.dumps(
            entities.get("ABBREVIATION_UNKNOWN", []), ensure_ascii=False
        ),
    }

    # Insert into database
    meta_id = insert_note_meta(
        extracted_data, img_path_proc=image_path_str, db_path=db_path_str
    )
    logger.info(f"Note inserted (note_id={note_id}, meta_id={meta_id})")
    return meta_id


def add_audio2db(
    audio_path: str | Path,
    transcription_brute: str,
    transcription_clean: str,
    db_path: str | Path = DB_PATH,
) -> Optional[int]:
    """
    Insert an audio transcription as a notes_meta entry in the database.

    Each audio segment is treated as a new note (independent line) to ensure proper display
    in the front-end. Event grouping is handled separately via entity matching.

    Args:
        audio_path: Path to the audio file.
        transcription_brute: Raw ASR output before normalization.
        transcription_clean: LLM-normalized transcription text.
        db_path: Path to the SQLite database file.

    Returns:
        The meta_id of the inserted record, or None if the audio was skipped.

    Raises:
        Exception: Database insertion errors are propagated from insert_note_meta.
    """
    global AUD_COUNTER

    audio_path_str = str(audio_path)
    db_path_str = str(db_path)

    # Normalize transcription_clean: remove surrounding quotes if present
    def strip_surrounding_quotes_local(s: str) -> str:
        if s is None:
            return s
        s = s.strip()
        while len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
            s = s[1:-1].strip()
        return s

    # Reject empty or trivially empty transcriptions
    if (
        not transcription_clean
        or not transcription_clean.strip()
        or transcription_clean.strip() in ('""', "''")
    ):
        logger.info(
            f"[SKIP] Audio {audio_path_str} has no cleaned transcription (empty or literal quotes)"
        )
        return None

    transcription_clean = strip_surrounding_quotes_local(transcription_clean)

    # Quality firewall: detect and reject buggy ASR output
    buggy, reason = is_htr_buggy(transcription_brute or "", transcription_clean or "")
    if buggy:
        logger.warning(f"[SKIP][HTR-BUG] Audio {audio_path_str}: {reason}")
        return None

    if not has_meaningful_text(transcription_clean):
        logger.info(
            f"[SKIP] Audio {audio_path_str}: transcription is non-meaningful (anti-noise)"
        )
        return None

    # Prepare diff/texte_ajoute: treat entire audio as a single added line
    diff_human = f"+ Ligne 1. {transcription_clean.strip()}"
    diff_json: list[dict[str, str | int]] = [
        {"type": "insert", "line": 1, "content": transcription_clean.strip()}
    ]

    # Extract named entities from cleaned transcription
    cleaned_for_ner = clean_added_text(diff_human)
    entities = extract_entities(cleaned_for_ner) if cleaned_for_ner else {}

    # Generate unique note_id for this audio segment
    AUD_COUNTER += 1
    note_id = f"AUD-{AUD_COUNTER}"

    # Prepare raw metadata for database storage
    raw = {
        "source": "audio-wav2vec2",
        "audio_path": audio_path_str,
        "diff": diff_json,
    }

    # Assemble structured data for database insertion
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
        "entite_INFRASTRUCTURE": json.dumps(
            entities.get("INFRASTRUCTURE", []), ensure_ascii=False
        ),
        "entite_OPERATING_CONTEXT": json.dumps(
            entities.get("OPERATING_CONTEXT", []), ensure_ascii=False
        ),
        "entite_PHONE_NUMBER": json.dumps(
            entities.get("PHONE_NUMBER", []), ensure_ascii=False
        ),
        "entite_ELECTRICAL_VALUE": json.dumps(
            entities.get("ELECTRICAL_VALUE", []), ensure_ascii=False
        ),
        "entite_ABBREVIATION_UNKNOWN": json.dumps(
            entities.get("ABBREVIATION_UNKNOWN", []), ensure_ascii=False
        ),
    }

    # Insert into database
    meta_id = insert_note_meta(extracted_data, img_path_proc=None, db_path=db_path_str)
    logger.info(f"Audio inserted (note_id={note_id}, meta_id={meta_id})")
    return meta_id