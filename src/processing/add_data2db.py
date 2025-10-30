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

# Choose which OCR/HTR module to use between Mistral and Teklia :
#from src.processing.mistral_ocr_llm import image_transcription
from src.processing.teklia_ocr_llm import image_transcription

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

# =============================================================================
# Configuration Constants

# These parameters control the behavior of note deduplication, diff computation,
# and quality filtering. Adjust them to fine-tune the system's sensitivity.

# Note similarity detection (find_similar_note)
# Controls how similar two texts must be to be considered the same physical note sheet.
# Range: 0.0-1.0 where 1.0 = identical text required
# Higher values = stricter matching (fewer false positives, more duplicates)
# Lower values = looser matching (more false positives, fewer duplicates)
NOTE_SIMILARITY_THRESHOLD: float = 0.7

# Anti-repetition detection - character length difference tolerance
# Maximum allowed character count difference between two notes to trigger similarity check.
# If |len(note1) - len(note2)| > this value, notes are considered different regardless of content.
# Higher values = check similarity even for notes with very different lengths
# Lower values = only check similarity for notes of comparable length
ANTI_REPETITION_LENGTH_DIFF: int = 35

# Anti-repetition detection - minimum similarity ratio
# Minimum SequenceMatcher ratio to consider two notes as duplicates.
# Range: 0.0-1.0 where 1.0 = identical sequences
# Higher values = stricter duplicate detection (fewer false positives)
# Lower values = more aggressive duplicate detection (more false positives)
ANTI_REPETITION_SIMILARITY_MIN: float = 0.1

# Diff computation - minor change threshold
# Controls what percentage of similarity is considered a "minor change" vs "major update".
# Range: 0.0-1.0 where 1.0 = texts are identical
# Higher values = more changes are considered "minor" and may be ignored
# Lower values = even small changes are considered "major" and will be recorded
# At 0.90: only if new text is ≥90% similar to old text, the change is considered minor
DIFF_MINOR_CHANGE_THRESHOLD: float = 0.90

# Audio transcription - default confidence score (doesn't matter because the true confidence score is added when the text is inserted)
AUDIO_DEFAULT_CONFIDENCE: float = 0.5
# =============================================================================

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
        logger.info(f"Skipping image - already in database: {image_path_str}")
        return None

    # Run OCR and LLM-based text normalization
    logger.debug(f"Processing image: {image_path_str}")
    ocr_text, cleaned_text, confidence_score = image_transcription(image_path_str)

    # Quality firewall: detect and reject buggy HTR output
    buggy, reason = is_htr_buggy(ocr_text, cleaned_text)
    if buggy:
        logger.warning(f"Skipping image - HTR bug detected ({reason}): {image_path_str}")
        return None

    # Reject empty or trivially empty transcriptions
    if not cleaned_text or not cleaned_text.strip():
        logger.info(f"Skipping image - no exploitable text after normalization: {image_path_str}")
        return None

    if cleaned_text.strip() in ('""', "''"):
        logger.info(f"Skipping image - cleaned transcription is empty quotes: {image_path_str}")
        return None

    # Search for an existing similar note (same physical sheet)
    logger.debug(f"Searching for similar notes (threshold={NOTE_SIMILARITY_THRESHOLD})")
    similar_note_id = find_similar_note(
        cleaned_text, db_path=db_path_str, threshold=NOTE_SIMILARITY_THRESHOLD
    )

    if similar_note_id:
        logger.debug(f"Found similar note: {similar_note_id}")

    # Retrieve last known text for all notes
    try:
        last_texts = get_last_text_for_notes(db_path_str)
    except Exception as e:
        logger.warning(f"Could not retrieve last texts from database: {e}")
        last_texts = {}

    # Anti-repetition brigade: detect near-duplicate short notes across all existing notes
    logger.debug(f"Running anti-repetition check across {len(last_texts or {})} existing notes")
    for nid, prev_text in (last_texts or {}).items():
        s_prev = reflow_sentences(prev_text or "", width=80)
        s_new = reflow_sentences(cleaned_text or "", width=80)
        score_info = score_and_categorize_texts(s_prev, s_new)

        length_diff = abs(len(s_prev) - len(s_new))
        similarity_ratio = SequenceMatcher(None, s_prev, s_new).ratio()

        # Check if notes are similar enough in length and content to be considered duplicates
        if (
            length_diff < ANTI_REPETITION_LENGTH_DIFF
            and similarity_ratio > ANTI_REPETITION_SIMILARITY_MIN
        ):
            logger.info(
                f"Skipping image - duplicate note detected (similarity={similarity_ratio:.2f}, "
                f"length_diff={length_diff}): {image_path_str}"
            )
            logger.debug(f"Existing note ({len(s_prev)} chars): {s_prev}")
            logger.debug(f"New note ({len(s_new)} chars): {s_new}")
            return None

    diff_human = ""
    diff_json: list[dict[str, str | int]] = []

    # Determine if this is a new note or an update to an existing one
    if similar_note_id:
        # Same sheet: compute actual new content
        logger.debug(f"Computing diff for existing note {similar_note_id}")
        old_text = last_texts.get(similar_note_id, "")
        diff_human, diff_json = compute_diff(
            old_text, cleaned_text, minor_change_threshold=DIFF_MINOR_CHANGE_THRESHOLD
        )

        if not diff_human.strip():
            logger.info(
                f"Skipping image - no meaningful changes for note {similar_note_id}: {image_path_str}"
            )
            return None

        if not has_meaningful_line(diff_human):
            logger.info(
                f"Skipping image - diff has no meaningful content for note {similar_note_id}: {image_path_str}"
            )
            return None

        if not has_meaningful_text(cleaned_text):
            logger.info(
                f"Skipping image - no exploitable text after noise filtering: {image_path_str}"
            )
            return None

        note_id = similar_note_id
        logger.info(f"Updating existing note: {note_id}")

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
        logger.info(f"Creating new note: {note_id}")

    # Prepare raw metadata for database storage
    raw = {
        "source": "mistral-ocr-latest + mistral-large-latest",
        "image_path": image_path_str,
        "diff": diff_json,
    }

    # Extract named entities from new content
    logger.debug(f"Extracting entities from diff text ({len(diff_human)} chars)")
    if diff_human.strip():
        cleaned_diff_human = clean_added_text(diff_human)
        entities = extract_entities(cleaned_diff_human)
        logger.debug(f"Extracted entities: {list(entities.keys())}")
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
    logger.debug(f"Inserting note metadata into database: {note_id}")
    meta_id = insert_note_meta(
        extracted_data, img_path_proc=image_path_str, db_path=db_path_str
    )
    logger.info(f"Successfully inserted note {note_id} with meta_id={meta_id}")
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

    logger.debug(f"Processing audio: {audio_path_str}")

    # Reject empty or trivially empty transcriptions
    if (
        not transcription_clean
        or not transcription_clean.strip()
        or transcription_clean.strip() in ('""', "''")
    ):
        logger.info(
            f"Skipping audio - empty or trivial transcription: {audio_path_str}"
        )
        return None

    transcription_clean = strip_surrounding_quotes_local(transcription_clean)

    # Quality firewall: detect and reject buggy ASR output
    buggy, reason = is_htr_buggy(transcription_brute or "", transcription_clean or "")
    if buggy:
        logger.warning(f"Skipping audio - ASR bug detected ({reason}): {audio_path_str}")
        return None

    if not has_meaningful_text(transcription_clean):
        logger.info(
            f"Skipping audio - no meaningful text after noise filtering: {audio_path_str}"
        )
        return None

    # Prepare diff/texte_ajoute: treat entire audio as a single added line
    diff_human = f"+ Ligne 1. {transcription_clean.strip()}"
    diff_json: list[dict[str, str | int]] = [
        {"type": "insert", "line": 1, "content": transcription_clean.strip()}
    ]

    # Extract named entities from cleaned transcription
    logger.debug(f"Extracting entities from audio transcription ({len(transcription_clean)} chars)")
    cleaned_for_ner = clean_added_text(diff_human)
    entities = extract_entities(cleaned_for_ner) if cleaned_for_ner else {}
    if entities:
        logger.debug(f"Extracted entities: {list(entities.keys())}")

    # Generate unique note_id for this audio segment
    AUD_COUNTER += 1
    note_id = f"AUD-{AUD_COUNTER}"
    logger.info(f"Creating new audio note: {note_id}")

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
        "confidence_score": AUDIO_DEFAULT_CONFIDENCE,
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
    logger.debug(f"Inserting audio note metadata into database: {note_id}")
    meta_id = insert_note_meta(extracted_data, img_path_proc=None, db_path=db_path_str)
    logger.info(f"Successfully inserted audio note {note_id} with meta_id={meta_id}")