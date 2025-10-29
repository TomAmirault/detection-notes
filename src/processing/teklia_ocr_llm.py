"""
OCR and LLM-based text normalization for handwritten dispatcher notes using Teklia API.

This module processes images of handwritten notes using Teklia OCR and normalizes the output
via LLM to produce clean, structured text suitable for diff computation and database insertion.
"""

import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from mistralai import Mistral

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from logger_config import setup_logger
from src.processing.prompts import OCR_NORMALIZATION_PROMPT

logger = setup_logger(__name__)

# Load API credentials from environment
load_dotenv()
mistral_api_key: str | None = os.getenv("MISTRAL_API_KEY")
teklia_api_key: str | None = os.getenv("TEKLIA_API_KEY")

client: Mistral = Mistral(api_key=mistral_api_key)

# =============================================================================
# Configuration Constants

# Teklia OCR API configuration
TEKLIA_OCR_URL: str = "https://atr.ocelus.teklia.com/api/v1/transcribe/"
TEKLIA_LANGUAGE: str = "fr"

# Confidence threshold for including OCR lines
# Lines with confidence below this threshold are discarded
# Range: 0.0-1.0 where 1.0 = perfect confidence
OCR_CONFIDENCE_THRESHOLD: float = 0.5

# LLM model for text normalization
# Options: mistral-small-latest (fast, cheap), mistral-medium-latest (balanced),
#          mistral-large-latest (best quality, slower, expensive)
NORMALIZATION_LLM_MODEL: str = "mistral-small-latest"

# LLM temperature for normalization (0.0 = deterministic, recommended)
NORMALIZATION_TEMPERATURE: float = 0.0

# Default confidence score when OCR produces no valid lines
DEFAULT_OCR_CONFIDENCE: float = 0.5

# =============================================================================


def pre_collapse_continuations(text: str) -> str:
    """
    Normalize line starts without collapsing lines together.

    Removes common bullet markers, checkbox markers, and continuation markers at the start
    of each line, while preserving the original line structure.

    Args:
        text: Raw OCR output text.

    Returns:
        Normalized text with cleaned line prefixes but preserved line breaks.
    """
    out_lines: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        s = line.strip()

        # Remove checkbox markers like "[ ]" or "[x]"
        s = re.sub(r"^\[\s*[xX]?\s*\]\s*", "", s)

        # Remove common bullet markers at line start
        if s.startswith(("•", "-", "*", "\u2022")):
            s = re.sub(
                r"^[\u2022\-\*\u2023\u25E6\u2043\u2219\u25AA\u25CF\s]+", "", s
            ).strip()

        # Remove explicit continuation markers (↳, >) at line start
        s = re.sub(r"^[\u21b3>]+\s*", "", s)

        if s:
            out_lines.append(s)

    return "\n".join(out_lines)


def postprocess_normalized(text: str) -> str:
    """
    Apply deterministic post-processing rules to LLM-normalized text.

    Performs additional cleanup including:
    - Removal of code fence artifacts and markup
    - Space normalization
    - Filtering of administrative noise and empty semantic lines
    - Deterministic format normalization (hours, phone numbers, voltages)
    - Smart splitting of lines containing colons (except for times/URLs)

    Args:
        text: LLM-normalized text output.

    Returns:
        Final cleaned and structured text ready for database insertion.
    """
    # Remove code fences and markup artifacts
    text = text.replace("```", "")
    text = text.replace("<<<", "").replace(">>>", "")

    # Normalize whitespace globally
    text = re.sub(r"[ \t]+", " ", text)

    # Process lines with filtering and normalization
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Remove bullet markers and keep as separate lines
        stripped = line
        if stripped.startswith(("•", "-", "*", "\u2022")):
            stripped = re.sub(
                r"^[\u2022\-\*\u2023\u25E6\u2023\u2043\u2219\u25AA\u25CF\s]+",
                "",
                stripped,
            ).strip()
            lines.append(stripped)
            continue

        # Filter out semantically empty lines
        if not re.search(r"[A-Za-zÀ-ÿ0-9]", line):
            continue

        # Apply deterministic format normalizations
        line = re.sub(r"\b(\d{1,2})\s*h\b", r"\1h", line)  # "16 h" -> "16h"
        line = re.sub(r"\b(\d+)\s*RV\b", r"\1kV", line)  # "20RV" -> "20kV"
        line = re.sub(
            r"\b0\d(?:\s?\d{2}){4}\b", lambda m: m.group(0).replace(" ", ""), line
        )  # Phone number spacing

        # Smart colon handling: split into separate lines except for times/URLs
        if ":" in line:
            # Preserve lines with time patterns (HH:MM)
            if re.search(r"\b\d{1,2}:\d{2}\b", line):
                lines.append(line)
            # Preserve lines with URL schemes
            elif re.search(r"https?://|\w+://", line.lower()):
                lines.append(line)
            else:
                # Split at colon for field:value patterns
                head, tail = line.split(":", 1)
                head = head.strip()
                tail = tail.strip()
                if head and tail:
                    lines.append(head + ":")
                    lines.append(tail)
                else:
                    lines.append(line)
        else:
            lines.append(line)

    # Filter out administrative noise and numeric debris
    final: list[str] = []
    for line in lines:
        # Remove standalone "Vote" or "Note" lines
        if re.fullmatch(r"(?i)\s*vote(\s+\d+)?\s*", line):
            continue
        if re.fullmatch(r"(?i)\s*note(\s+\d+)?\s*", line):
            continue
        # Remove standalone short numbers
        if re.fullmatch(r"\d{1,3}", line):
            continue
        # Remove "None" artifacts
        if line.lower() == "none":
            continue
        final.append(line)

    return "\n".join(final).strip()


def image_transcription(image_path: str | Path) -> tuple[str, str, float]:
    """
    Perform OCR and LLM-based normalization on a handwritten note image using Teklia API.

    Workflow:
    1. Call Teklia OCR API to extract raw text with confidence scores
    2. Filter lines below confidence threshold
    3. Pre-process OCR output (normalize line prefixes)
    4. Apply LLM normalization via prompt engineering
    5. Post-process LLM output with deterministic rules

    Args:
        image_path: Path to the input image file.

    Returns:
        A tuple containing:
        - ocr_text: Raw OCR output after pre-processing
        - clean_text: Fully normalized and cleaned text
        - confidence_score: Weighted average confidence score from OCR

    Raises:
        Exception: Propagates errors from Teklia API calls or network issues.
    """
    image_path_str = str(image_path)

    # Step 1: Call Teklia OCR API
    logger.debug(f"Calling Teklia OCR API for image: {image_path_str}")
    headers = {"API-Key": teklia_api_key}
    
    with open(image_path_str, "rb") as img_file:
        files = {"image": img_file}
        params = {"language": TEKLIA_LANGUAGE}
        
        response = requests.post(
            TEKLIA_OCR_URL, headers=headers, files=files, params=params
        )

    if response.status_code != 200:
        error_msg = f"Teklia OCR API error: {response.status_code} - {response.text}"
        logger.error(error_msg)
        raise Exception(error_msg)

    # Step 2: Extract and filter OCR results by confidence threshold
    ocr_text = ""
    confidence_per_line: list[float] = []
    char_per_line: list[int] = []

    logger.debug(f"Processing OCR results (threshold={OCR_CONFIDENCE_THRESHOLD})")
    for line_result in response.json()["results"]:
        confidence = line_result["confidence"]
        text = line_result["text"]
        
        logger.debug(f"OCR line confidence={confidence:.3f}: {text}")
        
        if confidence >= OCR_CONFIDENCE_THRESHOLD:
            ocr_text += text + "\n"
            confidence_per_line.append(confidence)
            char_per_line.append(len(text))

    # Calculate weighted average confidence
    if len(ocr_text) > 0 and sum(char_per_line) > 0:
        ocr_text_confidence = sum(
            conf * nb_chars for conf, nb_chars in zip(confidence_per_line, char_per_line)
        ) / sum(char_per_line)
    else:
        ocr_text_confidence = DEFAULT_OCR_CONFIDENCE

    logger.info(
        f"Teklia OCR complete - {len(confidence_per_line)} lines extracted, "
        f"weighted confidence={ocr_text_confidence:.3f}"
    )

    # Step 3: Pre-process OCR text
    ocr_text = pre_collapse_continuations(ocr_text)

    # Step 4: Apply LLM normalization
    # TODO(maintainer): Large prompt moved to prompts.py for readability.
    prompt = OCR_NORMALIZATION_PROMPT.format(ocr_text=ocr_text)

    logger.debug("Sending text to LLM for normalization")
    response = client.chat.complete(
        model=NORMALIZATION_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=NORMALIZATION_TEMPERATURE,
    )
    clean_text = response.choices[0].message.content.strip()

    # Step 5: Post-process LLM output
    clean_text = postprocess_normalized(clean_text)
    logger.info(f"OCR normalization complete. Final length: {len(clean_text)} chars")

    return ocr_text, clean_text, ocr_text_confidence