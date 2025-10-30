"""
OCR and LLM-based text normalization for handwritten dispatcher notes.

This module processes images of handwritten notes using Mistral OCR and normalizes the output
via LLM to produce clean, structured text suitable for diff computation and database insertion.
"""

import os
import re
import sys
from pathlib import Path

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from dotenv import load_dotenv
from mistralai import Mistral

from logger_config import setup_logger
from src.processing.prompts import OCR_NORMALIZATION_PROMPT
from src.utils.image_utils import encode_image

logger = setup_logger(__name__)

# Load API credentials from environment
load_dotenv()
api_key: str | None = os.getenv("MISTRAL_API_KEY")
client: Mistral = Mistral(api_key=api_key)

# =============================================================================
# Configuration Constants

# OCR model selection
OCR_MODEL: str = "mistral-ocr-latest"

# LLM model for text normalization
# Options: mistral-small-latest (fast, cheap), mistral-medium-latest (balanced), 
#          mistral-large-latest (best quality, slower, expensive)
NORMALIZATION_LLM_MODEL: str = "mistral-small-latest"

# LLM temperature for normalization (0.0 = deterministic, recommended)
NORMALIZATION_TEMPERATURE: float = 0.0

# Default confidence score for OCR pipeline (placeholder value)
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
    Perform OCR and LLM-based normalization on a handwritten note image.

    Workflow:
    1. Encode image to base64
    2. Run Mistral OCR to extract raw text
    3. Pre-process OCR output (normalize line prefixes)
    4. Apply LLM normalization via prompt engineering
    5. Post-process LLM output with deterministic rules

    Args:
        image_path: Path to the input image file.

    Returns:
        A tuple containing:
        - ocr_text: Raw OCR output after pre-processing
        - clean_text: Fully normalized and cleaned text
        - confidence_score: Fixed confidence score (0.5) for this pipeline

    Raises:
        Exception: Propagates errors from Mistral API calls.
    """
    # Encode image for API transmission
    print(image_path)
    base64_image = encode_image(str(image_path))

    # Step 1: Extract raw text via OCR
    logger.debug(f"Running OCR on image: {image_path}")
    response = client.ocr.process(
        model=OCR_MODEL,
        document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_image}",
        },
        include_image_base64=True,
    )
    ocr_text = response.pages[0].markdown.strip()
    logger.debug(f"Raw OCR output:\n{ocr_text}")

    # Step 2: Pre-process OCR text
    ocr_text = pre_collapse_continuations(ocr_text)

    # Step 3: Apply LLM normalization
    prompt = OCR_NORMALIZATION_PROMPT.format(ocr_text=ocr_text)

    logger.debug("Sending text to LLM for normalization")
    response = client.chat.complete(
        model=NORMALIZATION_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=NORMALIZATION_TEMPERATURE,
    )
    clean_text = response.choices[0].message.content.strip()

    # Step 4: Post-process LLM output
    clean_text = postprocess_normalized(clean_text)
    logger.info(f"OCR normalization complete. Final length: {len(clean_text)} chars")

    return ocr_text, clean_text, DEFAULT_OCR_CONFIDENCE