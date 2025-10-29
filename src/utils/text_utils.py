"""
Text processing utilities for note comparison, diff computation, and quality filtering.

This module provides core text manipulation functions used throughout the project, including:
- Diff computation with intelligent line alignment and similarity detection
- HTR/OCR quality filtering (bug detection, noise removal)
- Text normalization and canonicalization for comparison
- Similarity scoring between text blocks
"""

import re
import textwrap
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from logger_config import setup_logger

logger = setup_logger(__name__)

# =============================================================================
# Configuration Constants

# Diff computation - similarity threshold for line alignment
# Lines with similarity >= this value are considered potential matches for replacement
# Range: 0.0-1.0 where 1.0 = identical lines
LINE_ALIGNMENT_SIMILARITY_MIN: float = 0.0

# Split/merge detection threshold
# Minimum similarity to consider 1->2 or 2->1 line transformations as valid
# Range: 0.0-1.0 where 1.0 = perfect match after split/merge
SPLIT_MERGE_THRESHOLD: float = 0.85

# HTR bug detection - minimum token count for analysis
# Texts with fewer tokens are considered too short to reliably detect bugs
MIN_TOKENS_FOR_BUG_DETECTION: int = 5

# HTR bug detection - dominant word threshold
# If one word represents >= this ratio of all tokens, text is considered buggy
DOMINANT_WORD_THRESHOLD: float = 0.60

# HTR bug detection - consecutive repetition threshold
# Maximum allowed consecutive repetitions of the same token before flagging as buggy
MAX_CONSECUTIVE_REPETITIONS: int = 5

# HTR bug detection - unique token ratio threshold
# If unique tokens / total tokens <= this value, text is considered too repetitive
UNIQUE_TOKEN_RATIO_MIN: float = 0.20

# HTR bug detection - line repetition threshold
# Maximum times the same line can appear before flagging as buggy
MAX_LINE_REPETITIONS: int = 5

# Text reflow - non-terminal words (French prepositions and articles)
# These words should not trigger sentence ending when appearing at line end
NON_TERMINAL_WORDS: set[str] = {
    "sur", "à", "le", "la", "les", "des", "de", "du", "en", "et", "ou",
    "par", "pour", "avec", "au", "aux", "chez", "dans", "vers"
}

# Text scoring - default similarity thresholds for categorization
SIMILARITY_THRESHOLDS: dict[str, float] = {
    "identical": 0.90,
    "close": 0.75,
    "related": 0.50,
}

# =============================================================================


def has_meaningful_line(s: str) -> bool:
    """
    Check if at least one line contains alphanumeric content.

    Args:
        s: Input text (potentially multi-line).

    Returns:
        True if any line contains at least one letter (including accented) or digit.
    """
    for ln in (s or "").splitlines():
        if re.search(r"[A-Za-zÀ-ÿ0-9]", ln):
            return True
    return False


def has_meaningful_text(s: str) -> bool:
    """
    Check if text contains substantial content (anti-noise filter).

    Args:
        s: Input text.

    Returns:
        True if text contains at least one word of 2+ letters or any digit.
    """
    if not s or not s.strip():
        return False
    return bool(re.search(r"[A-Za-zÀ-ÿ]{2,}", s) or re.search(r"\d", s))


def _normalize_for_similarity(s: str) -> str:
    """
    Normalize text for similarity comparison by minimizing typographic variations.

    Applies the following transformations:
    - Lowercase
    - Normalize dashes and spacing around punctuation
    - Collapse multiple spaces

    Args:
        s: Input text.

    Returns:
        Normalized text suitable for similarity comparison.
    """
    s = s.strip().lower()

    # Normalize dashes and spaces around punctuation
    s = re.sub(r"\s*[-–—]\s*", "-", s)
    s = re.sub(r"\s*:\s*", ":", s)
    s = re.sub(r"\s*;\s*", ";", s)
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\s*\.\s*", ".", s)

    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s


def _similarity(a: str, b: str) -> float:
    """
    Compute similarity ratio between two strings after normalization.

    Args:
        a: First text string.
        b: Second text string.

    Returns:
        Similarity ratio between 0.0 (completely different) and 1.0 (identical).
    """
    return SequenceMatcher(
        None, _normalize_for_similarity(a), _normalize_for_similarity(b)
    ).ratio()


def _align_block(
    old_block: list[str], new_block: list[str]
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """
    Align two text blocks using greedy similarity-based matching.

    Uses a greedy algorithm to pair lines from old_block and new_block based on
    text similarity, prioritizing highest similarity matches first.

    Args:
        old_block: List of lines from the old text.
        new_block: List of lines from the new text.

    Returns:
        A tuple containing:
        - matches: List of (old_idx, new_idx, similarity) for paired lines
        - old_unmatched: Indices of unmatched old lines (deletions)
        - new_unmatched: Indices of unmatched new lines (insertions)
    """
    # Compute all pairwise similarities
    pairs = []
    for i, a in enumerate(old_block):
        for j, b in enumerate(new_block):
            sim = _similarity(a, b)
            pairs.append((sim, i, j))

    # Sort by descending similarity
    pairs.sort(reverse=True, key=lambda t: t[0])

    matched_old = set()
    matched_new = set()
    matches = []

    # Greedy matching: take best available pairs
    for sim, i, j in pairs:
        if i in matched_old or j in matched_new:
            continue
        matched_old.add(i)
        matched_new.add(j)
        matches.append((i, j, sim))

    old_unmatched = [i for i in range(len(old_block)) if i not in matched_old]
    new_unmatched = [j for j in range(len(new_block)) if j not in matched_new]

    # Sort matches by new line index for stability
    matches.sort(key=lambda t: (t[1], t[0]))
    return matches, old_unmatched, new_unmatched


def _try_split_merge_matches(
    old_block: list[str], new_block: list[str], split_thresh: float = SPLIT_MERGE_THRESHOLD
) -> tuple[list[tuple[tuple[int, int], tuple[int, int], str]], set[int], set[int]]:
    """
    Detect local split (1->2) and merge (2->1) transformations between text blocks.

    Identifies cases where:
    - One old line was split into two new lines (1->2 split)
    - Two old lines were merged into one new line (2->1 merge)

    Args:
        old_block: List of lines from the old text.
        new_block: List of lines from the new text.
        split_thresh: Minimum similarity threshold to accept split/merge detection.

    Returns:
        A tuple containing:
        - matches: List of ((old_start, old_end), (new_start, new_end), kind)
          where kind is "1to2" or "2to1"
        - used_old: Set of old line indices consumed by split/merge operations
        - used_new: Set of new line indices consumed by split/merge operations
    """
    n_old, n_new = len(old_block), len(new_block)
    used_old, used_new = set(), set()
    matches = []

    # Detect 1->2 splits
    for i in range(n_old):
        if i in used_old:
            continue
        best = None
        for j in range(n_new - 1):
            if j in used_new or (j + 1) in used_new:
                continue
            # Compare one old line to concatenation of two new lines
            sim = _similarity(old_block[i], f"{new_block[j]} {new_block[j+1]}".strip())
            if best is None or sim > best[0]:
                best = (sim, i, j)
        if best and best[0] >= split_thresh:
            sim, i0, j0 = best
            matches.append(((i0, i0), (j0, j0 + 1), "1to2"))
            used_old.add(i0)
            used_new.update({j0, j0 + 1})

    # Detect 2->1 merges
    for j in range(n_new):
        if j in used_new:
            continue
        best = None
        for i in range(n_old - 1):
            if i in used_old or (i + 1) in used_old:
                continue
            # Compare concatenation of two old lines to one new line
            sim = _similarity(f"{old_block[i]} {old_block[i+1]}".strip(), new_block[j])
            if best is None or sim > best[0]:
                best = (sim, i, j)
        if best and best[0] >= split_thresh:
            sim, i0, j0 = best
            matches.append(((i0, i0 + 1), (j0, j0), "2to1"))
            used_old.update({i0, i0 + 1})
            used_new.add(j0)

    matches.sort(key=lambda m: m[1][0])
    return matches, used_old, used_new


def compute_diff(
    old_text: str, new_text: str, minor_change_threshold: float = 0.90
) -> tuple[str, list[dict[str, Any]]]:
    """
    Compute intelligent diff between two texts with line-level operations.

    This function performs sophisticated diff computation with:
    - Split/merge detection (1->2, 2->1 line transformations)
    - Similarity-based line alignment
    - Filtering of minor changes below threshold
    - Re-pairing of unmatched lines to avoid spurious delete+insert pairs

    Args:
        old_text: Original text.
        new_text: Updated text.
        minor_change_threshold: Similarity threshold below which changes are reported.
            Range: 0.0-1.0. At 0.90, only changes with <90% similarity are reported.

    Returns:
        A tuple containing:
        - human_str: Human-readable diff with line numbers and operation markers
          (+ for insert, ~ for replace, - for delete)
        - diff_json: List of operation dictionaries with structure:
          {type: "insert|replace|delete", line: int, content: str, ...}

    Note:
        Line numbers in output are 1-based. For 'replace' operations, both 'line'
        (new text) and 'old_line' (old text) are included.
    """
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()

    sm = SequenceMatcher(None, old_lines, new_lines, autojunk=False)

    human_rows: list[str] = []
    diff_json: list[dict[str, Any]] = []

    logger.debug(f"Computing diff: {len(old_lines)} old lines vs {len(new_lines)} new lines")

    # Process each diff block
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        old_block = old_lines[i1:i2]
        new_block = new_lines[j1:j2]

        if tag in ("replace", "insert", "delete"):
            # Step 1: Detect split/merge transformations (1->2, 2->1)
            split_matches, used_old, used_new = _try_split_merge_matches(
                old_block, new_block, split_thresh=SPLIT_MERGE_THRESHOLD
            )

            # Emit replace operations for split/merge cases
            for (i_start, i_end), (j_start, j_end), kind in split_matches:
                if kind == "1to2":
                    # One old line split into two new lines
                    a = old_block[i_start]
                    b1, b2 = new_block[j_start], new_block[j_start + 1]

                    sim1 = _similarity(a, b1)
                    if (
                        _normalize_for_similarity(a) != _normalize_for_similarity(b1)
                        and sim1 < minor_change_threshold
                        and b1.strip()
                    ):
                        line_new = j1 + j_start + 1
                        human_rows.append(f"~ Ligne {line_new}. {b1}")
                        diff_json.append(
                            {
                                "type": "replace",
                                "line": line_new,
                                "old_line": i1 + i_start + 1,
                                "old_content": a,
                                "content": b1,
                                "similarity": float(sim1),
                                "note": "split(1→2)-part1",
                            }
                        )

                    sim2 = _similarity(a, b2)
                    if (
                        _normalize_for_similarity(a) != _normalize_for_similarity(b2)
                        and sim2 < minor_change_threshold
                        and b2.strip()
                    ):
                        line_new = j1 + j_start + 2
                        human_rows.append(f"~ Ligne {line_new}. {b2}")
                        diff_json.append(
                            {
                                "type": "replace",
                                "line": line_new,
                                "old_line": i1 + i_start + 1,
                                "old_content": a,
                                "content": b2,
                                "similarity": float(sim2),
                                "note": "split(1→2)-part2",
                            }
                        )

                elif kind == "2to1":
                    # Two old lines merged into one new line
                    a1, a2 = old_block[i_start], old_block[i_end]
                    b = new_block[j_start]
                    sim = _similarity(f"{a1} {a2}".strip(), b)
                    if (
                        _normalize_for_similarity(f"{a1} {a2}")
                        != _normalize_for_similarity(b)
                        and sim < minor_change_threshold
                        and b.strip()
                    ):
                        line_new = j1 + j_start + 1
                        human_rows.append(f"~ Ligne {line_new}. {b}")
                        diff_json.append(
                            {
                                "type": "replace",
                                "line": line_new,
                                "old_line": [i1 + i_start + 1, i1 + i_end + 1],
                                "old_content": f"{a1} {a2}",
                                "content": b,
                                "similarity": float(sim),
                                "note": "merge(2→1)",
                            }
                        )

            # Step 2: Remove lines consumed by split/merge from further processing
            old_rest_map = [k for k in range(len(old_block)) if k not in used_old]
            new_rest_map = [k for k in range(len(new_block)) if k not in used_new]
            old_rest = [old_block[k] for k in old_rest_map]
            new_rest = [new_block[k] for k in new_rest_map]

            # Step 3: Align remaining lines 1<->1
            matches, old_unmatched_rel, new_unmatched_rel = _align_block(old_rest, new_rest)

            # Remap relative indices to original block indices
            remapped_matches = []
            for i_rel2, j_rel2, sim in matches:
                i_rel_orig = old_rest_map[i_rel2] if i_rel2 < len(old_rest_map) else None
                j_rel_orig = new_rest_map[j_rel2] if j_rel2 < len(new_rest_map) else None
                remapped_matches.append((i_rel_orig, j_rel_orig, sim))

            old_unmatched = [old_rest_map[i] for i in old_unmatched_rel]
            new_unmatched = [new_rest_map[j] for j in new_unmatched_rel]

            # Step 4: Emit REPLACE for aligned pairs (1<->1)
            for i_rel, j_rel, sim in remapped_matches:
                old_content = old_block[i_rel]
                new_content = new_block[j_rel]
                new_abs_line = j1 + j_rel + 1
                old_abs_line = i1 + i_rel + 1

                # Skip if normalized content is identical
                if _normalize_for_similarity(old_content) == _normalize_for_similarity(
                    new_content
                ):
                    continue

                # Only report if similarity is below threshold
                if sim < minor_change_threshold:
                    human_rows.append(f"~ Ligne {new_abs_line}. {new_content}")
                    diff_json.append(
                        {
                            "type": "replace",
                            "line": new_abs_line,
                            "old_line": old_abs_line,
                            "old_content": old_content,
                            "content": new_content,
                            "similarity": float(sim),
                        }
                    )

            # Step 5: Re-pair unmatched lines to avoid spurious DELETE+INSERT
            re_pairs = []
            if old_unmatched and new_unmatched:
                cand = []
                for i_rel in old_unmatched:
                    for j_rel in new_unmatched:
                        sim = _similarity(old_block[i_rel], new_block[j_rel])
                        cand.append((sim, i_rel, j_rel))
                cand.sort(reverse=True, key=lambda t: t[0])

                used_o, used_n = set(), set()
                for sim, i_rel, j_rel in cand:
                    if i_rel in used_o or j_rel in used_n:
                        continue
                    used_o.add(i_rel)
                    used_n.add(j_rel)
                    re_pairs.append((i_rel, j_rel, sim))

                old_unmatched = [i for i in old_unmatched if i not in used_o]
                new_unmatched = [j for j in new_unmatched if j not in used_n]

            # Step 6: Emit REPLACE for re-paired lines
            for i_rel, j_rel, sim in re_pairs:
                old_content = old_block[i_rel]
                new_content = new_block[j_rel]
                new_abs_line = j1 + j_rel + 1
                old_abs_line = i1 + i_rel + 1

                if _normalize_for_similarity(old_content) == _normalize_for_similarity(
                    new_content
                ):
                    continue

                if sim < minor_change_threshold:
                    human_rows.append(f"~ Ligne {new_abs_line}. {new_content}")
                    diff_json.append(
                        {
                            "type": "replace",
                            "line": new_abs_line,
                            "old_line": old_abs_line,
                            "old_content": old_content,
                            "content": new_content,
                            "similarity": float(sim),
                        }
                    )

            # Step 7: Emit INSERT for remaining new lines
            for j_rel in new_unmatched:
                new_content = new_block[j_rel]
                if not new_content.strip():
                    continue
                new_abs_line = j1 + j_rel + 1
                human_rows.append(f"+ Ligne {new_abs_line}. {new_content}")
                diff_json.append(
                    {"type": "insert", "line": new_abs_line, "content": new_content}
                )

            # Step 8: Emit DELETE for remaining old lines
            for i_rel in old_unmatched:
                old_content = old_block[i_rel]
                if not old_content.strip():
                    continue
                old_abs_line = i1 + i_rel + 1
                human_rows.append(f"- Ancienne ligne {old_abs_line}. {old_content}")
                diff_json.append(
                    {
                        "type": "delete",
                        "old_line": old_abs_line,
                        "old_content": old_content,
                    }
                )

    human_str = "\n".join(human_rows)
    logger.debug(
        f"Diff computation complete: {len(diff_json)} operations, "
        f"{len(human_rows)} lines in human output"
    )
    return human_str, diff_json


def _max_consecutive_run(tokens: list[str]) -> int:
    """
    Find the maximum consecutive repetition count for any token in a sequence.

    Args:
        tokens: List of tokens to analyze.

    Returns:
        Maximum number of consecutive repetitions of any single token.
    """
    max_run, cur, prev = 1, 1, None
    for t in tokens:
        if t == prev:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 1
            prev = t
    return max_run


def is_htr_buggy(ocr_text: str, cleaned_text: str = "") -> tuple[bool, str]:
    """
    Detect buggy OCR/HTR output using heuristics for abnormal repetition patterns.

    Analyzes token distribution, consecutive repetitions, and line duplications
    to identify OCR failures that produce nonsensical repetitive output.

    Args:
        ocr_text: Raw OCR output text.
        cleaned_text: Optional cleaned/normalized text for additional validation.

    Returns:
        A tuple containing:
        - is_buggy: True if the text appears to be buggy OCR output
        - reason: Human-readable explanation of why the text was flagged (empty if not buggy)

    Note:
        Very short texts (< MIN_TOKENS_FOR_BUG_DETECTION tokens) are not analyzed
        and return False to avoid false positives on legitimate short notes.
    """
    if not ocr_text or not ocr_text.strip():
        return True, "empty OCR output"

    # Extract tokens and lines
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", ocr_text.lower())
    lines = [ln.strip().lower() for ln in ocr_text.splitlines() if ln.strip()]

    # Skip analysis for very short texts (often legitimate short notes)
    if len(tokens) < MIN_TOKENS_FOR_BUG_DETECTION:
        return False, ""

    N = len(tokens)
    cnt = Counter(tokens)
    top_word, top_freq = cnt.most_common(1)[0]

    # Calculate metrics
    dom_ratio = top_freq / N  # Ratio of most frequent token
    uniq_ratio = len(cnt) / N  # Token diversity ratio
    run_max = _max_consecutive_run(tokens)  # Longest consecutive repetition

    # Line repetition analysis
    line_cnt = Counter(lines)
    max_line_repeat = max(line_cnt.values()) if line_cnt else 0

    logger.debug(
        f"HTR bug check: dom_ratio={dom_ratio:.2f}, uniq_ratio={uniq_ratio:.2f}, "
        f"run_max={run_max}, max_line_repeat={max_line_repeat}"
    )

    # Apply detection rules
    if dom_ratio >= DOMINANT_WORD_THRESHOLD:
        return True, f"abnormal dominant word ({top_word}={top_freq}/{N})"

    if run_max >= MAX_CONSECUTIVE_REPETITIONS:
        return True, f"abnormal consecutive repetition (run={run_max})"

    if uniq_ratio <= UNIQUE_TOKEN_RATIO_MIN and N >= 15:
        return True, f"low token diversity (uniq_ratio={uniq_ratio:.2f})"

    if max_line_repeat >= MAX_LINE_REPETITIONS:
        return True, f"repeated identical lines (x{max_line_repeat})"

    # Additional check: if cleaned text is empty but OCR is repetitive
    if (
        cleaned_text is not None
        and not cleaned_text.strip()
        and (dom_ratio > 0.5 or run_max >= 4)
    ):
        return True, "cleaned text empty with repetitive OCR"

    return False, ""


def clean_added_text(text: str) -> str:
    """
    Remove diff markers and prefixes from diff output to extract clean content.

    Processes text containing diff-like markers (lines starting with '+ Ligne X',
    '- Ancienne ligne X', or '~ Ligne X') and extracts only the actual content.

    Args:
        text: Input text containing diff-like markers.

    Returns:
        Cleaned text with all prefixes removed and deleted lines omitted.

    Example:
        Input:  "+ Ligne 1. Hello\\n- Ancienne ligne 2. Old\\n~ Ligne 3. World"
        Output: "Hello\\nWorld"
    """
    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()

        # Skip deleted lines (starting with "- Ancienne ligne X.")
        if re.match(r"^\-\s*Ancienne\s+ligne\s+\d+\.", line, flags=re.IGNORECASE):
            continue

        # Remove prefixes from added or modified lines ("+ Ligne X.", "~ Ligne X.")
        new_line = re.sub(
            r"^[+~]\s*Ligne\s+\d+\.\s*", "", line, flags=re.IGNORECASE
        ).strip()

        if new_line:
            cleaned_lines.append(new_line)

    return "\n".join(cleaned_lines)


def reflow_sentences(text: str, width: int = 80) -> str:
    """
    Reflow multi-line text into properly punctuated and wrapped paragraphs.

    Applies intelligent sentence reconstruction with French grammar rules:
    - Joins line segments that don't end with terminal punctuation
    - Avoids adding periods after prepositions or short words
    - Joins lines when next line starts with lowercase (continuation)
    - Capitalizes sentence starts
    - Wraps output to specified width without breaking words

    Args:
        text: Input multi-line text.
        width: Maximum line width for wrapping (default: 80).

    Returns:
        Reflowed text with proper sentence structure and line wrapping.

    Note:
        Uses NON_TERMINAL_WORDS to identify French prepositions and articles
        that should not trigger sentence endings.
    """
    if not text or not text.strip():
        return text or ""

    # Split into parts and normalize whitespace
    parts = [p.rstrip() for p in text.splitlines()]
    parts = [re.sub(r"\s+", " ", p).strip() for p in parts]

    merged_parts = []
    i = 0

    # Merge lines intelligently based on punctuation and capitalization
    while i < len(parts):
        p = parts[i]
        if not p:
            i += 1
            continue

        # If line ends with terminal punctuation, keep as-is
        if re.search(r"[\.\?!]$", p):
            merged_parts.append(p)
            i += 1
            continue

        # Look ahead to next non-empty part
        j = i + 1
        next_part = None
        while j < len(parts):
            if parts[j]:
                next_part = parts[j]
                break
            j += 1

        last_word = p.split()[-1].lower() if p.split() else ""

        # Check if next line starts with lowercase
        if next_part:
            m = re.match(r"\s*([a-zà-ÿ])", next_part, flags=re.IGNORECASE)
            next_starts_lower = bool(m and m.group(1).islower())
        else:
            next_starts_lower = False

        # Don't end sentence if last word is a preposition and there's a next line
        if last_word in NON_TERMINAL_WORDS and next_part:
            combined = p + " " + next_part
            merged_parts.append(combined)
            i = j + 1
            continue

        # Join with next line if it starts with lowercase (continuation)
        if next_part and next_starts_lower:
            combined = p + " " + next_part
            merged_parts.append(combined)
            i = j + 1
            continue

        # Otherwise, add period and continue
        merged_parts.append(p + ".")
        i += 1

    def cap_sentence(s: str) -> str:
        """Capitalize sentence starts after terminal punctuation."""
        s = s.strip()
        s = re.sub(r"\s*([\.\?!])\s*", lambda m: m.group(1) + " ", s)
        s = re.sub(
            r"(^|[\.\?!]\s+)([a-zà-ÿ])",
            lambda m: m.group(1) + m.group(2).upper(),
            s,
            flags=re.IGNORECASE,
        )
        return s.strip()

    # Apply capitalization and join sentences
    sentences = [cap_sentence(s) for s in merged_parts if s.strip()]
    paragraph = " ".join(s.rstrip() for s in sentences)

    # Wrap to specified width
    wrapped = textwrap.fill(
        paragraph.strip(), width=width, break_long_words=False, break_on_hyphens=False
    )
    return wrapped


def canonicalize_for_compare(s: str) -> list[str]:
    """
    Normalize text into canonical tokens for comparison.

    Applies aggressive normalization:
    - Remove diacritics (accents)
    - Lowercase
    - Keep only alphanumeric characters and plus signs (for phone numbers)
    - Split on whitespace

    Args:
        s: Input text string.

    Returns:
        List of normalized tokens suitable for similarity comparison.

    Example:
        Input:  "Hôtel à Paris +33612345678"
        Output: ["hotel", "a", "paris", "+33612345678"]
    """
    if not s:
        return []

    # Normalize Unicode and remove diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()

    # Keep only alphanumeric and plus signs
    s = re.sub(r"[^0-9a-z+]+", " ", s)
    tokens = [t for t in s.split() if t]
    return tokens


def token_jaccard(a: list[str], b: list[str]) -> float:
    """
    Compute Jaccard similarity between two token lists.

    Jaccard similarity = |intersection| / |union|

    Args:
        a: First list of tokens.
        b: Second list of tokens.

    Returns:
        Jaccard similarity score between 0.0 (no overlap) and 1.0 (identical sets).
    """
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = sa & sb
    uni = sa | sb
    return len(inter) / len(uni) if uni else 0.0


def token_f1(a: list[str], b: list[str]) -> float:
    """
    Compute F1 score between two token lists (treats tokens as multisets).

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        a: First list of tokens.
        b: Second list of tokens.

    Returns:
        F1 score between 0.0 (no overlap) and 1.0 (perfect match).
    """
    if not a and not b:
        return 1.0

    ca, cb = Counter(a), Counter(b)
    common = sum((ca & cb).values())

    if common == 0:
        return 0.0

    prec = common / sum(ca.values())
    rec = common / sum(cb.values())
    return 2 * prec * rec / (prec + rec)


def score_and_categorize_texts(
    a: str,
    b: str,
    weights: tuple[float, float] = (0.5, 0.5),
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Compute comprehensive similarity score and categorization between two texts.

    Combines multiple similarity metrics (Jaccard, F1) and includes phone number
    similarity detection for dispatcher notes.

    Args:
        a: First text string.
        b: Second text string.
        weights: Tuple of (jaccard_weight, f1_weight) for score computation.
        thresholds: Optional dict with keys "identical", "close", "related" mapping
            to float thresholds for categorization. Uses SIMILARITY_THRESHOLDS if None.

    Returns:
        Dictionary containing:
        - score: Weighted similarity score (0.0-1.0)
        - category: Text categorization ("identical", "close", "related", "different")
        - jaccard: Jaccard similarity coefficient
        - f1: F1 score
        - phone_ok: True if similar phone numbers detected in both texts

    Example:
        >>> score_and_categorize_texts("Call 0612345678", "Call 0612345679")
        {'score': 0.95, 'category': 'identical', 'jaccard': 0.9, 'f1': 1.0, 'phone_ok': True}
    """
    thresholds = thresholds or SIMILARITY_THRESHOLDS

    # Tokenize both texts
    ta = canonicalize_for_compare(a or "")
    tb = canonicalize_for_compare(b or "")

    # Compute similarity metrics
    ja = token_jaccard(ta, tb)
    fa = token_f1(ta, tb)

    def phone_tokens(ts: list[str]) -> list[str]:
        """Extract tokens that look like phone numbers."""
        return [t for t in ts if re.fullmatch(r"\+?\d{6,}", t)]

    # Phone number similarity check
    pa = phone_tokens(ta)
    pb = phone_tokens(tb)
    phone_ok = False

    if pa and pb:

        def close_nums(x: str, y: str) -> bool:
            """Check if two phone numbers are close (≤2 character differences)."""
            diff = sum(1 for c1, c2 in zip(x, y) if c1 != c2)
            diff += abs(len(x) - len(y))
            return diff <= 2

        phone_ok = any(close_nums(x, y) for x in pa for y in pb)

    # Compute weighted score
    score = float(weights[0]) * ja + float(weights[1]) * fa
    score = max(0.0, min(1.0, score))

    # Categorize based on thresholds
    if score >= thresholds["identical"]:
        cat = "identical"
    elif score >= thresholds["close"]:
        cat = "close"
    elif score >= thresholds["related"]:
        cat = "related"
    else:
        cat = "different"

    return {
        "score": round(score, 3),
        "category": cat,
        "jaccard": round(ja, 3),
        "f1": round(fa, 3),
        "phone_ok": phone_ok,
    }