# -*- coding: utf-8 -*-
import re
import textwrap
import unicodedata
from typing import Any, List, Optional, Tuple, Dict
from difflib import SequenceMatcher
from collections import Counter


def has_meaningful_line(s: str) -> bool:
    """
    Retourne True si au moins une ligne de s contient une lettre (y compris accentuée) ou un chiffre.
    """
    for ln in (s or "").splitlines():
        if re.search(r"[A-Za-zÀ-ÿ0-9]", ln):
            return True
    return False


def has_meaningful_text(s: str) -> bool:
    """
    Retourne True si s contient au moins un mot de 2 lettres (y compris accentuées) ou un chiffre.
    """
    if not s or not s.strip():
        return False
    return bool(re.search(r"[A-Za-zÀ-ÿ]{2,}", s) or re.search(r"\d", s))


def _normalize_for_similarity(s: str) -> str:
    # Minimise l'effet de variations typographiques mineures
    s = s.strip().lower()

    # Normalise les tirets et espaces autour des signes - : ; , .
    # "Caen - Cherbourg" -> "caen-cherbourg"
    s = re.sub(r"\s*[-–—]\s*", "-", s)
    s = re.sub(r"\s*:\s*", ":", s)
    s = re.sub(r"\s*;\s*", ";", s)
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\s*\.\s*", ".", s)

    # Écrase espaces multiples
    s = re.sub(r"\s+", " ", s)
    return s


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_for_similarity(a), _normalize_for_similarity(b)).ratio()


def _align_block(old_block: List[str], new_block: List[str]) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Aligne un bloc old_block (indexés par i) et new_block (indexés par j) par similarité.
    Retourne:
      - matches: liste de (i, j, sim) appariés (greedy, sim décroissante)
      - old_unmatched: indices i non appariés (à supprimer)
      - new_unmatched: indices j non appariés (à insérer)
    """
    # calcule toutes les paires avec leur similarité
    pairs = []
    for i, a in enumerate(old_block):
        for j, b in enumerate(new_block):
            sim = _similarity(a, b)
            pairs.append((sim, i, j))
    # tri décroissant par similarité
    pairs.sort(reverse=True, key=lambda t: t[0])

    matched_old = set()
    matched_new = set()
    matches = []

    for sim, i, j in pairs:
        if i in matched_old or j in matched_new:
            continue
        # On accepte un match même si sim est faible ; on décidera ensuite s’il faut le logguer
        matched_old.add(i)
        matched_new.add(j)
        matches.append((i, j, sim))

    old_unmatched = [i for i in range(len(old_block)) if i not in matched_old]
    new_unmatched = [j for j in range(len(new_block)) if j not in matched_new]
    # tri par indices croissants pour stabilité
    # principalement par j (ordre des nouvelles lignes)
    matches.sort(key=lambda t: (t[1], t[0]), reverse=False)
    return matches, old_unmatched, new_unmatched


def compute_diff(old_text: str,
                 new_text: str,
                 minor_change_threshold: float = 0.90) -> Tuple[str, List[Dict]]:
    """
    Renvoie (human_str, diff_json)
    - human_str : lignes ajoutées / modifiées / supprimées (monospace) avec n° de ligne
    - diff_json : liste d'opérations {type, line, content, ...}
      type ∈ {"insert","replace","delete"}
      line = n° de ligne dans le NOUVEAU texte (1-based) pour insert/replace,
             n° de ligne dans l’ANCIEN pour delete (clé 'old_line').
    Règles :
      - insert : toujours listé
      - replace : listé seulement si différence significative (similarité < minor_change_threshold)
      - delete : toujours listé
    """
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()

    sm = SequenceMatcher(None, old_lines, new_lines, autojunk=False)

    human_rows: List[str] = []
    diff_json: List[Dict] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        old_block = old_lines[i1:i2]
        new_block = new_lines[j1:j2]

        if tag in ("replace", "insert", "delete"):
            # Aligne intelligemment, même si tailles identiques
            matches, old_unmatched, new_unmatched = _align_block(
                old_block, new_block)

            # 1) REPLACE (pour les paires appariées)
            for i_rel, j_rel, sim in matches:
                old_content = old_block[i_rel]
                new_content = new_block[j_rel]
                new_abs_line = j1 + j_rel + 1      # 1-based dans le nouveau texte
                old_abs_line = i1 + i_rel + 1      # 1-based dans l’ancien texte

                # Ne journalise pas si c'est un changement mineur
                if _normalize_for_similarity(old_content) == _normalize_for_similarity(new_content):
                    # équivalent “tolérant” → rien
                    continue

                if sim < minor_change_threshold:
                    human_rows.append(f"~ Ligne {new_abs_line}. {new_content}")
                    diff_json.append({
                        "type": "replace",
                        "line": new_abs_line,
                        "old_line": old_abs_line,
                        "old_content": old_content,
                        "content": new_content,
                        "similarity": float(sim)
                    })
                # sinon (sim >= seuil) → on considère que c'est mineur → pas de log

            # 2) INSERT (nouvelles lignes non appariées)
            for j_rel in new_unmatched:
                new_content = new_block[j_rel]
                if not new_content.strip():
                    continue
                new_abs_line = j1 + j_rel + 1
                human_rows.append(f"+ Ligne {new_abs_line}. {new_content}")
                diff_json.append({
                    "type": "insert",
                    "line": new_abs_line,
                    "content": new_content
                })

            # 3) DELETE (anciennes lignes non appariées)
            for i_rel in old_unmatched:
                old_content = old_block[i_rel]
                if not old_content.strip():
                    continue
                old_abs_line = i1 + i_rel + 1
                human_rows.append(
                    f"- Ancienne ligne {old_abs_line}. {old_content}")
                diff_json.append({
                    "type": "delete",
                    "old_line": old_abs_line,
                    "old_content": old_content
                })

    human_str = "\n".join(human_rows)
    return human_str, diff_json


# ---------- Pare-feu HTR: détection de sorties OCR "en boucle" ----------
def _max_consecutive_run(tokens):
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
    Retourne (is_buggy, reason). Heuristiques pour détecter un bug OCR/HTR (répétitions absurdes).
    """
    if not ocr_text or not ocr_text.strip():
        return True, "ocr_text vide"

    # Tokens & lignes
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", ocr_text.lower())
    lines = [ln.strip().lower() for ln in ocr_text.splitlines() if ln.strip()]

    if len(tokens) < 5:
        # Très court → on laisse passer (c’est souvent une vraie petite note)
        return False, ""

    N = len(tokens)
    cnt = Counter(tokens)
    top_word, top_freq = cnt.most_common(1)[0]
    dom_ratio = top_freq / N                    # part du mot le plus fréquent
    uniq_ratio = len(cnt) / N                    # diversité de tokens
    # plus longue répétition consécutive
    run_max = _max_consecutive_run(tokens)
    chars = "".join(tokens)
    char_divers = len(set(chars)) / max(1, len(chars))

    # Lignes identiques
    line_cnt = Counter(lines)
    max_line_repeat = max(line_cnt.values()) if line_cnt else 0

    # Règles (ajuste les seuils au besoin)
    if dom_ratio >= 0.60:
        return True, f"mot dominant anormal ({top_word}={top_freq}/{N})"
    if run_max >= 5:
        return True, f"répétition consécutive anormale (run={run_max})"
    if uniq_ratio <= 0.20 and N >= 15:
        return True, f"faible diversité de tokens (uniq_ratio={uniq_ratio:.2f})"
    # if char_divers <= 0.20 and len(chars) >= 30:
    #     return True, f"faible diversité de caractères (char_divers={char_divers:.2f})"
    if max_line_repeat >= 5:
        return True, f"mêmes lignes répétées (x{max_line_repeat})"

    # Optionnel: si le cleaned est vide alors que l'OCR semble du spam
    if cleaned_text is not None and not cleaned_text.strip() and (dom_ratio > 0.5 or run_max >= 4):
        return True, "cleaned_text vide et OCR répétitif"

    return False, ""


def clean_added_text_for_ner(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()

        # Ignore complètement les lignes supprimées
        if re.match(r"^\-\s*Ancienne\s+ligne\s+\d+\.", line, flags=re.IGNORECASE):
            continue

        # Supprime uniquement le préfixe des lignes ajoutées
        new_line = re.sub(
            r"^\+\s*Ligne\s+\d+\.\s*",
            "",
            line,
            flags=re.IGNORECASE
        ).strip()

        if new_line:
            cleaned_lines.append(new_line)

    return "\n".join(cleaned_lines)


def reflow_sentences(text: str, width: int = 80) -> str:
    """
    Réarrange un texte multi-lignes en paragraphes correctement ponctués et wrappés.
    - n'ajoute un point qu'à la fin probable d'une phrase,
    - évite d'ajouter un point si la ligne se termine par une préposition/mot court,
    - joint les segments non terminés au segment suivant si la ligne suivante commence par une minuscule,
    - met une majuscule uniquement en début de phrase,
    - wrappe à <= width caractères sans couper les mots.
    """
    if not text or not text.strip():
        return text or ""

    non_terminal_words = {"sur", "à", "le", "la", "les", "des", "de", "du", "en", "et", "ou", "par", "pour", "avec", "au", "aux", "chez", "dans", "vers"}

    parts = [p.rstrip() for p in text.splitlines()]
    parts = [re.sub(r"\s+", " ", p).strip() for p in parts]

    merged_parts = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if not p:
            i += 1
            continue

        if re.search(r"[\.\?!]$", p):
            merged_parts.append(p)
            i += 1
            continue

        # lookahead to next non-empty part
        j = i + 1
        next_part = None
        while j < len(parts):
            if parts[j]:
                next_part = parts[j]
                break
            j += 1

        last_word = p.split()[-1].lower() if p.split() else ""

        if next_part:
            m = re.match(r"\s*([a-zà-ÿ])", next_part, flags=re.IGNORECASE)
            next_starts_lower = bool(m and m.group(1).islower())
        else:
            next_starts_lower = False

        if last_word in non_terminal_words and next_part:
            combined = p + " " + next_part
            merged_parts.append(combined)
            i = j + 1
            continue

        if next_part and next_starts_lower:
            combined = p + " " + next_part
            merged_parts.append(combined)
            i = j + 1
            continue

        merged_parts.append(p + ".")
        i += 1

    def cap_sentence(s: str) -> str:
        s = s.strip()
        s = re.sub(r"\s*([\.\?!])\s*", lambda m: m.group(1) + " ", s)
        s = re.sub(r"(^|[\.\?!]\s+)([a-zà-ÿ])", lambda m: m.group(1) + m.group(2).upper(), s, flags=re.IGNORECASE)
        return s.strip()

    sentences = [cap_sentence(s) for s in merged_parts if s.strip()]
    paragraph = " ".join(s.rstrip() for s in sentences)
    wrapped = textwrap.fill(paragraph.strip(), width=width, break_long_words=False, break_on_hyphens=False)
    return wrapped


def canonicalize_for_compare(s: str) -> List[str]:
    """Return a list of normalized tokens for comparison.
    - lowercased
    - remove diacritics
    - replace common punctuation by spaces
    - keep numbers (phone numbers) as tokens
    - split on whitespace
    """
    if not s:
        return []
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^0-9a-z+]+", " ", s)
    tokens = [t for t in s.split() if t]
    return tokens


def token_jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = sa & sb
    uni = sa | sb
    return len(inter) / len(uni)


def token_f1(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 1.0
    ca, cb = Counter(a), Counter(b)
    common = sum((ca & cb).values())
    if common == 0:
        return 0.0
    prec = common / sum(ca.values())
    rec = common / sum(cb.values())
    return 2 * prec * rec / (prec + rec)


def score_and_categorize_texts(a: str, b: str, weights=(0.5, 0.5), thresholds=None) -> Dict[str, Any]:
    """Calcule un score continu [0,1] entre deux textes et retourne une catégorisation.

    Retourne dict: {score, category, jaccard, f1, phone_ok}
    """
    thresholds = thresholds or {"identical": 0.90, "close": 0.75, "related": 0.50}

    ta = canonicalize_for_compare(a or "")
    tb = canonicalize_for_compare(b or "")

    ja = token_jaccard(ta, tb)
    fa = token_f1(ta, tb)

    def phone_tokens(ts: List[str]):
        return [t for t in ts if re.fullmatch(r"\+?\d{6,}", t)]

    pa = phone_tokens(ta)
    pb = phone_tokens(tb)
    phone_ok = False
    if pa and pb:
        def close_nums(x, y):
            diff = sum(1 for c1, c2 in zip(x, y) if c1 != c2)
            diff += abs(len(x) - len(y))
            return diff <= 2
        phone_ok = any(close_nums(x, y) for x in pa for y in pb)

    score = float(weights[0]) * ja + float(weights[1]) * fa
    score = max(0.0, min(1.0, score))

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
