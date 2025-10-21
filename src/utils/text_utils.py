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


def _try_split_merge_matches(old_block: List[str], new_block: List[str], split_thresh: float = 0.85):
    """
    Detecte localement des splits (1 old -> 2 new) et merges (2 old -> 1 new).
    Retourne:
      - matches: liste de tuples ((i_start,i_end), (j_start,j_end), kind) avec kind in {"1to2","2to1"}
      - used_old: set d'indices old consommés
      - used_new: set d'indices new consommés
    """
    n_old, n_new = len(old_block), len(new_block)
    used_old, used_new = set(), set()
    matches = []

    # --- 1 -> 2 (split)
    for i in range(n_old):
        if i in used_old:
            continue
        best = None
        for j in range(n_new - 1):
            if j in used_new or (j + 1) in used_new:
                continue
            sim = _similarity(old_block[i], f"{new_block[j]} {new_block[j+1]}".strip())
            if best is None or sim > best[0]:
                best = (sim, i, j)
        if best and best[0] >= split_thresh:
            sim, i0, j0 = best
            matches.append(((i0, i0), (j0, j0 + 1), "1to2"))
            used_old.add(i0)
            used_new.update({j0, j0 + 1})

    # --- 2 -> 1 (merge)
    for j in range(n_new):
        if j in used_new:
            continue
        best = None
        for i in range(n_old - 1):
            if i in used_old or (i + 1) in used_old:
                continue
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
            # 0) split/merge detection local (1->2, 2->1)
            split_matches, used_old, used_new = _try_split_merge_matches(old_block, new_block, split_thresh=0.85)
            for (i_start, i_end), (j_start, j_end), kind in split_matches:
                if kind == "1to2":
                    a = old_block[i_start]
                    b1, b2 = new_block[j_start], new_block[j_start + 1]

                    sim1 = _similarity(a, b1)
                    if _normalize_for_similarity(a) != _normalize_for_similarity(b1) and sim1 < minor_change_threshold and b1.strip():
                        line_new = j1 + j_start + 1
                        human_rows.append(f"~ Ligne {line_new}. {b1}")
                        diff_json.append({
                            "type": "replace",
                            "line": line_new,
                            "old_line": i1 + i_start + 1,
                            "old_content": a,
                            "content": b1,
                            "similarity": float(sim1),
                            "note": "split(1→2)-part1"
                        })

                    sim2 = _similarity(a, b2)
                    if _normalize_for_similarity(a) != _normalize_for_similarity(b2) and sim2 < minor_change_threshold and b2.strip():
                        line_new = j1 + j_start + 2
                        human_rows.append(f"~ Ligne {line_new}. {b2}")
                        diff_json.append({
                            "type": "replace",
                            "line": line_new,
                            "old_line": i1 + i_start + 1,
                            "old_content": a,
                            "content": b2,
                            "similarity": float(sim2),
                            "note": "split(1→2)-part2"
                        })

                elif kind == "2to1":
                    a1, a2 = old_block[i_start], old_block[i_end]
                    b = new_block[j_start]
                    sim = _similarity(f"{a1} {a2}".strip(), b)
                    if _normalize_for_similarity(f"{a1} {a2}") != _normalize_for_similarity(b) and sim < minor_change_threshold and b.strip():
                        line_new = j1 + j_start + 1
                        human_rows.append(f"~ Ligne {line_new}. {b}")
                        diff_json.append({
                            "type": "replace",
                            "line": line_new,
                            "old_line": [i1 + i_start + 1, i1 + i_end + 1],
                            "old_content": f"{a1} {a2}",
                            "content": b,
                            "similarity": float(sim),
                            "note": "merge(2→1)"
                        })

            # 1) retire ce qui a été consommé par split/merge
            old_rest_map = [k for k in range(len(old_block)) if k not in used_old]
            new_rest_map = [k for k in range(len(new_block)) if k not in used_new]
            old_rest = [old_block[k] for k in old_rest_map]
            new_rest = [new_block[k] for k in new_rest_map]

            # 2) aligne le reste 1<->1
            matches, old_unmatched_rel, new_unmatched_rel = _align_block(old_rest, new_rest)

            # remappe indices relatifs
            remapped_matches = []
            for i_rel2, j_rel2, sim in matches:
                i_rel_orig = old_rest_map[i_rel2] if i_rel2 < len(old_rest_map) else None
                j_rel_orig = new_rest_map[j_rel2] if j_rel2 < len(new_rest_map) else None
                remapped_matches.append((i_rel_orig, j_rel_orig, sim))

            old_unmatched = [old_rest_map[i] for i in old_unmatched_rel]
            new_unmatched = [new_rest_map[j] for j in new_unmatched_rel]

            # 3) EMIT REPLACE pour les paires appariées restantes (align 1↔1)
            for i_rel, j_rel, sim in remapped_matches:
                old_content = old_block[i_rel]
                new_content = new_block[j_rel]
                new_abs_line = j1 + j_rel + 1
                old_abs_line = i1 + i_rel + 1
                if _normalize_for_similarity(old_content) == _normalize_for_similarity(new_content):
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

            # 3) tentative de re-pairing gourmand pour éviter DELETE+INSERT
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

            # 4) EMIT REPLACE pour paires re-appariées
            for i_rel, j_rel, sim in re_pairs:
                old_content = old_block[i_rel]
                new_content = new_block[j_rel]
                new_abs_line = j1 + j_rel + 1
                old_abs_line = i1 + i_rel + 1
                if _normalize_for_similarity(old_content) == _normalize_for_similarity(new_content):
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

            # 5) INSERT pour les nouvelles lignes restantes
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

            # 6) DELETE pour les anciennes lignes restantes
            for i_rel in old_unmatched:
                old_content = old_block[i_rel]
                if not old_content.strip():
                    continue
                old_abs_line = i1 + i_rel + 1
                human_rows.append(f"- Ancienne ligne {old_abs_line}. {old_content}")
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

        # Supprime les préfixes des lignes ajoutées / modifiées / supprimées
        # Exemples de préfixes attendus: "+ Ligne 3. ", "- Ancienne ligne 2. ", "~ Ligne 5. "
        # Nous voulons:
        #  - ignorer complètement les lignes supprimées (déjà géré plus haut),
        #  - enlever les préfixes + / ~ et retourner le reste nettoyé.
        new_line = re.sub(
            r"^[+~]\s*Ligne\s+\d+\.\s*",
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
