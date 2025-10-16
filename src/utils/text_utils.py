# -*- coding: utf-8 -*-
import re
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
    s = re.sub(r"\s*[-–—]\s*", "-", s)       # "Caen - Cherbourg" -> "caen-cherbourg"
    s = re.sub(r"\s*:\s*", ":", s)
    s = re.sub(r"\s*;\s*", ";", s)
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\s*\.\s*", ".", s)

    # Écrase espaces multiples
    s = re.sub(r"\s+", " ", s)
    return s

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_for_similarity(a), _normalize_for_similarity(b)).ratio()

def _align_block(old_block: List[str], new_block: List[str]) -> Tuple[List[Tuple[int,int,float]], List[int], List[int]]:
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
    matches.sort(key=lambda t: (t[1], t[0]), reverse=False)  # principalement par j (ordre des nouvelles lignes)
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
            matches, old_unmatched, new_unmatched = _align_block(old_block, new_block)

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

def is_htr_buggy(ocr_text: str, cleaned_text: str = "") -> (bool, str):
    """
    Retourne (is_buggy, reason). Heuristiques pour détecter un bug OCR/HTR (répétitions absurdes).
    """
    if not ocr_text or not ocr_text.strip():
        return True, "ocr_text vide"

    # Tokens & lignes
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", ocr_text.lower())
    lines   = [ln.strip().lower() for ln in ocr_text.splitlines() if ln.strip()]

    if len(tokens) < 5:
        # Très court → on laisse passer (c’est souvent une vraie petite note)
        return False, ""

    N = len(tokens)
    cnt = Counter(tokens)
    top_word, top_freq = cnt.most_common(1)[0]
    dom_ratio   = top_freq / N                    # part du mot le plus fréquent
    uniq_ratio  = len(cnt) / N                    # diversité de tokens
    run_max     = _max_consecutive_run(tokens)    # plus longue répétition consécutive
    chars       = "".join(tokens)
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
