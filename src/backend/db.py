
# Ajoute le dossier racine du projet au sys.path pour permettre les imports internes
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import sqlite3
import time
from typing import Optional, List, Dict, Tuple
from difflib import SequenceMatcher

DB_PATH = os.environ.get("RTE_DB_PATH", "data/db/notes.sqlite")

def _resolve_db_path(db_path: Optional[str]) -> str:
    return db_path or DB_PATH

def ensure_db(db_path: str = DB_PATH):
    db_path = _resolve_db_path(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("""
    CREATE TABLE IF NOT EXISTS notes_meta (
      id                    INTEGER PRIMARY KEY AUTOINCREMENT,
      ts                    INTEGER NOT NULL,
      note_id               TEXT,
      transcription_brute   TEXT,
      transcription_clean   TEXT,
      texte_ajoute          TEXT,
      img_path_proc         TEXT,
      images                TEXT,
      raw_json              TEXT
    );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_notes_meta_ts ON notes_meta(ts);")
    con.commit()
    con.close()

def insert_note_meta(meta: dict, img_path_proc: Optional[str] = None, db_path: str = DB_PATH) -> int:
    """
    Insère un enregistrement à partir d'un dict meta.
    Retourne l'id auto-incrémenté.
    """
    ensure_db(db_path)
    now = int(time.time())
    row = (
        now,
        meta.get("note_id"),
        meta.get("transcription_brute"),
        meta.get("transcription_clean"),
        meta.get("texte_ajoute"),
        img_path_proc,
        json.dumps(meta.get("images", []), ensure_ascii=False),
        json.dumps(meta, ensure_ascii=False)
    )
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO notes_meta
        (ts, note_id, transcription_brute, transcription_clean, texte_ajoute, img_path_proc, images, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, row)
    con.commit()
    new_id = cur.lastrowid
    con.close()
    return new_id

def list_notes(limit: int = 20, db_path: str = DB_PATH) -> List[Dict]:
    ensure_db(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("""
        SELECT id, ts, note_id, transcription_brute, transcription_clean, texte_ajoute, img_path_proc, images
        FROM notes_meta
        ORDER BY ts DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

def get_last_note_text(db_path: str = DB_PATH) -> Tuple[Optional[str], Optional[int]]:
    """
    Récupère le texte nettoyé (transcription_clean) de la dernière note ajoutée en base.
    """
    ensure_db(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("""
        SELECT id, transcription_clean FROM notes_meta
        ORDER BY ts DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    con.close()
    if row and row["transcription_clean"]:
        return row["transcription_clean"], row["id"]
    return None, None

def is_same_note(clean_text: str, db_path: str = DB_PATH, threshold: float = 0.7) -> Optional[int]:
    """
    Compare le texte nettoyé fourni à celui de la dernière note en base.
    Retourne l'id de la note si la similarité > seuil, sinon None.
    """
    ensure_db(db_path)
    last_summary, last_id = get_last_note_text(db_path)
    if last_summary is None:
        return None
    min_len = min(len(clean_text), len(last_summary))
    a_trunc = clean_text[:min_len]
    b_trunc = last_summary[:min_len]
    ratio = SequenceMatcher(None, a_trunc, b_trunc).ratio()
    if ratio >= threshold:
        return last_id
    return None

def get_last_image_for_notes(db_path: str = DB_PATH) -> Dict[str, str]:
    """
    Retourne un dict {note_id: img_path_proc} pour la dernière image de chaque note_id.
    """
    ensure_db(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("""
        SELECT note_id, img_path_proc, MAX(ts) as ts
        FROM notes_meta
        WHERE note_id IS NOT NULL
        GROUP BY note_id
    """)
    result = {}
    for row in cur.fetchall():
        if row["note_id"] and row["img_path_proc"]:
            result[row["note_id"]] = row["img_path_proc"]
    con.close()
    return result

def get_last_text_for_notes(db_path: str = DB_PATH) -> Dict[str, str]:
    """
    Retourne un dict {note_id: transcription_clean} pour le texte clean de la dernière note de chaque note_id.
    """
    ensure_db(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("""
        SELECT note_id, transcription_clean, MAX(ts) as ts
        FROM notes_meta
        WHERE note_id IS NOT NULL
        GROUP BY note_id
    """)
    result = {}
    for row in cur.fetchall():
        if row["note_id"] and row["transcription_clean"]:
            result[row["note_id"]] = row["transcription_clean"]
    con.close()
    return result

def find_similar_note(clean_text: str, db_path: str = DB_PATH, threshold: float = 0.3) -> Optional[str]:
    """
    Compare le texte clean à toutes les dernières notes en base.
    Retourne le note_id si la similarité > seuil, sinon None.
    """
    last_texts = get_last_text_for_notes(db_path)
    for note_id, last_summary in last_texts.items():
        min_len = min(len(clean_text), len(last_summary))
        a_trunc = clean_text[:min_len]
        b_trunc = last_summary[:min_len]
        ratio = SequenceMatcher(None, a_trunc, b_trunc).ratio()
        if ratio >= threshold:
            return note_id
    return None


def get_added_text(old_text: str,
                   new_text: str,
                   minor_change_threshold: float = 0.90) -> str:
    """
    Back-compat : ne renvoie que la version humaine (monospace) des changements.
    """
    human, _ = compute_diff(old_text, new_text, minor_change_threshold=minor_change_threshold)
    return human


def get_last_note_meta(db_path: str = DB_PATH) -> Optional[Dict]:
    """
    Retourne le dernier enregistrement complet de notes_meta.
    """
    ensure_db(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("""
        SELECT * FROM notes_meta
        ORDER BY ts DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    con.close()
    if row:
        return dict(row)
    return None
    
def clear_notes_meta(db_path: str = DB_PATH):
    """
    Supprime toutes les lignes de la table notes_meta et réinitialise l'AUTOINCREMENT.
    """
    ensure_db(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("DELETE FROM notes_meta;")
    # Réinitialise l'AUTOINCREMENT
    cur.execute("DELETE FROM sqlite_sequence WHERE name='notes_meta';")
    con.commit()
    con.close()
    print(f"La base de données '{db_path}' a été vidée (notes_meta clear, AUTOINCREMENT reset).")


def delete_entry_by_id(entry_id: int, db_path: str = DB_PATH) -> int:
    """Supprime UNE entrée (ligne) par id. Retourne le nb de lignes supprimées (0 ou 1)."""
    ensure_db(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("DELETE FROM notes_meta WHERE id = ?", (entry_id,))
    deleted = cur.rowcount
    con.commit()
    con.close()
    return deleted

def delete_thread_by_note_id(note_id: str, db_path: str = DB_PATH) -> int:
    """Supprime TOUTES les entrées liées à un note_id. Retourne le nb de lignes supprimées."""
    ensure_db(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("DELETE FROM notes_meta WHERE note_id = ?", (note_id,))
    deleted = cur.rowcount
    con.commit()
    con.close()
    return deleted


def list_notes_by_note_id(note_id: str, db_path: str = DB_PATH, limit: int = 50) -> List[Dict]:
    """
    Retourne toutes les entrées associées à un note_id donné, 
    triées par timestamp descendant.
    """
    ensure_db(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("""
        SELECT id, ts, note_id, transcription_brute, transcription_clean, texte_ajoute, img_path_proc, images
        FROM notes_meta
        WHERE note_id = ?
        ORDER BY ts DESC
        LIMIT ?
    """, (note_id, limit))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows
