import os, json, sqlite3, time
from typing import Optional


DB_PATH = os.environ.get("RTE_DB_PATH", "data/db/notes.sqlite")

def ensure_db(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("""
    CREATE TABLE IF NOT EXISTS notes_meta (
      id               INTEGER PRIMARY KEY AUTOINCREMENT,
      ts               INTEGER NOT NULL,
      note_id          TEXT,
      image_type       TEXT,
      short_description TEXT,
      summary          TEXT,
      rappels          TEXT,
      incidents        TEXT,
      call_recap       TEXT,
      additional_info  TEXT,
      img_path_proc    TEXT,
      raw_json         TEXT
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
        meta.get("image_type"),
        meta.get("short_description"),
        meta.get("summary"),
        meta.get("rappels"),
        meta.get("incidents"),
        meta.get("call_recap"),
        meta.get("additional_info"),
        img_path_proc,
        json.dumps(meta, ensure_ascii=False)
    )
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO notes_meta
        (ts, note_id, image_type, short_description, summary, rappels, incidents, call_recap, additional_info, img_path_proc, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, row)
    con.commit()
    new_id = cur.lastrowid
    con.close()
    return new_id

def list_notes(limit: int = 20, db_path: str = DB_PATH) -> list[dict]:
    ensure_db(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("""
        SELECT id, ts, note_id, image_type, short_description, summary, rappels, incidents, call_recap, additional_info, img_path_proc
        FROM notes_meta
        ORDER BY ts DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows
