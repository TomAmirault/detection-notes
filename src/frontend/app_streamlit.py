import os
import sqlite3
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st

# --- Config ---
DB_PATH = os.environ.get("RTE_DB_PATH", "data/db/notes.sqlite")
PAGE_TITLE = "RTE Notes — V0"
REFRESH_SECONDS = 5  # auto refresh (0 = désactiver)

# --- Utils DB ---
def get_conn(db_path: str):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con

def fetch_notes(limit: int = 50,
                ts_from: Optional[int] = None,
                ts_to: Optional[int] = None,
                q: str = "") -> List[Dict[str, Any]]:
    sql = """
        SELECT id, ts, note_id, image_type, short_description, summary,
               rappels, incidents, call_recap, additional_info, img_path_proc
        FROM notes_meta
        WHERE 1=1
    """
    params = []
    if ts_from is not None:
        sql += " AND ts >= ?"
        params.append(ts_from)
    if ts_to is not None:
        sql += " AND ts <= ?"
        params.append(ts_to)
    if q:
        # recherche simple sur quelques champs
        sql += " AND (summary LIKE ? OR short_description LIKE ? OR incidents LIKE ? OR call_recap LIKE ?)"
        like = f"%{q}%"
        params += [like, like, like, like]

    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)

    with get_conn(DB_PATH) as con:
        rows = con.execute(sql, params).fetchall()
    return [dict(r) for r in rows]

def ts_human(ts: int) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)

def safe_image(path: Optional[str]) -> Optional[str]:
    if path and os.path.exists(path):
        return path
    return None

# --- UI ---
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

# Sidebar filtres
with st.sidebar:
    st.subheader("Filtres")
    limit = st.slider("Nombre de notes (max)", min_value=5, max_value=200, value=50, step=5)
    q = st.text_input("Recherche texte (summary, incidents...)", value="")
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("Depuis (date)", value=None)
    with col2:
        date_to = st.date_input("Jusqu'à (date)", value=None)

    # Conversion dates → timestamps (en début/fin de journée)
    ts_from = int(time.mktime(datetime.combine(date_from, datetime.min.time()).timetuple())) if date_from else None
    ts_to = int(time.mktime(datetime.combine(date_to, datetime.max.time()).timetuple())) if date_to else None

    st.caption(f"DB: `{DB_PATH}`")
    if st.button("Rafraîchir maintenant"):
        st.rerun()

# Auto-refresh léger
if REFRESH_SECONDS > 0:
    st.experimental_set_query_params(_=int(time.time() // REFRESH_SECONDS))

# Chargement notes
notes = fetch_notes(limit=limit, ts_from=ts_from, ts_to=ts_to, q=q)

# Bandeau résumé
st.markdown(f"**{len(notes)}** notes affichées")

# Affichage en cartes
for n in notes:
    st.markdown("---")
    header_cols = st.columns([1, 4, 2])
    with header_cols[0]:
        st.markdown(f"**ID**: {n['id']}")
        st.markdown(f"**TS**: {ts_human(n['ts'])}")
        if n.get("note_id"):
            st.caption(f"note_id: {n['note_id']}")
    with header_cols[1]:
        st.markdown(f"**{n.get('short_description') or '(sans description)'}**")
        if n.get("summary"):
            st.write(n["summary"])
    with header_cols[2]:
        img = safe_image(n.get("img_path_proc"))
        if img:
            st.image(img, use_column_width=True, caption=os.path.basename(img))
        else:
            st.caption("Pas d'image disponible")

    # Détails en accordéon
    with st.expander("Détails"):
        left, right = st.columns(2)
        with left:
            st.markdown("**Rappels**")
            st.write(n.get("rappels") or "—")
            st.markdown("**Incidents**")
            st.write(n.get("incidents") or "—")
        with right:
            st.markdown("**Compte-rendu d'appel**")
            st.write(n.get("call_recap") or "—")
            st.markdown("**Infos supplémentaires**")
            st.write(n.get("additional_info") or "—")
