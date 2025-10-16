import streamlit as st
import importlib.util
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import time
import sqlite3
import sys
import os

# Ajoute le dossier racine du projet au sys.path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backend.db import ensure_db
ensure_db()

db_path = os.path.join(os.path.dirname(__file__), "../backend/db.py")
spec = importlib.util.spec_from_file_location("db", db_path)
db = importlib.util.module_from_spec(spec)
sys.modules["db"] = db
spec.loader.exec_module(db)

delete_entry_by_id = db.delete_entry_by_id
delete_thread_by_note_id = db.delete_thread_by_note_id
list_notes = db.list_notes
list_notes_by_note_id = db.list_notes_by_note_id


# Pour lancer le front : streamlit run src/frontend/app_streamlit.py


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
        SELECT id, ts, note_id, transcription_brute, transcription_clean, texte_ajoute,
               img_path_proc, images,
               entite_GEO, entite_ACTOR, entite_DATETIME, entite_EVENT,
               entite_INFRASTRUCTURE, entite_OPERATING_CONTEXT,
               entite_PHONE_NUMBER, entite_ELECTRICAL_VALUE,
               evenement_id
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
        sql += " AND (transcription_clean LIKE ? OR transcription_brute LIKE ? OR texte_ajoute LIKE ?)"
        like = f"%{q}%"
        params += [like, like, like]

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
    limit = st.slider("Nombre de notes (max)", min_value=5,
                      max_value=200, value=50, step=5)
    q = st.text_input("Recherche texte (OCR, clean, ajouté)", value="")

    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("Depuis (date)", value=None)
    with col2:
        date_to = st.date_input("Jusqu'à (date)", value=None)

    # Conversion dates en timestamps
    ts_from = int(time.mktime(datetime.combine(
        date_from, datetime.min.time()).timetuple())) if date_from else None
    ts_to = int(time.mktime(datetime.combine(
        date_to, datetime.max.time()).timetuple())) if date_to else None

    st.caption(f"DB: `{DB_PATH}`")
    if st.button("Rafraîchir maintenant"):
        st.rerun()

    # Filtre note_id
    all_notes = list_notes(limit=200)
    note_ids = sorted({n["note_id"] for n in all_notes if n["note_id"]})

    if note_ids:
        selected_note_id = st.selectbox(
            "Filtrer par note_id",
            ["(toutes)"] + note_ids,
            index=0
        )
    else:
        selected_note_id = "(toutes)"
        st.caption("Aucune note disponible pour le moment.")

    # Filtres par entité
    st.subheader("Filtres entités")

    # Catégories disponibles
    available_categories = ["GEO", "DATETIME", "EVENT", "ACTOR",
                            "INFRASTRUCTURE", "OPERATING_CONTEXT", "PHONE_NUMBER", "ELECTRICAL_VALUE"]

    all_clauses = []
    all_params = []

    # Champs de recherche entités, un par catégorie
    for cat in available_categories:
        search_input = st.text_input(
            f"{cat}",
            key=f"search_{cat}",
            placeholder=f"Rechercher dans {cat}..."
        )

        if search_input.strip():
            terms = [t.strip() for t in search_input.split() if t.strip()]
            col_name = f"entite_{cat}"

            # AND entre les mots de la même catégorie
            category_clauses = [
                f"LOWER({col_name}) LIKE LOWER(?)" for _ in terms]
            category_where = " AND ".join(category_clauses)

            all_clauses.append(f"({category_where})")
            all_params.extend([f"%{term}%" for term in terms])

    # Exécution de la requête combinée si au moins un champ est rempli
    if all_clauses:
        final_where = " AND ".join(all_clauses)
        query = f"""
            SELECT evenement_id, COUNT(*) as note_count
            FROM notes_meta
            WHERE {final_where}
            AND evenement_id IS NOT NULL
            GROUP BY evenement_id
        """

        with get_conn(DB_PATH) as con:
            rows = con.execute(query, all_params).fetchall()

        st.markdown("---")
        st.write(
            f"**{len(rows)}** événements trouvés correspondant aux critères :")
        for r in rows:
            st.write(f"{r['evenement_id']} ({r['note_count']} notes)")

    # Filtres par entité
    st.subheader("Filtre événement")

    # Barre de recherche par événement ID
    event_id_search = st.text_input(
        "Événement ID",
        key="search_event_id",
    )

# Chargement des notes
if event_id_search.strip():
    # Si on a tapé un ID d'événement, on affiche les notes de cet événement uniquement
    with get_conn(DB_PATH) as con:
        rows = con.execute("""
            SELECT *
            FROM notes_meta
            WHERE LOWER(evenement_id) LIKE LOWER(?)
            ORDER BY ts ASC
        """, (f"%{event_id_search.strip()}%",)).fetchall()
    notes = [dict(r) for r in rows]

elif selected_note_id == "(toutes)":
    # Filtrage classique sur note_id
    notes = fetch_notes(limit=limit, ts_from=ts_from, ts_to=ts_to, q=q)
else:
    notes = list_notes_by_note_id(selected_note_id, limit=limit)


# Auto-refresh léger
if REFRESH_SECONDS > 0:
    st.experimental_set_query_params(_=int(time.time() // REFRESH_SECONDS))

# Bandeau résumé
st.markdown(f"**{len(notes)}** notes affichées")

st.markdown(
    "<h1 style='text-align: center; color: #E74C3C;'>Analyse des Notes</h1>",
    unsafe_allow_html=True
)

# Affichage en cartes
for n in notes:
    st.markdown("---")
    header_cols = st.columns([1, 4, 2])

    # Colonne gauche : méta
    with header_cols[0]:
        st.markdown(f"**ID**: {n['id']}")
        st.markdown(f"**TS**: {ts_human(n['ts'])}")
        if n.get("note_id"):
            st.caption(f"note_id: {n['note_id']}")
        if n.get("evenement_id"):
            st.caption(f"événement: {n['evenement_id'][:8]}...")

    # Colonne centre : textes
    with header_cols[1]:
        st.markdown("**Texte OCR brut**")
        st.markdown(f"```\n{n.get('transcription_brute') or '—'}\n```")

        st.markdown("**Texte clean**")
        st.markdown(f"```\n{n.get('transcription_clean') or '—'}\n```")

        st.markdown("**Texte ajouté**")
        st.markdown(f"```\n{n.get('texte_ajoute') or '—'}\n```")

    # Colonne droite : image
    with header_cols[2]:
        img = safe_image(n.get("img_path_proc"))
        if img:
            st.image(img, width='stretch', caption=os.path.basename(img))
        else:
            st.caption("Pas d'image disponible")

    # Images extraites
    with st.expander("Images extraites"):
        try:
            images = json.loads(n.get("images") or "[]")
        except Exception:
            images = []
        for img_path in images:
            img = safe_image(img_path)
            if img:
                st.image(img, caption=os.path.basename(img), width='stretch')
            else:
                st.caption(f"Image non disponible: {img_path}")
    # Entités extraites
    with st.expander("Entités extraites"):
        def parse_entities_field(field_name: str):
            val = n.get(field_name)
            if val:
                try:
                    return json.loads(val)
                except Exception:
                    return []
            return []

        entities_display = {
            "GEO": parse_entities_field("entite_GEO"),
            "ACTOR": parse_entities_field("entite_ACTOR"),
            "DATETIME": parse_entities_field("entite_DATETIME"),
            "EVENT": parse_entities_field("entite_EVENT"),
            "INFRASTRUCTURE": parse_entities_field("entite_INFRASTRUCTURE"),
            "OPERATING_CONTEXT": parse_entities_field("entite_OPERATING_CONTEXT"),
            "PHONE_NUMBER": parse_entities_field("entite_PHONE_NUMBER"),
            "ELECTRICAL_VALUE": parse_entities_field("entite_ELECTRICAL_VALUE"),
        }

        if not any(entities_display.values()):
            st.caption("Aucune entité stockée pour cette note.")
        else:
            for label, values in entities_display.items():
                st.write(f"**{label}** : {', '.join(values) or '—'}")

    # ----- Actions -----
    st.markdown("**Actions**")
    a1, a2, _ = st.columns([1, 1, 4])

    # Supprimer une entrée
    with a1:
        with st.popover("🗑️ Supprimer cette entrée"):
            st.caption(
                "Supprime uniquement CETTE ligne (id). Opération irréversible.")
            confirm1 = st.checkbox(
                f"Confirmer suppression id={n['id']}", key=f"del_id_ck_{n['id']}")
            if st.button("Supprimer", key=f"del_id_btn_{n['id']}", disabled=not confirm1):
                deleted = delete_entry_by_id(int(n["id"]), db_path=DB_PATH)
                st.success(f"{deleted} entrée supprimée (id={n['id']}).")
                st.rerun()

    # Supprimer toute une note_id
    with a2:
        disabled_thread = not n.get("note_id")
        with st.popover("🗑️ Supprimer TOUTE la note_id", disabled=disabled_thread):
            if disabled_thread:
                st.caption("Pas de note_id pour cette entrée.")
            else:
                st.caption(
                    f"Supprime toutes les entrées de note_id={n['note_id']}. Opération irréversible.")
                confirm2 = st.checkbox(
                    f"Confirmer suppression note_id={n['note_id']}", key=f"del_thread_ck_{n['id']}")
                if st.button("Supprimer tout", key=f"del_thread_btn_{n['id']}", disabled=not confirm2):
                    deleted = delete_thread_by_note_id(
                        n["note_id"], db_path=DB_PATH)
                    st.success(
                        f"{deleted} entrées supprimées (note_id={n['note_id']}).")
                    st.rerun()

# --------------------------Audio----------------------#
st.markdown(
    "<h1 style='text-align: center; color: #E74C3C;'>Analyse des Audios</h1>",
    unsafe_allow_html=True
)
tmp_dir = os.path.join("src/transcription/tmp")
audio_json_path = os.path.join(tmp_dir, "transcriptions_log.json")

# 1. Charger la liste des audios depuis le JSON
notes_audio = []
if os.path.exists(audio_json_path):
    with open(audio_json_path, "r") as f:
        data = json.load(f)
        # Chaque entrée du JSON devient une "note"
        for item in data:
            filename = item.get("filename", "")
            full_path = os.path.join(tmp_dir, filename)
            if os.path.exists(full_path):
                notes_audio.append({
                    "id": os.path.splitext(filename)[0],
                    "ts": item.get("start_time", ""),
                    "audio_path": full_path,
                    "transcription_audio_brute": item.get("transcription", "—"),
                    "transcription_audio_clean": item.get("transcription", "—"),
                    "commentaire_audio": "",
                })
else:
    st.warning(f"Aucun fichier JSON trouvé à {audio_json_path}")

# 2. Affichage des audios sous forme de cartes
for n in notes_audio:
    st.markdown("---")
    header_cols = st.columns([1, 4, 2])

    with header_cols[0]:
        st.markdown(f"**ID**: {n['id']}")
        st.markdown(f"**TS**: {n['ts']}")

    with header_cols[1]:
        st.markdown("**Transcription brute (audio)**")
        st.markdown(f"```\n{n.get('transcription_audio_brute') or '—'}\n```")

        st.markdown("**Transcription clean (audio)**")
        st.markdown(f"```\n{n.get('transcription_audio_clean') or '—'}\n```")

        st.markdown("**Commentaires audio**")
        st.markdown(f"```\n{n.get('commentaire_audio') or '—'}\n```")

    with header_cols[2]:
        audio_path = n.get("audio_path")
        if audio_path and os.path.exists(audio_path):
            st.audio(audio_path, format="audio/wav")
            st.caption(os.path.basename(audio_path))
        else:
            st.caption("Pas d’audio disponible")
    # ------------------------------------------------------#

    # ----- Actions -----
    st.markdown("**Actions**")
    a1, a2, a3 = st.columns([1, 1, 4])

    # Suppression d'une entrée (ligne)
    with a1:
        with st.popover("🗑️ Supprimer cette entrée"):
            st.caption(
                "Supprime uniquement CETTE ligne (id). Opération irréversible.")
            confirm1 = st.checkbox(
                f"Confirmer suppression id={n['id']}", key=f"del_id_ck_{n['id']}")
            if st.button("Supprimer", key=f"del_id_btn_{n['id']}", disabled=not confirm1):
                deleted = delete_entry_by_id(int(n["id"]), db_path=DB_PATH)
                st.success(f"{deleted} entrée supprimée (id={n['id']}).")
                st.rerun()

    # Suppression de tout le fil (même note_id)
    with a2:
        disabled_thread = not n.get("note_id")
        with st.popover("🗑️ Supprimer TOUTE la note_id", disabled=disabled_thread):
            if disabled_thread:
                st.caption("Pas de note_id pour cette entrée.")
            else:
                st.caption(
                    f"Supprime toutes les entrées de note_id={n['note_id']}. Opération irréversible.")
                confirm2 = st.checkbox(
                    f"Confirmer suppression note_id={n['note_id']}", key=f"del_thread_ck_{n['id']}")
                if st.button("Supprimer tout", key=f"del_thread_btn_{n['id']}", disabled=not confirm2):
                    deleted = delete_thread_by_note_id(
                        n["note_id"], db_path=DB_PATH)
                    st.success(
                        f"{deleted} entrées supprimées (note_id={n['note_id']}).")
                    st.rerun()

# Rafraîchissement automatique
if REFRESH_SECONDS > 0:
    time.sleep(REFRESH_SECONDS)
    st.rerun()
