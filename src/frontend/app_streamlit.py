import streamlit as st
import importlib.util
from typing import List, Dict, Any, Optional
from datetime import datetime
from PIL import Image
import json
import time
import sqlite3

# Ajoute la racine du projet au sys.path pour permettre les imports internes
import sys
import os
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

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
list_notes_by_evenement_id = db.list_notes_by_evenement_id


# Pour lancer le front : streamlit run src/frontend/app_streamlit.py

# --- Config ---
DB_PATH = os.environ.get("RTE_DB_PATH", "data/db/notes.sqlite")
PAGE_TITLE = "Historique Chronologique"
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
    SELECT id, ts, note_id, transcription_brute, transcription_clean, texte_ajoute, confidence_score,
           img_path_proc,
               entite_GEO, entite_ACTOR, entite_DATETIME, entite_EVENT,
               entite_INFRASTRUCTURE, entite_OPERATING_CONTEXT,
               entite_PHONE_NUMBER, entite_ELECTRICAL_VALUE, entite_ABBREVIATION_UNKNOWN,
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

# --- Fonctions utilitaires ---

CONFIG_PATH = "src/audio/pause_status.json"
def load_config():
    """Charge le fichier pause_status.json, ou le crée s'il n'existe pas."""
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump({"pause": True}, f)
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(cfg):
    """Sauvegarde la configuration dans pause_status.json."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)

def toggle_pause():
    """Inverse la valeur de pause dans pause_status.json."""
    cfg = load_config()
    cfg["pause"] = not cfg["pause"]
    save_config(cfg)


def evaluation_fiabilite(score: float) -> str:
    """
    Retourne un niveau de fiabilité textuel selon un score de confiance Whisper.
    """

    if score >= 0.90:
        return "Très fiable"
    elif score >= 0.75:
        return "Fiable"
    elif score >= 0.50:
        return "Fiable"
    elif score >= 0.30:
        return "Peu fiable"
    else:
        return "Pas fiable"



# --- UI ---
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)


# --- AJOUT DU BOUTON DE NAVIGATION ---
# Placez ce bloc où vous souhaitez voir le bouton
if st.button("Consulter la synthèse"):
    st.switch_page("pages/nouvelle_page.py") # Chemin vers votre nouvelle page
# ------------------------------------


cfg = load_config()
etat = "▶️ En lecture (True)" if cfg["pause"] else "⏸️ En pause (False)"
st.write(f"**État actuel :** {etat}")


if st.button("Pause / Lecture", on_click=toggle_pause):
    st.rerun()


# Sidebar filtres
with st.sidebar:
    st.subheader("Filtres généraux")
    limit = st.slider("Nombre de notes (max)", min_value=5,
                      max_value=200, value=50, step=5)
    q = st.text_input("Recherche texte (clean, ajouté)", value="")

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
    st.subheader("Filtrer par note")
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
        
    # Filtres par événement
    st.subheader("Filtrer par événement")
    evenement_ids = list(set(sorted([n["evenement_id"] for n in all_notes if n["evenement_id"]])))
    if note_ids:
        selected_evenement_id = st.selectbox(
            "Filtrer par evenement_id",
            ["(tous)"] + evenement_ids,
            index=0
        )
    else:
        selected_evenement_id = "(tous)"
        st.caption("Aucun événement disponible pour le moment.")
        
    # Filtres par entité
    st.subheader("Recherche entités")
    entity_query = st.text_input(
        "Rechercher dans toutes les entités (mots séparés par des espaces)",
        key="search_all_entities",
    )

    notes_entity_filtered = None

    if entity_query.strip():
        # Découpage en mots-clés
        terms = [t.strip() for t in entity_query.split() if t.strip()]

        entity_columns = [
            "entite_GEO", "entite_DATETIME", "entite_EVENT", "entite_ACTOR",
            "entite_INFRASTRUCTURE", "entite_OPERATING_CONTEXT",
            "entite_PHONE_NUMBER", "entite_ELECTRICAL_VALUE", "entite_ABBREVIATION_UNKNOWN"
        ]

        # Pour chaque terme, on crée un OR entre toutes les colonnes entités
        # Puis on combine les différents termes avec AND (chaque mot doit apparaître quelque part)
        term_clauses = []
        params = []

        for term in terms:
            like_clauses = [f"LOWER({col}) LIKE LOWER(?)" for col in entity_columns]
            term_clause = "(" + " OR ".join(like_clauses) + ")"
            term_clauses.append(term_clause)
            params.extend([f"%{term}%"] * len(entity_columns))

        final_where = " AND ".join(term_clauses)

        query = f"""
            SELECT *
            FROM notes_meta
            WHERE {final_where}
            ORDER BY ts DESC
        """

        with get_conn(DB_PATH) as con:
            notes_entity_filtered = [dict(r) for r in con.execute(query, params).fetchall()]

        st.markdown("---")
        st.write(f"**{len(notes_entity_filtered)}** notes trouvées correspondant aux critères entités.")


# Chargement des notes
if notes_entity_filtered is not None:
    notes = notes_entity_filtered
elif selected_evenement_id != "(tous)":
    # Filtrage sur evenement_id
    notes = list_notes_by_evenement_id(selected_evenement_id, limit=limit)
elif selected_note_id == "(toutes)":
    # Filtrage classique sur note_id
    notes = fetch_notes(limit=limit, ts_from=ts_from, ts_to=ts_to, q=q)
else:
    notes = list_notes_by_note_id(selected_note_id, limit=limit)

# Auto-refresh léger
if REFRESH_SECONDS > 0:
    st.query_params["_"] = str(int(time.time() // REFRESH_SECONDS))

# Bandeau résumé
st.markdown(f"**{len(notes)}** notes affichées")

st.markdown(
    "<h1 style='text-align: center; color: #E74C3C;'>Informations extraites (texte & audio)</h1>",
    unsafe_allow_html=True
)

with open("src/frontend/log.txt", "w") as f:  
    f.write("Nouveau contenu du log.\n\n")

# Affichage en cartes
for n in notes:
    st.markdown("---")
    score_confiance = n.get("confidence_score")

    
    # Colonnes principales : meta, résumé et entités
    cols = st.columns([1, 3, 2])

    img_path = safe_image(n.get("img_path_proc"))
    trans = n.get("transcription_clean")
    
    tmp_dir = os.path.join(os.path.join(os.path.dirname(__file__), "../../src/audio/tmp"))
    audio_json_path = os.path.join(tmp_dir, "transcriptions_log.json")
    
    audio_path = None
    audio_score = 1
    audio_start = None
    if os.path.exists(audio_json_path):
        with open(audio_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            if trans == d.get("transcription_clean"):
                audio_path = tmp_dir + "/" + d.get("filename")
                audio_score = d.get("score")
                audio_start = d.get("start_time")
                

    if audio_score > 0.3 :
        # Colonne gauche : méta
        with cols[0]:
            st.markdown(f"**ID:** {n['id']}")
            
            if audio_path and os.path.exists(audio_path):
                st.markdown(f"**TS:** {audio_start}")
                with open("src/frontend/log.txt", "a") as f:  
                    f.write(f"{audio_start}\n")

            elif img_path:
                st.markdown(f"**TS:** {ts_human(n['ts'])}")
                with open("src/frontend/log.txt", "a") as f:  
                    f.write(f"{ts_human(n['ts'])}\n")

            if n.get("note_id"):
                st.caption(f"note_id: {n['note_id']}")
            if n.get("evenement_id"):
                st.caption(f"événement: {n['evenement_id']}")

        # Colonne centre : Informations ajoutées
        with cols[1]:
            st.markdown("**Informations ajoutées**")
            st.markdown(f"```\n{n.get('texte_ajoute') or '—'}\n```")
            with open("src/frontend/log.txt", "a") as f:  # "a" = append (ajouter à la fin du fichier)
                f.write(f"{n.get('texte_ajoute') or '—'}\n")
            
        # Colonne droite : entités
        with cols[2]:
            def parse_entities_field(field_name: str):
                val = n.get(field_name)
                if val:
                    try:
                        return json.loads(val)
                    except:
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

            entity_colors = {
                "GEO": "#F5A6A7",
                "ACTOR": "#7FB3D5",
                "DATETIME": "#88D8C0",
                "EVENT": "#FFD699",
                "INFRASTRUCTURE": "#FFF0A6",
                "OPERATING_CONTEXT": "#C39BD3",
                "PHONE_NUMBER": "#7F8C8D",
                "ELECTRICAL_VALUE": "#F7A8D1",
            }


            all_entities_html = []
            for label, values in entities_display.items():
                for v in values:
                    v_display = v if len(v) <= 30 else v[:27] + "…"
                    all_entities_html.append(
                        f'<span style="background-color:{entity_colors[label]}; color:#000; padding:3px 8px; border-radius:5px; margin:2px; display:inline-block;">{v_display}</span>'
                    )
            if all_entities_html:
                st.markdown(" ".join(all_entities_html), unsafe_allow_html=True)
            else:
                st.caption("Aucune entité")

        # --- Menu déroulant pour détails complets ---
        with st.expander("Voir plus de détails et fichiers"):
            # Colonnes dans l'expander
            detail_cols = st.columns([2,2])
            
            # Image ou audio
            img_path = safe_image(n.get("img_path_proc"))
        
            trans = n.get("transcription_clean")
            
            tmp_dir = os.path.join(os.path.join(os.path.dirname(__file__), "../../src/audio/tmp"))
            audio_json_path = os.path.join(tmp_dir, "transcriptions_log.json")
            
            audio_path = None
            audio_score = None
            if os.path.exists(audio_json_path):
                with open(audio_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for d in data:
                    if trans == d.get("transcription_clean"):
                        audio_path = tmp_dir + "/" + d.get("filename")
                        audio_score = d.get("score")
                        try:
                            audio_score = float(audio_score) if audio_score is not None else None
                        except Exception:
                            audio_score = None

            
            with detail_cols[0]:

                if img_path:
                    st.markdown(f"**Fiabilité de la transcription:**  **{evaluation_fiabilite(score_confiance if score_confiance is not None else 0.0)}** (Score = {round(score_confiance,2)})")


                elif audio_path and os.path.exists(audio_path):
                    st.markdown(f"**Fiabilité de la transcription:**  **{evaluation_fiabilite(audio_score if audio_score is not None else 0.0)}** (Score = {audio_score})")

                st.markdown("**Transcription brute**")
                st.markdown(f"```\n{n.get('transcription_brute') or '—'}\n```")
                st.markdown("**Transcription nettoyée**")
                st.markdown(f"```\n{n.get('transcription_clean') or '—'}\n```")
                
            with detail_cols[1]:

                if img_path:
                    img = Image.open(img_path)
                    if img.width > img.height:
                        img = img.rotate(-90, expand=True)
                    st.image(img, width='stretch', caption=os.path.basename(img_path))
                elif audio_path and os.path.exists(audio_path):
                    st.audio(audio_path, format="audio/wav")
                    st.caption(os.path.basename(audio_path))
                else:
                    st.caption("Pas d'image/audio disponible")

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





# Rafraîchissement automatique
if REFRESH_SECONDS > 0:
    time.sleep(REFRESH_SECONDS)
    st.rerun()
