import streamlit as st
from datetime import datetime, timedelta
import re
import os
from src.frontend.mistral import synthèse
from src.utils.text_utils import clean_added_text
import markdown2

MAX_GROUP_DURATION = 2 #minutes
MAX_PAUSE = 30  # seconde         


@st.cache_data
def get_synthese(contenu):
    return synthèse(contenu)


st.set_page_config(page_title="Cartes Logs", layout="wide")



with open("src/frontend/log.txt", "r", encoding="utf-8") as f:
    lignes = f.readlines()

# === Extraction des blocs temporels ===
logs = []
for ligne in lignes:
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(.*)", ligne.strip())
    if match:
        timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        texte = match.group(2).strip()
        logs.append((timestamp, texte))
    elif logs:
        logs[-1] = (logs[-1][0], logs[-1][1] + "\n" + ligne.strip())




logs.sort(key=lambda x: x[0])

groupes = []
current_group = []
start_time = None

for i, (date, text) in enumerate(logs):
    if not current_group:
        current_group.append(text)
        start_time = date
        end_time = date
    else:
        delta = date - end_time
        if delta.total_seconds() > MAX_PAUSE or (date - start_time).total_seconds() > MAX_GROUP_DURATION*60:
            groupes.append([start_time, end_time, "\n".join(current_group)])
            current_group = [text]
            start_time = date
            end_time = date
        else:
            current_group.append(text)
            end_time = date

if current_group:
    groupes.append([start_time, end_time, "\n".join(current_group)])



# === Couleurs pour les cartes ===
couleurs = [
    "#009fe3",  # Bleu RTE principal
    "#43b7ea",  # Bleu lumineux
    "#85cff1",  # Bleu doux et clair
    "#c7e7f8",  # Bleu très pâle
]
st.title("Synthèses Chronologique")

if st.button("Retour à l'accueil"):
    st.switch_page("app_streamlit.py")


groupes = list(reversed(groupes))
for i, groupe in enumerate(groupes):
    couleur = couleurs[i % len(couleurs)]
    
    
    date_start = groupe[0].strftime("%Y-%m-%d %H:%M:%S")
    date_end = groupe[1].strftime("%Y-%m-%d %H:%M:%S")
    contenu = groupe[2]  
    
    contenu = clean_added_text(contenu)
    synthese = get_synthese(contenu)


      

    synthese_html = markdown2.markdown(synthese)
    st.markdown(
        f"""
        <div style="
            background-color:{couleur};
            border-radius:15px;
            padding:15px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            margin-bottom:20px;
            font-family: Arial;
        ">
            <h4>Synthèse {i+1} : {date_start} → {date_end}</h4>
            {synthese_html}
        """,
        unsafe_allow_html=True
    )


