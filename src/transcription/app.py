# streamlit run src/transcription/app.py

import streamlit as st
import json

def toggle_pause():
    with open("src/transcription/config.json", "r") as f:
        cfg = json.load(f)
    cfg["pause"] = not cfg["pause"]  # On inverse la valeur
    with open("src/transcription/config.json", "w") as f:
        json.dump(cfg, f)

st.button("Changer pause", on_click=toggle_pause)
