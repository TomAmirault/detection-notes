# streamlit run src/transcription/app.py

import streamlit as st
import subprocess
import time

SCRIPT_PATH = "src/transcription/audio_watcher.py"

if "process" not in st.session_state:
    st.session_state.process = None
    st.session_state.log_placeholder = st.empty()

if st.button("Start / Stop"):
    if st.session_state.process is None or st.session_state.process.poll() is not None:
        # Démarrage du script
        st.session_state.process = subprocess.Popen(
            ["python3", SCRIPT_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        st.success("Script démarré")
        
        # Boucle pour afficher les logs en temps réel
        while True:
            line = st.session_state.process.stdout.readline()
            if line == "" and st.session_state.process.poll() is not None:
                break
            if line:
                st.session_state.log_placeholder.text(line.strip())

    else:
        # Arrêt du script
        st.session_state.process.terminate()
        st.session_state.process = None
        st.warning("Script arrêté")


