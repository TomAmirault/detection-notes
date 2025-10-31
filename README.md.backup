# detection-notes

## Description
Plateforme complète pour la détection, la capture, la reconnaissance et la gestion de notes manuscrites pour le dispatching électrique (RTE).
Le projet combine :
- Capture d’images (webcam, téléphone, AirDrop)
- Détection de feuilles (YOLO, segmentation)
- OCR/HTR (Tesseract, Mistral, TrOCR…)
- Nettoyage, correction et structuration du texte
- Stockage en base SQLite
- Visualisation et recherche via une interface Streamlit

## Architecture du projet
```text
detection-notes/
  src/
    add_data2db.py
    backend/
      clear_db.py
      db.py
      find_similar_note.py
      get_last_text_for_notes.py
      list_notes.py
    frontend/
      app_streamlit.py
    proc/
      detection_model/
      paper_detection/
      paper_detection_and_segmentation_Yolo/
    raspberry/
      capture_video.py
      pilote_camera_photo.py
      pilote_camera_video.py
    recog/
      archives/
      mistral_ocr_llm.py
    scripts/
      move_recent_photos.sh
      watcher.sh
    transcription/
      test_whisper.ipynb
      whisper_transcribe.py
    watch/
      watch_images.py
  requirements.txt
  uv.lock
  pyproject.toml
  README.md  
```

## Installation
  1. Installe Python 3.10+ et [uv](https://github.com/astral-sh/uv) (ou pip).
  2. Clone le repo :
      git clone <repo_url>
      cd detection-notes
  3. Crée et active un venv :
      python3 -m venv .venv
      source .venv/bin/activate
  4. Installe les dépendances :  
      uv pip install -r requirements.txt
      # ou
      pip install -r requirements.txt

## Utilisation
* Lancer le front Streamlit :
    streamlit run src/frontend/app_streamlit.py
* Lancer la capture vidéo
    python3 src/proc/paper_detection/yolo_tracker_with_deblurring.py
    python3 src/proc/paper_detection_and_segmentation_Yolo/yolo_tracker_segmented_with_deblurring.py

## Dépendances principales
streamlit
opencv-python
torch, ultralytics
transformers, gradio
sqlite3, pandas, numpy

## Conseils d’utilisation
Active toujours ton venv avant d’installer ou lancer les scripts Python.
Pour utiliser la caméra de ton téléphone : active le mode webcam (Continuity Camera sur Mac/iPhone, ou apps type DroidCam/EpocCam).
Pour ajouter une dépendance : uv add nom_du_package (ou pip install + ajout dans requirements.txt).
Pour rafraîchir le front automatiquement, modifie la variable REFRESH_SECONDS dans app_streamlit.py.

## Auteurs
Mohammed, Alexandre et Tom