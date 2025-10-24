import logging
from logging.handlers import TimedRotatingFileHandler
import os, sys
REPO_PATH = os.path.abspath(os.path.dirname(__file__))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
from pathlib import Path

def setup_logger(name: str, log_file: str = "logs/detection_notes.log", level=logging.DEBUG):
    """Crée et retourne un logger configuré"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        # File handler
        fh = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7)
        fh.setLevel(level)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Ajouter handlers
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

def save_fig_with_limit(file_name: str, fig, max_files: int = 20):
    """
    Sauvegarde un fichier et supprime le plus ancien si le nombre de fichiers dépasse max_files.
    """
    file_path = os.path.join(REPO_PATH, file_name)
    file_path = Path(file_path)
    folder = file_path.parent
    folder.mkdir(parents=True, exist_ok=True)

    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    # !! rajouter un logger

    # Vérifier le nombre de fichiers existants
    files = sorted(folder.glob('*'), key=lambda f: f.stat().st_mtime)  # tri par date de modification

    # Supprimer les fichiers les plus anciens si on dépasse max_files
    while len(files) > max_files:
        oldest = files.pop(0)
        print(f"Suppression du fichier ancien : {oldest.name}")
        oldest.unlink()