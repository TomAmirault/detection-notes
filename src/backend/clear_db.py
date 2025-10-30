import sys, os
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
from src.backend.db import clear_notes_meta

if __name__ == "__main__":
    clear_notes_meta()
