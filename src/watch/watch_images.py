import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing.add_data2db import add_data2db

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Lancer avec : python src/watch/watch_images.py

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

class NewImageHandler(FileSystemEventHandler):
    def __init__(self, debounce_sec=1.0):
        super().__init__()
        self.debounce = debounce_sec
        self._last_seen = {}

    def on_created(self, event):
        if event.is_directory:
            return
        path = event.src_path
        ext = os.path.splitext(path)[1].lower()
        if ext not in IMAGE_EXT:
            return

        # debounce : attendre un peu que le fichier soit entièrement écrit
        now = time.time()
        last = self._last_seen.get(path, 0)
        if now - last < self.debounce:
            return
        self._last_seen[path] = now

        # petite attente pour fiabilité écriture
        time.sleep(0.5)
        print(f"[watcher] Nouvelle image détectée: {path}")
        try:
            add_data2db(path)
        except Exception as e:
            print(f"[watcher] Erreur add_data2db: {e}")

def watch_dir(directory: str):
    if not os.path.isdir(directory):
        raise FileNotFoundError(directory)
    event_handler = NewImageHandler()
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    print(f"[watcher] Surveillance du dossier: {directory}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    watch_dir("/Users/tomamirault/Documents/projects/p1-dty-rte/detection-notes/tmp/paper")  # adapte si besoin
