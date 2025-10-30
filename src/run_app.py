"""
Main application launcher - orchestrates all system services.

This module starts and manages all components of the note detection system:
- Frontend UI (Streamlit)
- Paper detection (camera/YOLO/Raspberry Pi)
- Audio transcription pipeline
- Database initialization

All services run as background processes with synchronized lifecycle management.
"""
import sys
import os
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

import subprocess
import sys
import time

from logger_config import setup_logger

logger = setup_logger(__name__)

# =============================================================================
# Configuration Constants

# Service startup delay (seconds)
# Time to wait between launching each service to ensure proper initialization
SERVICE_STARTUP_DELAY: float = 0.5

# Enable/disable individual services
ENABLE_FRONTEND: bool = True
ENABLE_EDGES_BASED_DETECTION: bool = True
ENABLE_YOLO_TRACKER: bool = False  # Advanced paper detection (optional)
ENABLE_RASPBERRY_PI: bool = False  # Physical camera integration (optional)
ENABLE_AUDIO_PIPELINE: bool = True

# Database cleanup on startup
CLEAR_DATABASE_ON_STARTUP: bool = True

# =============================================================================


def main():
    """
    Launch all application services in coordinated fashion.

    Workflow:
    1. Optionally clear database (fresh start)
    2. Launch frontend UI (Streamlit)
    3. Launch paper detection service(s)
    4. Launch audio transcription pipeline
    5. Monitor all processes until user interruption

    All subprocesses inherit stdout/stderr for unified logging.
    On shutdown (KeyboardInterrupt or error), all processes are gracefully terminated.
    """
    # Determine base directory (src folder containing this script)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

    # Service paths
    streamlit_path = os.path.join(BASE_DIR, "frontend/app_streamlit.py")
    edges_based_path = os.path.join(BASE_DIR, "paper_detection/edges_based/video_capture.py")
    yolo_path = os.path.join(BASE_DIR, "paper_detection/yolo/yolo_tracker_photos.py")
    raspberry_path = os.path.join(BASE_DIR, "raspberry/launch_rasp.py")
    audio_path = os.path.join(BASE_DIR, "audio/pipeline_watcher.py")
    clear_db_path = os.path.join(BASE_DIR, "backend/clear_db.py")

    processes = []

    try:
        # Step 1: Database initialization
        if CLEAR_DATABASE_ON_STARTUP:
            logger.info("Clearing database for fresh start")
            try:
                subprocess.run([sys.executable, clear_db_path], check=True)
                logger.info("Database cleared successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clear database: {e}")
                raise

        # Step 2: Launch frontend UI
        if ENABLE_FRONTEND:
            logger.info("Starting frontend UI (Streamlit)")
            processes.append(
                subprocess.Popen(
                    ["streamlit", "run", streamlit_path],
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                )
            )
            logger.info("Frontend UI launched successfully")
            time.sleep(SERVICE_STARTUP_DELAY)

        # Step 3: Launch paper detection services
        if ENABLE_EDGES_BASED_DETECTION:
            logger.info("Starting paper detection system (edges-based)")
            processes.append(
                subprocess.Popen(
                    [sys.executable, edges_based_path],
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    cwd=PROJECT_ROOT,
                )
            )
            logger.info("Paper detection system launched successfully")
            time.sleep(SERVICE_STARTUP_DELAY)

        if ENABLE_YOLO_TRACKER:
            logger.info("Starting YOLO tracker for advanced paper detection")
            processes.append(
                subprocess.Popen(
                    [sys.executable, yolo_path],
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    cwd=PROJECT_ROOT,
                )
            )
            logger.info("YOLO tracker launched successfully")
            time.sleep(SERVICE_STARTUP_DELAY)

        if ENABLE_RASPBERRY_PI:
            logger.info("Starting Raspberry Pi camera integration")
            processes.append(
                subprocess.Popen(
                    [sys.executable, raspberry_path],
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    cwd=PROJECT_ROOT,
                )
            )
            logger.info("Raspberry Pi integration launched successfully")
            time.sleep(SERVICE_STARTUP_DELAY)

        # Step 4: Launch audio transcription pipeline
        if ENABLE_AUDIO_PIPELINE:
            logger.info("Starting audio transcription pipeline")
            processes.append(
                subprocess.Popen(
                    [sys.executable, audio_path], stdout=sys.stdout, stderr=sys.stderr
                )
            )
            logger.info("Audio pipeline launched successfully")
            time.sleep(SERVICE_STARTUP_DELAY)

        # Summary of launched services
        active_services = []
        if ENABLE_FRONTEND:
            active_services.append("Frontend UI")
        if ENABLE_EDGES_BASED_DETECTION:
            active_services.append("Edges-based Detection")
        if ENABLE_YOLO_TRACKER:
            active_services.append("YOLO Tracker")
        if ENABLE_RASPBERRY_PI:
            active_services.append("Raspberry Pi")
        if ENABLE_AUDIO_PIPELINE:
            active_services.append("Audio Pipeline")

        logger.info(
            f"All services launched successfully ({len(processes)} processes)"
        )
        logger.info(f"Active services: {', '.join(active_services)}")
        logger.info(
            "System is now digitizing handwritten notes and audio transcriptions"
        )
        logger.info("Press Ctrl+C to stop all services")

        # Step 5: Monitor processes indefinitely
        while True:
            # Check if any process has died unexpectedly
            for i, p in enumerate(processes):
                if p.poll() is not None:
                    logger.warning(
                        f"Process {i+1} ({active_services[i]}) terminated unexpectedly with code {p.returncode}"
                    )

            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user (Ctrl+C)")

    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}", exc_info=True)

    finally:
        # Graceful shutdown: terminate all child processes
        logger.info("Shutting down all services")
        for i, p in enumerate(processes):
            try:
                p.terminate()
                logger.debug(f"Terminated process {i+1}")
            except Exception as e:
                logger.warning(f"Failed to terminate process {i+1}: {e}")

        # Wait briefly for processes to exit cleanly
        time.sleep(1)

        # Force kill any remaining processes
        for i, p in enumerate(processes):
            if p.poll() is None:
                try:
                    p.kill()
                    logger.debug(f"Force killed process {i+1}")
                except Exception as e:
                    logger.warning(f"Failed to kill process {i+1}: {e}")

        logger.info("All services shut down successfully")


if __name__ == "__main__":
    main()