# D-ScrAibe — Intelligent Note Digitization for Grid Dispatchers

> **Project D-ScrAibe**: Real-time multimodal note-taking system for RTE electrical grid dispatchers, combining computer vision (YOLO), handwriting recognition (Mistral OCR, Teklia HTR), voice activity detection (pyannote.audio), speech-to-text (Whisper), and LLM-powered entity extraction for automated knowledge management.

**Organization**: RTE (Réseau de Transport d'Électricité) — Paris Digital Lab  
**Academic Partner**: CentraleSupélec, Illuin Technology  
**Project Period**: Fall 2025

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Prerequisites & Setup](#prerequisites--setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Team](#team)
- [License](#license)

---

## Overview

# D-ScrAibe – Digitization System for RTE Dispatcher Notes

**D-ScrAibe** is an automated system designed to digitize, transcribe, and analyze handwritten notes from RTE (Réseau de Transport d'Électricité) electrical grid dispatchers. The system processes both paper-based notes and audio recordings to extract structured information about electrical grid events, enable searchability, and support operational decision-making.

The system is designed for 24/7 operations in dispatching control rooms, where rapid information capture and retrieval are critical for grid stability.

---

## Key Features

### 🖼️ **Multi-modal Input**
- **Paper detection**: Real-time webcam monitoring with YOLO-based sheet detection and deblurring
- **Audio recording**: Continuous recording with automatic pause detection and segmentation (VAD)
- **Multiple camera sources**: Built-in webcam, USB cameras, or phone via Continuity Camera/DroidCam

### ✍️ **Advanced OCR/HTR**
- **Mistral OCR Pipeline**: `mistral-ocr-latest` + `mistral-large-latest` for normalization
- **Teklia HTR Pipeline**: Industry-grade handwriting recognition with confidence-based filtering
- **LLM Post-processing**: Custom prompts for French dispatcher terminology (63 official abbreviations)
- **Quality Filters**: HTR bug detection (repetitive patterns, dominant words, low diversity)

### 🔍 **Intelligent Text Processing**
- **Diff Computation**: Line-level change detection with split/merge handling (1→2, 2→1 transformations)
- **Deduplication**: 
  - Visual similarity via perceptual hashing (last 20 images)
  - Text similarity with configurable thresholds (0.1-0.9)
  - Anti-repetition filters for OCR instability
- **Text Canonicalization**: Accent removal, case normalization, tokenization for robust comparison

### 🏷️ **Entity Extraction (NER)**
9 domain-specific entity types via LLM prompts:
- **GEO**: Locations, substations, cities, lines (e.g., "Paris-Lyon 400kV")
- **ACTOR**: Operators, teams, third-parties (e.g., "Chargé de conduite", "GEH")
- **DATETIME**: Dates, times, relative expressions (e.g., "14h", "demain matin")
- **EVENT**: Network events, external incidents (e.g., "RADA", "TST")
- **INFRASTRUCTURE**: Transformers, lines, circuit breakers (e.g., "TR", "DIFFB")
- **OPERATING_CONTEXT**: Operational modes (e.g., "RACR", "RSD", "SUAV")
- **PHONE_NUMBER**: French phone formats (e.g., "06 12 34 56 78", "+33 1 23 45 67 89")
- **ELECTRICAL_VALUE**: Voltages, power (e.g., "225kV", "1500MW")
- **ABBREVIATION_UNKNOWN**: Unrecognized uppercase abbreviations for dictionary expansion

### 📊 **Event Grouping**
- Automatic clustering of notes by entity similarity (fuzzy matching on locations, actors, times)
- Event timeline visualization with expandable note threads
- Supports both image-based notes (TEXT-N) and audio notes (AUD-N)

### 🎨 **User Interface**
- **Streamlit Timeline**: Card-based display with auto-refresh, search, and filtering
- **Note Thread View**: Expandable history showing all versions of a note
- **Event Summary**: Automatic summarization of grouped notes (optional)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER INPUT SOURCES                                  │
├─────────────────────┬──────────────────────┬─────────────────────────────────┤
│   📷 Webcam         │   🎙️ Microphone      │   📱 Phone Camera              │
│   (Paper Detection) │   (Audio Recording)  │   (Continuity/DroidCam)        │
└──────────┬──────────┴──────────┬───────────┴──────────┬──────────────────────┘
           │                     │                      │
           v                     v                      v
    ┌────────────┐        ┌────────────┐        ┌────────────┐
    │    YOLO    │        │    VAD     │        │   Manual   │
    │  Detection │        │ Segmentat. │        │   Upload   │
    └─────┬──────┘        └─────┬──────┘        └─────┬──────┘
          │                     │                      │
          v                     v                      v
    ┌────────────────────────────────────────────────────────┐
    │         📝 PROCESSING PIPELINE (add_data2db)           │
    ├────────────────────────────────────────────────────────┤
    │  1. OCR/HTR (Mistral/Teklia) + LLM Normalization       │
    │  2. Quality Filters (HTR bugs, noise, empty text)      │
    │  3. Similarity Search (find existing note threads)     │
    │  4. Diff Computation (line-level changes)              │
    │  5. Entity Extraction (9 types via LLM)                │
    │  6. Event Grouping (entity similarity matching)        │
    └────────────────┬───────────────────────────────────────┘
                     │
                     v
         ┌───────────────────────┐
         │   💾 SQLite Database   │
         │  ┌─────────────────┐  │
         │  │  notes_meta     │  │
         │  │  • transcriptions│  │
         │  │  • entities (9) │  │
         │  │  • event_id     │  │
         │  │  • timestamps   │  │
         │  │  • images paths │  │
         │  └─────────────────┘  │
         └───────────┬───────────┘
                     │
                     v
         ┌───────────────────────┐
         │   🖥️ Streamlit UI      │
         │  • Event Timeline      │
         │  • Note Threads        │
         │  • Search & Filters    │
         └───────────────────────┘
```

### Core Workflow

1. **Capture**: Camera detects paper → YOLO extracts note → saves to `tmp/paper/`
2. **OCR/HTR**: Image → Mistral/Teklia → Raw text → LLM normalization → Clean text
3. **Similarity Check**: Compare against existing notes (visual hash + text similarity)
4. **Diff Computation**: If same note detected → compute line-level diff (only new content)
5. **Entity Extraction**: Extract 9 entity types from new content via LLM prompt
6. **Event Grouping**: Find/create event based on entity similarity with existing notes
7. **Database Insert**: Store in `notes_meta` with entities, event_id, and diff
8. **UI Update**: Timeline refreshes automatically to display new note/event

**Parallel Audio Path**: VAD segments audio → Whisper transcribes → LLM cleans → same steps 4-8

---

## Project Structure

```
detection-notes/
├── docs/                                 # 📚 Project Documentation
│   ├── Documentation D-ScrAibe.pdf       # Comprehensive technical documentation
│   └── presentations/                    # Project slides and presentations
│
├── src/
│   ├── run_app.py                        # 🚀 Main orchestrator (launches all services)
│   │
│   ├── processing/                       # 📝 OCR/HTR & Database Ingestion
│   │   ├── add_data2db.py                # Core pipeline: OCR → diff → entities → DB
│   │   ├── mistral_ocr_llm.py            # Mistral OCR + LLM normalization
│   │   ├── teklia_ocr_llm.py             # Teklia HTR + LLM normalization
│   │   └── prompts.py                    # LLM prompt templates (OCR normalization)
│   │
│   ├── backend/                          # 💾 Database Operations
│   │   ├── db.py                         # SQLite CRUD, similarity search, event grouping
│   │   └── clear_db.py                   # Database reset utility
│   │
│   ├── audio/                            # 🎙️ Audio Processing Pipeline
│   │   ├── pipeline_watcher.py           # File watcher orchestrator
│   │   ├── audio_recorder.py             # Real-time recording with pause detection
│   │   ├── vad_detector.py               # Voice Activity Detection (pyannote)
│   │   ├── whisper_transcriber.py        # Whisper ASR transcription
│   │   ├── audio_cleaner.py              # LLM-based transcript cleaning
│   │   └── dictionary/
│   │       ├── vocabulary.py             # Dispatcher vocabulary lists
│   │       └── prompts.py                # Audio cleaning prompts
│   │
│   ├── paper_detection/                  # 📷 Computer Vision
│   │   ├── edges_based/                  # Edge-based paper detection
│   │   │   └── video_capture.py          # Main capture script
│   │   └── yolo/                         # YOLO-based detection
│   │       ├── yolo_tracker_photos.py    # YOLO tracking & deblurring
│   │       └── model/
│   │           └── best-detect.pt        # YOLO weights (included in repo)
│   │
│   ├── ner/                              # 🏷️ Named Entity Recognition
│   │   ├── llm_extraction.py             # Entity extraction via Mistral
│   │   ├── compare_entities.py           # Entity similarity for event grouping
│   │   ├── ner_prompt_template.py        # Entity extraction prompts
│   │   └── archives/spacy_model.py       # Legacy spaCy NER (replaced by LLM)
│   │
│   ├── utils/                            # 🛠️ Text & Image Utilities
│   │   ├── text_utils.py                 # Diff computation, HTR bug detection
│   │   └── image_utils.py                # Image encoding, perceptual hashing
│   │
│   ├── frontend/                         # 🖥️ User Interface
│   │   ├── app_streamlit.py              # Timeline UI with event cards
│   │   └── pages/
│   │       ├── app_timeline_cards.py     # Timeline view implementation
│   │       └── config.py                 # UI configuration
│   │
│   ├── summary/                          # 📊 Event Summarization (Optional)
│   │   ├── note_summarizer.py            # LLM-based event summaries
│   │   └── prompt.py                     # Summarization prompts
│   │
│   ├── image_similarity/                 # 🔍 Visual Deduplication
│   │   ├── image_comparison.py           # Perceptual hashing (pHash)
│   │   └── resize_minkowski_interpolation.py  # Image preprocessing
│   │
│   └── raspberry/                        # 🍓 Raspberry Pi Integration
│       └── launch_rasp.py                # Pi camera control
│
├── data/                                 # 📁 Data Storage
│   ├── db/notes.sqlite                   # SQLite database
│   ├── images/                           # Captured note images
│   │   ├── raw/                          # Original captures
│   │   └── text-zones/                   # Extracted text regions
│   ├── lines/                            # Line-level segmentation
│   └── abréviations.xlsx                 # Official RTE abbreviations (63 terms)
│
├── logs/                                 # 📋 Application Logs
│   ├── detection_notes.log               # Main log file (rotated daily)
│   ├── image-comparison/                 # Visual comparison debug images
│   └── color-criteria/                   # Color filter debug images
│
├── tests/                                # 🧪 Unit & Integration Tests
│
├── pyproject.toml                        # 📦 Python 3.12 dependencies (uv/pip)
├── uv.lock                               # 🔒 Locked dependency versions
├── logger_config.py                      # 📝 Centralized logging config
├── .env                                  # 🔐 API keys (not in repo)
├── .gitignore                            # 🚫 Excluded files
└── README.md                             # 📖 This file
```

---

## Prerequisites & Setup

### System Requirements

- **Python 3.12** (required, NOT 3.13+)
- **FFmpeg 6.x** (critical for audio processing)
- **Git**
- **Camera** (built-in, USB, or smartphone)
- **16GB+ RAM** (for YOLO + Whisper models)

### API Keys Required

1. **Mistral API** → [console.mistral.ai](https://console.mistral.ai)
   - Used for: OCR normalization, entity extraction, audio cleaning
   
2. **Teklia Ocelus API** (optional) → [ocelus.teklia.com](https://atr.ocelus.teklia.com)
   - Used for: Alternative HTR pipeline (higher quality, slower)
   
3. **Teklia API Key** (for Teklia HTR alternative)
   - **Commercial Access**: Contact Teklia directly to purchase API credits
   - **Free Trial**: Request test account with limited credits via email to Teklia
   - Once obtained, add to `.env`: `TEKLIA_API_KEY=your-key-here`

4. **Hugging Face Token** (for pyannote.audio)
   - Go to [huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - Accept user conditions
   - Create **READ token** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Connect to VS Code: `huggingface-cli login`

### Installation

#### 1. Install FFmpeg 6.x (CRITICAL — Do this FIRST)

```bash
# macOS
brew install ffmpeg@6
brew link --overwrite ffmpeg@6

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Verify
ffmpeg -version  # Must show version 6.x.x
```

⚠️ **WARNING**: Installing Python dependencies before FFmpeg may cause audio libraries to fail.

#### 2. Clone Repository

```bash
git clone https://gitlab.paris-digital-lab.com/rte/rte-f2025-p1/detection-notes.git
cd detection-notes
```

#### 3. Create Virtual Environment

```bash
python3.12 -m venv .venv_py312
source .venv_py312/bin/activate  # Linux/macOS
# .venv_py312\Scripts\activate   # Windows
```

#### 4. Install Python Dependencies

```bash
# Option A: Using uv (recommended, faster)
pip install uv
uv pip install -e .

# Option B: Using standard pip
pip install -e .
```

#### 5. Download spaCy French Model

```bash
python -m spacy download fr_core_news_lg
```

#### 6. Verify YOLO Weights

```bash
ls src/paper_detection/yolo/model/best-detect.pt
# Expected: src/paper_detection/yolo/model/best-detect.pt
```

The YOLO model weights are included in the Git repository.

#### 7. Configure Environment Variables

Create `.env` file in project root:

```bash
# Required: Mistral API for OCR/NER/Audio
MISTRAL_API_KEY=your_mistral_api_key_here

# Optional: Teklia HTR (alternative to Mistral OCR)
TEKLIA_API_KEY=your_teklia_api_key_here

# Optional: Custom database path
RTE_DB_PATH=/custom/path/to/notes.sqlite
```

#### 7. Initialize Database

```bash
python -c "from src.backend.db import ensure_db; ensure_db()"
```

#### 8. Login to Hugging Face (for pyannote.audio)

```bash
huggingface-cli login
# Paste your READ token when prompted
```

---

## Configuration

### Service Enable/Disable

Edit `src/run_app.py`:

```python
# Service flags
ENABLE_FRONTEND = True              # Streamlit UI
ENABLE_PAPER_DETECTION = True       # Webcam capture + YOLO
ENABLE_YOLO_TRACKER = False         # Advanced YOLO tracking (optional)
ENABLE_RASPBERRY_PI = False         # Raspberry Pi camera (requires Pi hardware)
ENABLE_AUDIO_PIPELINE = True        # Audio recording + transcription
CLEAR_DATABASE_ON_STARTUP = True    # Fresh start on each launch
```

### OCR/HTR Engine Selection

Edit `src/processing/add_data2db.py` line 24:

```python
# Option 1: Mistral OCR (faster, API-based)
from src.processing.mistral_ocr_llm import image_transcription

# Option 2: Teklia HTR (higher quality, slower, requires API key)
# from src.processing.teklia_ocr_llm import image_transcription
```

### Deduplication Thresholds

Edit `src/processing/add_data2db.py` lines 52-72:

```python
# Text similarity for same note detection (0.0-1.0)
NOTE_SIMILARITY_THRESHOLD = 0.7

# Anti-repetition filters
ANTI_REPETITION_LENGTH_DIFF = 35        # Max char difference
ANTI_REPETITION_SIMILARITY_MIN = 0.5    # Min similarity to block duplicate

# Diff computation
DIFF_MINOR_CHANGE_THRESHOLD = 0.90      # Ignore changes >90% similar
```

**When to adjust:**
- Too many duplicates → Lower `NOTE_SIMILARITY_THRESHOLD` to 0.6
- Notes incorrectly merged → Raise `NOTE_SIMILARITY_THRESHOLD` to 0.8
- Legitimate notes blocked → Raise `ANTI_REPETITION_SIMILARITY_MIN` to 0.7

### OCR Pipeline Configuration

**Mistral OCR** (`src/processing/mistral_ocr_llm.py`):
```python
OCR_MODEL = "mistral-ocr-latest"
NORMALIZATION_LLM_MODEL = "mistral-small-latest"  # or mistral-large-latest
NORMALIZATION_TEMPERATURE = 0.0  # Deterministic output
```

**Teklia HTR** (`src/processing/teklia_ocr_llm.py`):
```python
OCR_CONFIDENCE_THRESHOLD = 0.5  # Filter lines below 50% confidence
NORMALIZATION_LLM_MODEL = "mistral-small-latest"
```

### Audio Processing

Edit `src/audio/vad_detector.py`:

```python
DEFAULT_MIN_DURATION_ON = 3.0   # Min speech segment (seconds)
DEFAULT_MIN_DURATION_OFF = 2.0  # Min pause to split (seconds)
```

---

## Usage

### Launch Full System

```bash
python src/run_app.py
```

This starts:
- ✅ **Streamlit UI** at [http://localhost:8501](http://localhost:8501)
- ✅ **Paper detection** (webcam capture loop)
- ✅ **Audio pipeline** (recording + transcription)

Press **Ctrl+C** to stop all services.

### Run Components Individually

#### Frontend Only
```bash
streamlit run src/frontend/app_streamlit.py
```

#### Paper Detection Only
```bash
# Edge-based detection
python src/paper_detection/edges_based/video_capture.py

# YOLO-based detection
python src/paper_detection/yolo/yolo_tracker_photos.py
```

#### Audio Recording Only
```bash
python src/audio/audio_recorder.py
# Press SPACE to pause/resume, CTRL+C to stop
```

#### Process Single Image
```python
from src.processing.add_data2db import add_data2db
from pathlib import Path

image_path = Path("data/images/raw/note_001.jpg")
meta_id = add_data2db(image_path)
print(f"Inserted with meta_id: {meta_id}")
```

#### Process Single Audio File
```python
from src.processing.add_data2db import add_audio2db

audio_path = "src/audio/tmp/segment_001.wav"
transcription_brute = "appel jean quatorze heures"  # Raw Whisper output
transcription_clean = "Appel Jean 14h"              # After LLM cleaning

meta_id = add_audio2db(audio_path, transcription_brute, transcription_clean)
```

---

## Development

### Code Quality Standards

- **Python 3.12** syntax (`list[str]`, `X | Y` unions)
- **Black** formatting (100-char lines)
- **Type hints** everywhere (`pathlib.Path` for paths)
- **Google-style docstrings** for all public functions
- **Centralized logging** via `logger_config.py` (no `print()`)
- **English** for all code, comments, docstrings

### Project Conventions

```python
# Good: Type hints, Path objects, logger
from pathlib import Path
from logger_config import setup_logger

logger = setup_logger(__name__)

def process_image(image_path: Path) -> tuple[str, float]:
    """Process image with OCR and return cleaned text with confidence."""
    logger.info(f"Processing image: {image_path}")
    # ...
    return cleaned_text, confidence

# Bad: No types, string paths, print
def process_image(image_path):
    print(f"Processing {image_path}")
    # ...
    return cleaned_text, confidence
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific module
pytest tests/test_text_utils.py

# With coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Linting & Formatting

```bash
# Format code
black src/ --line-length 100

# Lint
ruff check src/

# Type checking
mypy src/
```

### Database Management

```bash
# Clear database
python src/backend/clear_db.py

# Inspect with SQLite
sqlite3 data/db/notes.sqlite
```

```sql
-- Useful queries
SELECT COUNT(*) FROM notes_meta;
SELECT note_id, COUNT(*) as versions FROM notes_meta GROUP BY note_id;
SELECT DISTINCT evenement_id FROM notes_meta WHERE evenement_id IS NOT NULL;
```

```python
# Python API examples
from src.backend.db import list_notes_by_evenement_id, delete_thread_by_note_id

# Get all notes for an event
notes = list_notes_by_evenement_id("EVT-42")
for note in notes:
    print(f"{note['note_id']}: {note['transcription_clean']}")

# Delete entire note thread
deleted_count = delete_thread_by_note_id("TEXT-5")
print(f"Deleted {deleted_count} entries")
```

### Adding New Abbreviations

1. Edit `data/abréviations.xlsx`
2. Update `src/ner/ner_prompt_template.py` → `ABBREVIATIONS_DICT`
3. Restart system to reload dictionary

### Extending Entity Types

1. Add new entity type to `src/ner/ner_prompt_template.py` → `ENTITY_DEFINITIONS`
2. Update database schema in `src/backend/db.py` → `ensure_db()`:
   ```sql
   ALTER TABLE notes_meta ADD COLUMN entite_NEW_TYPE TEXT;
   ```
3. Update extraction logic in `src/ner/llm_extraction.py`
4. Update UI display in `src/frontend/app_streamlit.py`

---

## Troubleshooting

### Audio Issues

**Problem**: Audio recording fails or produces no segments

**Solutions**:
1. Verify FFmpeg 6.x: `ffmpeg -version`
2. Check microphone permissions: System Settings → Privacy & Security → Microphone
3. Test microphone: `python -c "import sounddevice; print(sounddevice.query_devices())"`
4. Ensure virtual environment is activated

**Problem**: pyannote.audio throws authentication error

**Solutions**:
1. Accept conditions at [huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
2. Login with READ token: `huggingface-cli login`
3. Verify token: `huggingface-cli whoami`

### OCR Issues

**Problem**: OCR produces gibberish or repetitive text

**Solutions**:
1. Check `MISTRAL_API_KEY` in `.env`
2. Verify image quality (blur, lighting, resolution)
3. Switch to Teklia HTR: uncomment line 24 in `add_data2db.py`
4. Adjust `OCR_CONFIDENCE_THRESHOLD` in `teklia_ocr_llm.py` (try 0.6-0.7)

**Problem**: HTR marked as "buggy" and skipped

**Explanation**: System detected repetitive patterns (e.g., same word 10+ times)

**Solutions**:
1. Retake photo with better lighting/focus
2. Lower thresholds in `src/utils/text_utils.py`:
   ```python
   DOMINANT_WORD_THRESHOLD = 0.70  # default: 0.60
   MAX_CONSECUTIVE_REPETITIONS = 7  # default: 5
   ```

### Deduplication Issues

**Problem**: Too many duplicate notes created

**Solutions**:
1. Lower `NOTE_SIMILARITY_THRESHOLD`: `0.7 → 0.6`
2. Lower `ANTI_REPETITION_SIMILARITY_MIN`: `0.5 → 0.3`
3. Check image similarity logs in `logs/image-comparison/`

**Problem**: Different notes incorrectly merged

**Solutions**:
1. Raise `NOTE_SIMILARITY_THRESHOLD`: `0.7 → 0.8`
2. Check diff computation logs for incorrect matches

### Entity Extraction Issues

**Problem**: Entities not extracted or incorrect

**Solutions**:
1. Verify `MISTRAL_API_KEY` validity
2. Check prompt in `src/ner/ner_prompt_template.py`
3. Test extraction manually:
   ```python
   from src.ner.llm_extraction import extract_entities
   entities = extract_entities("Appel Jean 14h pour TR Paris 225kV")
   print(entities)
   ```

**Problem**: Notes not grouping into events

**Solutions**:
1. Check entity similarity rules in `src/ner/compare_entities.py`
2. Lower similarity thresholds for entity matching
3. Verify entities are being extracted (check database: `SELECT entite_ACTOR FROM notes_meta`)

### Performance Issues

**Problem**: System slow or unresponsive

**Solutions**:
1. Reduce `IMAGE_COMPARISON_LIMIT` in `src/backend/db.py` (default: 20 → 10)
2. Use Mistral Small instead of Large for normalization
3. Disable YOLO tracker: `ENABLE_YOLO_TRACKER = False`
4. Clear old logs: `rm logs/detection_notes.log.*`

---

## Team

### Development Team

**Project Lead:**
- Tom Amirault — [tom.amirault@student-cs.fr](mailto:tom.amirault@student-cs.fr)

**Core Developers:**
- Mohammed Lbakali — [mohammed.lbakali@student-cs.fr](mailto:mohammed.lbakali@student-cs.fr)
- Alexandre Corrard — [alexandre.corrard@student-cs.fr](mailto:alexandre.corrard@student-cs.fr)
- Noah Parisse — [noah.parisse@student-cs.fr](mailto:noah.parisse@student-cs.fr)

**Technical Advisors:**
- Nahel Zidi — [nahel.zidi@illuin.tech](mailto:nahel.zidi@illuin.tech) (Illuin Technology)
- Philippe Pelissier — [philippe.pelissier@illuin.tech](mailto:philippe.pelissier@illuin.tech) (Illuin Technology)

### Organization

**RTE — Réseau de Transport d'Électricité**  
Paris Digital Lab  
[www.rte-france.com](https://www.rte-france.com)

**Academic Partner**  
CentraleSupélec — Université Paris-Saclay

**Industry Partner**  
Illuin Technology — AI/ML Consulting  
[www.illuin.tech](https://www.illuin.tech)

---

## License

**Proprietary Software** — All Rights Reserved

This project is confidential and proprietary to RTE (Réseau de Transport d'Électricité).  
Unauthorized copying, distribution, or use is strictly prohibited.

**Repository**: Private GitLab  
[https://gitlab.paris-digital-lab.com/rte/rte-f2025-p1/detection-notes](https://gitlab.paris-digital-lab.com/rte/rte-f2025-p1/detection-notes)

License terms are to be determined in coordination with RTE's legal department upon project handoff.

**Additional Documentation**:
- See `docs/Documentation D-ScrAibe.pdf` for comprehensive technical documentation
- Presentation slides available in `docs/presentations/`

---

## Acknowledgments

- **Mistral AI** for OCR and LLM APIs
- **Teklia** for HTR Ocelus platform
- **Hugging Face** for pyannote.audio models
- **OpenAI** for Whisper speech recognition
- **Ultralytics** for YOLO object detection
- **RTE Dispatchers** for domain expertise and feedback

---

## Citation

If referencing this work in publications, please cite:

```bibtex
@software{dscraibe_rte_2025,
  title={D-ScrAibe: Intelligent Note Digitization System for Electrical Grid Dispatchers},
  author={Amirault, Tom and Lbakali, Mohammed and Corrard, Alexandre and Parisse, Noah},
  year={2025},
  organization={RTE, CentraleSupélec, Illuin Technology},
  url={https://gitlab.paris-digital-lab.com/rte/rte-f2025-p1/detection-notes}
}
```

---

## Quick Start Checklist

- [ ] Install FFmpeg 6.x
- [ ] Clone repository
- [ ] Create Python 3.12 venv
- [ ] Install dependencies (`uv pip install -e .`)
- [ ] Download spaCy model (`python -m spacy download fr_core_news_lg`)
- [ ] Create `.env` with `MISTRAL_API_KEY`
- [ ] Login to Hugging Face (`huggingface-cli login`)
- [ ] Initialize database (`python -c "from src.backend.db import ensure_db; ensure_db()"`)
- [ ] Run system (`python src/run_app.py`)
- [ ] Open UI at [http://localhost:8501](http://localhost:8501)

**Questions?** Contact [tom.amirault@student-cs.fr](mailto:tom.amirault@student-cs.fr)
