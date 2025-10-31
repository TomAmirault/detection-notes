import sys
import os
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
from src.audio.dictionary.prompts import MISTRAL_CLEAN_PROMPT
from src.audio.dictionary.vocabulary import KNOWN_ABBREVIATIONS, KNOWN_CITY, KNOWN_NAMES

from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

def clean_audio_transcription(texte: str) -> str:
    """
    Cleans raw text from an audio transcription.
    - Corrects obvious errors
    - Restores minimal punctuation
    - Does not change the meaning or word order
    - Does not rephrase
    
    Args:
        texte (str): Raw transcription text to clean.

    Returns:
        str: Cleaned transcription with corrected errors and minimal punctuation.
    """
    prompt = MISTRAL_CLEAN_PROMPT.format(
        texte=texte,
        KNOWN_ABBREVIATIONS=KNOWN_ABBREVIATIONS,
        KNOWN_NAMES=KNOWN_NAMES,
        KNOWN_CITY=KNOWN_CITY
    )

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    clean_text = response.choices[0].message.content.strip()
    return clean_text






