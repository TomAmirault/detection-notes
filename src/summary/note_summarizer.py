import os
from mistralai import Mistral
from dotenv import load_dotenv

from src.summary.prompt import SUMMARY_PROMPT

# Load Mistral API key
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

def synthèse(texte: str) -> str:
    """
    Cleans raw text obtained from an audio transcription.
    - Corrects obvious mistakes
    - Restores minimal punctuation
    - Does not change the meaning or order
    - Does not rephrase
    """

    prompt = SUMMARY_PROMPT.format(
        texte=texte
    )


    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    clean_text = response.choices[0].message.content.strip()
    return clean_text


