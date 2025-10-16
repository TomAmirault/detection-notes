import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

def image_transcription(base64_image: str) -> str:
    extracted_text = ""
    response = client.ocr.process(
      model="mistral-ocr-latest",
      document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_image}"
        },
      include_image_base64=True
    )
    extracted_text += response.pages[0].markdown + "\n\n"

    return extracted_text