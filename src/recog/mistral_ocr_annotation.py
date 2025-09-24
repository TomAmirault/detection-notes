import os
from pydantic import BaseModel, Field
from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, ResponseFormat
from mistralai.extra import response_format_from_pydantic_model
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)


### Image Annotation with custom response format using Pydantic models ###

# Image Annotation response format
class Document(BaseModel):
  image_type: str = Field(..., description="The type of the image in french.")
  note_id: str = Field(..., description="The number of the note in french. Often written on the top right of the note.")
  short_description: str = Field(..., description="A description in french describing the image.")
  summary: str = Field(..., description="Summarize the image in french.")
  rappels: str = Field(..., description="Rappels importants à retenir de l'image en francais. En explicitant bien toutes les infos qu'on a sur chaque rappel.")
  incidents: str = Field(..., description="Incidents ou anomalies notés par l'utilisateur sur la feuille de l'image en francais.")
  call_recap: str = Field(..., description="Recapitulatif de l'appel en francais.")
  additional_info: str = Field(..., description="Any additional information that could be useful in french.")

def image_annotation(base64_image: str) -> ResponseFormat:
    response = client.ocr.process(
      model="mistral-ocr-latest",
      document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_image}"
        },
      document_annotation_format=response_format_from_pydantic_model(Document),
      include_image_base64=True
    )
    extracted_data_raw = response.document_annotation
    extracted_data = json.loads(extracted_data_raw)  # convertit la chaîne JSON en dict

    return extracted_data