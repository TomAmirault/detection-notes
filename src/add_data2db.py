### image_path -> load_image -> encode_image -> mistral_ocr_annotation

#image_path = "/Users/tomamirault/Documents/Projects/p1-dty-rte/detection-notes/data/images/raw/IMG_3268.JPG"

from capture.encode_image import encode_image
from recog.mistral_ocr_annotation import image_annotation
from backend.db import insert_note_meta, list_notes
import json



def add_data2db(image_path: str):
    
  image_encoded = encode_image(image_path) # base64 string

  extracted_data = image_annotation(image_encoded)



  new_id = insert_note_meta(extracted_data, img_path_proc=image_path)
  print("Inserted id:", new_id)


  rows = list_notes(10000)
  for r in rows:
      print(json.dumps(r, ensure_ascii=False, indent=2))  # Affichage de la db dans le terminal



"""
en dict
extracted_data = {
  "image_type": "Note manuscrite",
  "note_id": "2",
  "short_description": "Une note concernant un changement de relais et un appel avec Jean.",
  "summary": "La note décrit un changement de relais de 40KV à 20KV, un incident à Cergy 5 Maintenance, et un appel avec Jean concernant le SNFC.",
  "rappels": "1. Changement numéro de relais : M. Martin, 07 66 37 82 17. 2. Incident Cergy 5 Maintenance (jusqu'à demain). 3. Appel avec Jean : négocier le cas SNFC demain.",
  "incidents": "Incident à Cergy 5 Maintenance, nécessite une attention jusqu'à demain.",
  "call_recap": "Appel avec Jean pour négocier le cas SNCF demain.",
  "additional_info": "La note inclut un schéma de changement de relais de 40KV à 20KV."
}
"""


# add_data2db("/Users/tomamirault/Documents/Projects/p1-dty-rte/detection-notes/data/images/raw/IMG_3268.JPG")