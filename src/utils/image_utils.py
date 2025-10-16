import base64

def encode_image(path: str) -> str:
    """
    Encode une image en base64 à partir de son chemin.
    Retourne la chaîne encodée (utf-8).
    """
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
