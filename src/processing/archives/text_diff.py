### Une fois qu'on sait qu'on va ajouter le texte, et à quelle note_id, on detecte le texte qui a été ajouté en plus par rapport à avant

def get_added_text(old_text: str, new_text: str) -> str:
    min_len = min(len(old_text), len(new_text))
    if new_text[:min_len] == old_text[:min_len]:
        return new_text[min_len:]
    return new_text
