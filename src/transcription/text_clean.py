import unicodedata
import re

def text_clean(text):
    # Décompose les caractères accentués en base + accent, puis garde seulement la base
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', '', text)  
    return text