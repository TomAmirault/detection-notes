import re

def cleaning(text):
      # Remplace les sauts de ligne multiples par un seul saut de ligne
      text = re.sub(r"\n{2,}", "\n", text)
      # Nettoie les espaces multiples sur chaque ligne
      text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
      # Suppression des éléments markdown
      text = re.sub(r"[-*•#]", "", text)     # tirets, puces, astérisques, dièses
      text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # gras markdown
      text = re.sub(r"\*([^*]+)\*", r"\1", text)      # italique markdown
      text = re.sub(r"`([^`]+)`", r"\1", text)        # code markdown
      text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # liens markdown
      # Suppression des flèches "→ "
      text = text.replace("→ ", "")
      text = text.strip()
      return text