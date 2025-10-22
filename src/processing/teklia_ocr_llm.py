import requests
import os, re
from mistralai import Mistral
from dotenv import load_dotenv

# Charger clé API
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")
teklia_api_key = os.getenv("TEKLIA_API_KEY")

client = Mistral(api_key=mistral_api_key)


# --- Pré-process : collapse des lignes de continuation ---
def pre_collapse_continuations(text: str) -> str:
    """
    Normalize line starts but do NOT collapse any lines together.

    - Strip surrounding whitespace.
    - Remove common bullet markers and checkbox markers at start of line.
    - Remove explicit continuation markers (↳, >) at start of line.
    - Keep every (non-empty) input line as its own output line.
    """
    out_lines = []
    for line in text.splitlines():
        if not line.strip():
            continue
        s = line.strip()

        # Remove checkbox markers like "[ ]" or "[x]"
        s = re.sub(r'^\[\s*[xX]?\s*\]\s*', '', s)

        # If line starts with common bullet markers, strip them but keep the line
        if s.startswith(('•', '-', '*', '\u2022')):
            s = re.sub(r'^[\u2022\-\*\u2023\u25E6\u2043\u2219\u25AA\u25CF\s]+', '', s).strip()

        # Remove explicit continuation markers at line start (↳, >) but do NOT merge lines
        s = re.sub(r'^[\u21b3>]+\s*', '', s)

        if s:
            out_lines.append(s)

    return "\n".join(out_lines)


# --- Post-process : normalisations déterministes ---
def postprocess_normalized(text: str) -> str:
    # 0) Nettoyages durs de balises ou code fences
    text = text.replace("```", "")
    text = text.replace("<<<", "").replace(">>>", "")

    # 1) Normalisation espaces globaux
    text = re.sub(r"[ \t]+", " ", text)

    # 2) Split lignes + filtres
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # a) supprimer les explications style "(Aucune information exploitable ...)"
            # Bullets - keep as separate lines; strip common bullet markers
            if stripped.startswith(('•', '-', '*', '\u2022')):
                stripped = re.sub(r'^[\u2022\-\*\u2023\u25E6\u2023\u2043\u2219\u25AA\u25CF\s]+', '', stripped).strip()
                lines.append(stripped)
                continue
        # b) ignorer les lignes vides sémantiquement
        if not re.search(r"[A-Za-zÀ-ÿ0-9]", line):
            continue

        # c) normalisations déterministes utiles
        line = re.sub(r"\b(\d{1,2})\s*h\b", r"\1h", line)             # 16 h -> 16h
        line = re.sub(r"\b(\d+)\s*RV\b", r"\1kV", line)               # 20RV -> 20kV
        line = re.sub(r"\b0\d(?:\s?\d{2}){4}\b",
                      lambda m: m.group(0).replace(" ", ""), line)    # 07 66 .. -> 0766..

        # Si la ligne contient ':' on peut vouloir mettre la suite sur la ligne suivante.
        # Exemples: "Entrée : texte" -> ["Entrée:", "texte"]
        # Mais on évite de splitter les motifs de type heure (16:30) ou les URLs (http://...),
        # ou autres schémas 'scheme:'.
        if ':' in line:
            # heuristiques pour ne PAS splitter
            if re.search(r"\b\d{1,2}:\d{2}\b", line):
                # contient une heure type 16:30 -> ne pas splitter
                lines.append(line)
            elif re.search(r"https?://|\w+://", line.lower()):
                # url ou schéma -> ne pas splitter
                lines.append(line)
            else:
                head, tail = line.split(':', 1)
                head = head.strip()
                tail = tail.strip()
                if head and tail:
                    # garder le ':' à la fin de la première ligne
                    lines.append(head + ':')
                    lines.append(tail)
                else:
                    lines.append(line)
        else:
            lines.append(line)

    # 3) Supprimer les lignes “admin” et bruit numérique court
    final = []
    for line in lines:
        if re.fullmatch(r"(?i)\s*vote(\s+\d+)?\s*", line):
            continue
        if re.fullmatch(r"(?i)\s*note(\s+\d+)?\s*", line):
            continue
        if re.fullmatch(r"\d{1,3}", line):
            continue
        if line.lower() == "none":
            continue
        final.append(line)

    # 4) Join + trim
    out = "\n".join(final).strip()
    return out


# --- Fonction principale ---
def image_transcription(image_path: str) -> str:

    # 1. OCR brut
    url = 'https://atr.ocelus.teklia.com/api/v1/transcribe/'
    headers = {
        'API-Key': teklia_api_key
    }
    files = {
        'image': open(image_path, 'rb')
    }
    params = {
        'language': 'fr'
    }
    response = requests.post(url, headers=headers, files=files, params=params)

    if response.status_code != 200:
        raise Exception(f"Erreur OCR Teklia : {response.status_code} - {response.text}")
    
    ocr_text = ""
    confidence_per_line = []
    char_per_line = []
    for line in response.json()['results']:
        if line['confidence'] >= 0.5:
            ocr_text += line['text'] + "\n"
            confidence_per_line.append(line['confidence'])
            char_per_line.append(len(line['text']))
        print(line['confidence'], line['text'])

    if len(ocr_text) > 0:
        ocr_text_confidence = sum(conf*nb_chars for conf, nb_chars in zip(confidence_per_line, char_per_line)) / max(1, sum(char_per_line))
    else:
        ocr_text_confidence = 0.5

    print("=== OCR brut ===\n", ocr_text, "\nScore de confiance OCR moyen pondéré :", ocr_text_confidence)

    # 2. Pré-process
    ocr_text = pre_collapse_continuations(ocr_text)
    # print("=== Après pré-process ===\n", ocr_text)

    # 3. Prompt LLM
    prompt = f"""Tu es un assistant de normalisation pour des notes manuscrites RTE.
    Entrée : texte brut d’un OCR (peut contenir du markdown, des puces, des retours ligne erratiques, des erreurs).
    Sortie : texte canonique, stable, prêt pour une comparaison ligne-à-ligne.

    RÈGLES OBLIGATOIRES (déterministes)
    1) Préserve exactement l’ORDRE d’apparition des informations.
    - Ne reclasse pas par thèmes, ne réordonne pas les lignes.
    2) 1 information = 1 ligne.
    - Ne crée PAS de lignes vides.
    - Ne fusionne PAS des lignes éloignées.
    3) Nettoyage MINIMAL :
    - Supprime les préfixes décoratifs uniquement en DÉBUT de ligne : **, *, -, •, [], >, “## …”, “# …”.
    - Normalise les espaces (un seul espace entre mots).
    - Supprime les guillemets parasites isolés en DÉBUT/FIN de ligne.
    4) Corrections OCR : uniquement si ÉVIDENT dans le contexte technique électrique.
    - Confusions lettres/chiffres : O↔0, I↔1, l↔1, Z↔2, S↔5, B↔8, G↔6, T↔7.
    - Mots proches/faux : rebois→relais, plannig→planning, Maintenace→Maintenance, travau→travaux.
    - Mots collés/séparés : conduitedes→conduite des.
    - Abréviations mal reconnues proches → corrige vers la FORME OFFICIELLE si c’est manifestement ça.
    - Accentuation manquante (é/è) → corrige si non ambigu.
    5) Abréviations officielles (ne PAS développer, garder la forme abrégée ; corrige une variante proche vers la forme officielle) :
    RACR, RDCR, TIR, PO, RSD, SUAV, MNV, PF, CSS, GEH, PDM, SMACC, HO, BR, GT, TST, CCO, FDE, DIFFB, RADA, TR, RA, CTS, CEX, COSE, COSE-P, RTE.
    6) Éléments administratifs peu informatifs (seuls sur leur ligne) : “Note X”, “Vote”, “## Vote”, “### Note” → SUPPRIME la ligne entière si elle n’apporte aucune info liée à l’exploitation (ne pas supprimer si la ligne contient une action/heure/poste/etc.).
    7) Formats déterministes :
    - Fléchage “->” sans espaces autour si déjà correct ; sinon normalise en “ -> ” (avec espaces) pour la lisibilité.
    - Heures : “16 h”, “16h” → normalise en “16h” sans espace.
    - Numéros de téléphone : supprimer les espaces → “0766378217”.
    - Tensions : normaliser en “kV” (ex: 20RV→20kV si clair que c’est kV).
    8) STRICT : ne jamais ajouter de texte comme “None”, pas de commentaires ou d’explications. Ne rien inventer.

    ERREURS OCR TYPIQUES (référence mentale)
    - O↔0, I↔1, l↔1, Z↔2, S↔5, B↔8, G↔6, T↔7
    - rebois→relais, plannig→planning, Maintenace→Maintenance, travau→travaux
    - conduitedes→conduite des, “COSE P”→“COSE-P”, “SMAC”→“SMACC”
    - kV mal lu en “RV”, “Joan”→“Jean” si contexte francophone courant

    EXEMPLES (respect absolu de l’ordre d’entrée)

    Exemple A — entrées avec mini fautes :
    Entrée :
    **Rappel:** prévenir M. Martin (ancien: 0766 37 0247)
    nouveau: 07 66 37 8247
    Appel privé avec Joan (Maintenace), vérifier
    planning travau
    > Confirmation travaux
    demain 8h
    T4 T2 T3
    Envoyer CR à CM à 16 h

    Sortie attendue :
    Prévenir M. Martin ancien: 0766370247
    Nouveau: 0766378247
    Appel privé avec Jean Maintenance vérifier 
    planning travaux
    Confirmation travaux 
    demain 8h
    T4 T2 T3
    Envoyer CR à CM à 16h

    Exemple B — lignes administratives à supprimer + corrections abrégées :
    Entrée:
    28/09/2025
    Note 1
    Changement rebais 40RV -> 20RV (secteur T4)
    Vote
    Envoyer CR à CM à 16 h

    Sortie:
    28/09/2025
    Changement relais 40kV -> 20kV (secteur T4)
    Envoyer CR à CM à 16h

    Exemple C — bruits markdown et concaténation de continuation :
    Entrée:
    - [ ] *Envoyer CR à CM à 16h*
    ## Vote
    **T4 T2 T3**

    Sortie:
    Envoyer CR à CM à 16h
    T4 T2 T3

    Exemple D — abréviations proches et fautes évidentes :
    Entrée:
    RACN poste T4
    problème DIFFR sur TR 3
    SMAC déclenché hier sojr

    Sortie:
    RACR poste T4
    Problème DIFFB sur TR 3
    SMACC déclenché hier soir

    RÉPONSE FINALE (obligatoire)
- Si l'entrée ne contient aucune information exploitable (par exemple, juste un point ".") : renvoie une CHAÎNE VIDE (exactement "").
- Sinon, renvoie UNIQUEMENT le texte final normalisé, sans balises, sans ``` et sans commentaires.
Contenu à traiter :
<<<
{ocr_text}
>>>
    """
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    clean_text = response.choices[0].message.content.strip()
    # print("=== Réponse LLM brute ===\n", clean_text)

    # 4. Post-process
    clean_text = postprocess_normalized(clean_text)
    print("=== LLM après post-process ===\n", clean_text)

    return ocr_text, clean_text, ocr_text_confidence
