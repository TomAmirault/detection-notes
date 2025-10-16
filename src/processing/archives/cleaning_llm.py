import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

def cleaning(text_ocr: str) -> str:
      prompt = f"""Tu es un assistant de normalisation de notes manuscrites pour le dispatching électrique (RTE).
      Tu reçois du texte brut d’un OCR. Ton objectif est de produire une version canonique et stable du texte pour permettre une comparaison ligne à ligne avec SequenceMatcher.

      Contraintes (obligatoires)

      Préserver l’ordre des informations exactement tel qu’il apparaît dans l’entrée.

      Ne reclasse pas par thèmes.

      Ne réordonne aucune ligne.

      Une ligne = une information courte.

      N’introduis pas de nouvelles lignes inutiles.

      N’insère pas de lignes vides supplémentaires.

      Ne fusionne pas plusieurs infos distantes.

      Ne pas reformuler : garde les mots d’origine autant que possible.

      Corrige seulement les erreurs OCR manifestes (voir ci-dessous) si le contexte technique électrique l’exige.

      Nettoyage minimal mais déterministe :

      Supprime markdown/puces/symboles décoratifs en tête de ligne (**, *, -, •, []), tout en préservant la ligne.

      Normalise les espaces (un seul espace entre mots).

      Conserve la ponctuation utile ; retire les guillemets isolés parasites.

      Ne pas regrouper en catégories (pas de “Rappels:”, “Incidents:”…).

      Absolument aucune explication : renvoie uniquement le texte final.

      Erreurs OCR typiques à corriger (si et seulement si évident)

      Confusions lettres/chiffres : O↔0, I↔1, l↔1, Z↔2, S↔5, B↔8, G↔6, T↔7.

      Mots faux ou proches : “rebois”→“relais”, “plannig”→“planning”, “Maintenace”→“Maintenance”, “travau”→“travaux”.

      Mots collés/séparés : “conduitedes”→“conduite des”.

      Abréviations mal reconnues proches : corrige vers la forme officielle SI c’est évident.

      Accentuation manquante (é/è) : corrige si le mot est sans ambiguïté.

      Symboles parasites en début/fin de ligne : retire sans créer de nouvelle ligne.

      Abréviations officielles (référence)

      Abréviation	Signification
      RACR	Retour à la conduite des réseaux
      RDCR	Retrait de la conduite des réseaux
      TIR	Thermographie Infrarouge
      PO	Poste Opérateur
      RSD	Régime Surveillance Dispatching
      SUAV	Sous tension à vide
      MNV	Manœuvre
      PF	Point Figé
      CSS	Central Sous Station (SNCF)
      GEH	Groupe Exploitation Hydraulique
      PDM	Personnel de Manoeuve
      SMACC	Système Avancé pour la Commande de la Compensation
      HO	Heures Ouvrées
      BR	Bâtiment de Relayage
      GT	Groupe de Traction
      TST 	Travaux Sous Tension
      CCO	Chargé de Conduite
      FDE	File d'Essai
      DIFFB	Protection différentielle de barre
      RADA	Retrait Agile suite à Détection d'Anomalie
      TR	Transformateur
      RA	Réenclencheur Automatique
      CTS	Cycle Triphasé Supplémentaire
      CEX	Chargé d'Exploitation
      COSE	Centre Opérationnel du Système Electrique
      COSE-P	Centre Opérationnel du Système Electrique Paris
      RTE	Réseau de Transport d'Électricité
      Si l’entrée contient une variante très proche d’une abréviation officielle, corrige vers l’abréviation officielle sans développer (ex: “SMAC”→“SMACC”).
      Sinon, laisse tel quel.

      Exemples (respect absolu de l’ordre d’origine)
      Exemple 1

      Entrée OCR :

      **Rappel:** Prévenir M. Martin (ancien: 0766378217, nv: 0766378217)
      Appel avec Jear (Maintenace) pr vérif plannig travau
      Changement rebois 40kV->20kV


      Sortie (même ordre, 1 info = 1 ligne) :

      Prévenir M. Martin ancien: 0766378217 nouveau: 0766378217
      Appel avec Jean Maintenance vérifier planning travaux
      Changement relais 40kV -> 20kV

      Exemple 2

      Entrée OCR :

      RACN poste T4
      problème DIFFR sur TR 3
      Rappel -> contacter ccc
      SMAC déclenché hier sojr


      Sortie :

      RACR poste T4
      Problème DIFFB sur TR 3
      Contacter CCO
      SMACC déclenché hier soir

      Exemple 3

      Entrée OCR :

      rdcr po 74
      TIR demain matn
      HO 9h-12h
      pf confirmé


      Sortie :

      RDCR PO T4
      TIR demain matin
      HO 9h-12h
      PF confirmé

      Exemple 4

      Entrée OCR :

      Appel JN SNCF pr CSS
      numero nv 0666 999 888
      MNV poste T2
      Suivis: PDM dispo


      Sortie :

      Appel Jean SNCF pour CSS
      Nouveau numéro 0666999888
      MNV poste T2
      PDM disponible

      Exemple 5

      Entrée OCR :

      "Revois" protectn TR -> OCR mauvais
      COSE P Paris dmain matin
      SMAC actif
      ``>
      Sortie :


      Protection TR revue
      COSE-P Paris demain matin
      SMACC actif


      ### Exemple 6 (bruit markdown et symboles)
      Entrée OCR :


      Envoyer CR à CM à 16h

      Vote

      T4 T2 T3

      Sortie :


      Envoyer CR à CM à 16h
      T4 T2 T3


      ### Exemple 7 (éviter les splits/merges inutiles)
      Entrée OCR :


      Appel avec Pierre pour TR2 vérif DIFFB demain 8h puis CM 16h

      Sortie (une seule ligne car info compacte, pas d’éclatement arbitraire) :


      Appel avec Pierre pour TR2 vérifier DIFFB demain 8h puis CM 16h


      ## Réponse finale
      Renvoie uniquement le texte final normalisé, avec le **même ordre** que l’entrée et sans aucun saut de ligne.
      Dans aucun cas, il dois y avoir une ligne vide entre deux lignes d’information.
      \"\"\"{text_ocr}\"\"\"
      """

      response = client.chat.complete(
      model="mistral-large-latest",
      messages=[{"role": "user", "content": prompt}]
      )

      synthetised_text = response.choices[0].message.content

      return synthetised_text.strip()
