from rapidfuzz import fuzz

# Pondération par importance du type d'entité 
ENTITY_WEIGHTS = {
    "GEO": 0.30,
    "INFRASTRUCTURE": 0.15,
    "ACTOR": 0.15,
    "PHONE_NUMBER": 0.15,
    "EVENT": 0.10,
    "DATETIME": 0.05,
    "OPERATING_CONTEXT": 0.05,
    "ELECTRICAL_VALUE": 0.05
}

def entity_similarity(a, b, threshold=70):
    """
    Similitude entre deux entités
    """
    return fuzz.token_set_ratio(a, b) >= threshold


def weighted_common_score(entities1, entities2, similarity_threshold=70):
    """
    Calcule un score pondéré proportionnel par catégorie :
    - Seules les catégories communes sont comparées
    - Le poids d'une catégorie est multiplié par la proportion d'entités qui matchent
    """
    total_weight = 0
    matched_weight = 0

    common_types = set(entities1.keys()) & set(entities2.keys())

    for etype in common_types:
        list1 = entities1.get(etype, [])
        list2 = entities2.get(etype, [])
        if not list1 or not list2:
            continue

        weight = ENTITY_WEIGHTS.get(etype, 0.01)
        total_weight += weight

        matched_pairs = 0
        
        # On compare chaque entité de la plus petite liste à toutes celles de l'autre
        if len(list1) <= len(list2):
            smaller, larger = list1, list2
        else:
            smaller, larger = list2, list1

        for e1 in smaller:
            for e2 in larger:
                if entity_similarity(e1, e2, similarity_threshold):
                    matched_pairs += 1
                    break
        
        # Proportion basée sur la plus petite liste 
        category_ratio = matched_pairs / len(smaller) # compris entre 0 et 1
        matched_weight += weight * category_ratio

    return matched_weight, total_weight



def same_event(entities1, entities2, similarity_threshold=70, proportion=0.60):
    """
    Détermine si deux dictionnaires d'entités parlent du même événement,
    avec pondération et comparaison seulement sur les catégories communes.
    """
    matched_weight, total_weight = weighted_common_score(
        entities1, entities2, similarity_threshold
    )

    if total_weight == 0:
        return False, 0.0, 0.0

    return matched_weight >= proportion * total_weight, matched_weight, proportion * total_weight


if __name__ == '__main__':
    from llm_extraction import extract_entities
    text1 = "Coupure au niveau de la ligne Trappes-Deauville à 14h: contacter le PDM."
    text2 = "MNV réalisé sur le N-1 entre Deauville et Trouville."

    result, common_count, min_required = same_event(
        extract_entities(text1), extract_entities(text2), similarity_threshold=50, proportion=0.60)

    print(extract_entities(text1))
    print(extract_entities(text2))
    print(result)
    print(common_count)
    print(min_required)
