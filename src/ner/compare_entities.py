from rapidfuzz import fuzz


def entity_similarity(a, b, threshold=60):
    return fuzz.ratio(a, b) >= threshold


def count_common_entities(entities1, entities2, entity_types=None, similarity_threshold=50):
    common_count = 0

    # Si pas de listes d'entités spécifiée, on compare sur tous
    if entity_types is None:
        entity_types = list(set(entities1.keys()) | set(entities2.keys()))

    for etype in entity_types:
        list1 = entities1.get(etype, [])
        list2 = entities2.get(etype, [])
        for e1 in list1:
            for e2 in list2:
                if entity_similarity(e1, e2, similarity_threshold):
                    common_count += 1

    return common_count


def total_entities_count(entities, entity_types=None):
    if entity_types:
        return sum(len(entities.get(t, [])) for t in entity_types)
    else:
        return sum(len(lst) for lst in entities.values())


def same_event(entities1, entities2, entity_types=None, similarity_threshold=50, proportion=0.75):
    common_count = count_common_entities(
        entities1, entities2, entity_types, similarity_threshold)
    total_count = total_entities_count(
        entities1, entity_types) + total_entities_count(entities2, entity_types)

    if total_count == 0:
        return False, 0, 0

    min_common = max(1, round(proportion * (total_count / 2)))

    return common_count >= min_common, common_count, min_common


if __name__ == '__main__':
    from spacy_model import extract_entities
    text1 = "Coupure au niveau de la ligne Trappes-Deauville à 14h: contacter le PDM."
    text2 = "MNV réalisé par le PDM sur le N-1 entre Deauville et Trappes."

    result, common_count, min_required = same_event(
        extract_entities(text1), extract_entities(text2), entity_types=None, similarity_threshold=50, proportion=0.75)

    print(extract_entities(text1))
    print(extract_entities(text2))
    print(result)
    print(common_count)
    print(min_required)
