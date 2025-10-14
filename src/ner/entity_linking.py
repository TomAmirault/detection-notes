from spacy_model import extraire_entites
from rapidfuzz import fuzz


def entity_similarity(a, b, threshold=60):
    return fuzz.ratio(a, b)  # >= threshold


def count_common_entities(text1, text2, entity_types=None, similarity_threshold=60):
    entities1 = extraire_entites(text1)["entites"]
    entities2 = extraire_entites(text2)["entites"]

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


def total_entities_count(text, entity_types=None):
    entities = extraire_entites(text)["entites"]
    if entity_types:
        return sum(len(entities.get(t, [])) for t in entity_types)
    else:
        return sum(len(lst) for lst in entities.values())


def same_event(text1, text2, entity_types=None, similarity_threshold=60, proportion=0.75):
    common_count = count_common_entities(
        text1, text2, entity_types, similarity_threshold)
    total_count = total_entities_count(
        text1, entity_types) + total_entities_count(text2, entity_types)

    if total_count == 0:
        return False, 0, 0

    min_common = max(1, round(proportion * (total_count / 2)))

    return common_count >= min_common, common_count, min_common


# text1 = "Coupure au niveau de la ligne Charles-Trappe à 14h: contacter le PDM."
# text2 = "MNV réalisé par le PDM sur le N-1 entre Paris et Marseille vers 14:00."

# result, common_count, min_required = same_event(
#     text1, text2, entity_types=None, similarity_threshold=50, proportion=0.75)

# print(extraire_entites(text1)["entites"])
# print(extraire_entites(text2)["entites"])
# print(result)
# print(common_count)
# print(min_required)

print(entity_similarity("14h", "14:00"))
