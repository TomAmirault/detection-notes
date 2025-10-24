# Compare the similarity between two sets of entities extracted from texts, using weighting by entity category
# The main goal is to determine whether two texts refer to the same event

from rapidfuzz import fuzz


# Weighting based on the importance of the category
ENTITY_WEIGHTS = {
    "GEO": 0.30,
    "INFRASTRUCTURE": 0.15,
    "ACTOR": 0.15,
    "PHONE_NUMBER": 0.10,
    "EVENT": 0.10,
    "ABBREVIATION_UNKNOWN": 0.05,
    "DATETIME": 0.05,
    "OPERATING_CONTEXT": 0.05,
    "ELECTRICAL_VALUE": 0.05,
}

# Default similarity threshold
SIMILARITY_THRESHOLD = 70

# Minimum proportion of the total score required to conclude that two sets of entities describe the same event
PROPORTION = 0.6


def entity_similarity(a: str, b: str, threshold: int = SIMILARITY_THRESHOLD) -> bool:
    """
    Checks if two textual entities are similar.
    """
    return fuzz.token_set_ratio(a, b) >= threshold


def weighted_common_score(entities1: dict[str, list], entities2: dict[str, list], threshold: int = SIMILARITY_THRESHOLD) -> tuple[float, float]:
    """
    Calculates a weighted score proportional by category (only common categories are compared)
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
        
        # Compare each entity from the smaller list to all entities in the other list
        if len(list1) <= len(list2):
            smaller, larger = list1, list2
        else:
            smaller, larger = list2, list1

        for e1 in smaller:
            for e2 in larger:
                if entity_similarity(e1, e2, threshold):
                    matched_pairs += 1
                    break
        
        # Proportion based on the smaller list
        category_ratio = matched_pairs / len(smaller) 
        matched_weight += weight * category_ratio

    return matched_weight, total_weight



def same_event(entities1: dict[str, list], entities2: dict[str, list], threshold: int = SIMILARITY_THRESHOLD, proportion: int = PROPORTION) -> bool:
    """
    Determines whether two dictionaries of entities refer to the same event,
    using weighting and comparing only the common categories.
    """
    matched_weight, total_weight = weighted_common_score(entities1, entities2, threshold)

    if total_weight == 0:
        return False

    return matched_weight >= proportion * total_weight
