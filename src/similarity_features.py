from rapidfuzz import fuzz

from normalization import (
    normalize_business_name,
    normalize_address,
    normalize_compact,
    normalize_tokens,
    extract_digits
)


def name_token_sort(a, b):
    a = normalize_business_name(a)
    b = normalize_business_name(b)

    if not a or not b:
        return 0.0

    return fuzz.token_sort_ratio(a, b)


def name_token_set(a, b):
    a = normalize_business_name(a)
    b = normalize_business_name(b)

    if not a or not b:
        return 0.0

    return fuzz.token_set_ratio(a, b)


def name_character(a, b):
    a = normalize_compact(a)
    b = normalize_compact(b)

    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b)


def name_partial(a, b):
    a = normalize_compact(a)
    b = normalize_compact(b)

    if not a or not b:
        return 0.0

    return fuzz.partial_ratio(a, b)


def name_jaccard(a, b):
    tokens_a = set(normalize_tokens(a))
    tokens_b = set(normalize_tokens(b))

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


# ---------------- ADDRESS ----------------

def address_token_sort(a, b):
    a = normalize_address(a)
    b = normalize_address(b)

    if not a or not b:
        return 0.0

    return fuzz.token_sort_ratio(a, b)


def address_token_set(a, b):
    a = normalize_address(a)
    b = normalize_address(b)

    if not a or not b:
        return 0.0

    return fuzz.token_set_ratio(a, b)


def address_character(a, b):
    a = normalize_address(a)
    b = normalize_address(b)

    a = a.replace(" ", "")
    b = b.replace(" ", "")

    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b)


def address_partial(a, b):
    a = normalize_address(a)
    b = normalize_address(b)

    if not a or not b:
        return 0.0

    return fuzz.partial_ratio(a, b)


def address_jaccard(a, b):
    tokens_a = set(normalize_tokens(a))
    tokens_b = set(normalize_tokens(b))

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def digit_similarity(a, b):
    digits_a = extract_digits(a)
    digits_b = extract_digits(b)

    if not digits_a or not digits_b:
        return 0.0

    return fuzz.ratio(digits_a, digits_b)


def country_match(a, b):
    if not a or not b:
        return 0

    return int(
        str(a).strip().lower()
        == str(b).strip().lower()
    )


def calculate_features(
    name_a,
    address_a,
    country_a,
    name_b,
    address_b,
    country_b
):

    return {
        "name_token_sort":
            name_token_sort(name_a, name_b),

        "name_token_set":
            name_token_set(name_a, name_b),

        "name_character":
            name_character(name_a, name_b),

        "name_partial":
            name_partial(name_a, name_b),

        "name_jaccard":
            name_jaccard(name_a, name_b),

        "address_token_sort":
            address_token_sort(address_a, address_b),

        "address_token_set":
            address_token_set(address_a, address_b),

        "address_character":
            address_character(address_a, address_b),

        "address_partial":
            address_partial(address_a, address_b),

        "address_jaccard":
            address_jaccard(address_a, address_b),

        "address_digit_similarity":
            digit_similarity(address_a, address_b),

        "country_match":
            country_match(country_a, country_b)
    }


if __name__ == "__main__":

    name1 = "Payne Enterprises"
    name2 = "PAYNE-ENRTPRMISES"

    address1 = "3315 Fremont Street, Peoria, IL"
    address2 = "3315 Fremont Saint, Peoria, IL"

    features = calculate_features(
        name1,
        address1,
        "US",
        name2,
        address2,
        "US"
    )

    print("\nFEATURES")
    print("-" * 50)

    for key, value in features.items():
        print(f"{key}: {value}")